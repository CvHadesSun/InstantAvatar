from pathlib import Path
import os
import numpy as np
import hydra
import torch
import cv2
import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from PIL import Image

from torch.multiprocessing import Manager
from instant_avatar.datasets.utils import (
    combine_smpl_params,
    get_view_data,
    get_camera_params,
    make_rays,
)


class MonoCapDataset(torch.utils.data.Dataset):
    def __init__(self, root, subject, split, opt):
        self.cams = get_camera_params(root, opt.view_id, opt.downscale, processed=False)
        self.processed_raw = opt.processed
        # prepare image and mask
        start = opt.start
        end = opt.end + 1
        skip = opt.get("skip", 1)

        self.img_lists = []
        self.msk_lists = []

        smpl_params_lists = []

        self.view_lists = []

        for vid in opt.view_id:
            _img_list, _msk_list, smpl_param, view_list = get_view_data(
                root, vid, start, end, skip, json_flag=False
            )
            self.img_lists.extend(_img_list)
            self.msk_lists.extend(_msk_list)
            self.view_lists.extend(view_list)
            smpl_params_lists.append(smpl_param)

        self.smpl_params = combine_smpl_params(smpl_params_lists)

        self.split = split
        self.downscale = opt.downscale
        self.near = opt.get("near", None)
        self.far = opt.get("far", None)

        if split == "train":
            self.sampler = hydra.utils.instantiate(opt.sampler)

        self.cam_extent = 3.0

        self.cache_dict = Manager().dict()

    def get_SMPL_params(self):
        return {k: torch.from_numpy(v.copy()) for k, v in self.smpl_params.items()}

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        if idx in self.cache_dict:
            return self.cache_dict[idx]
        else:
            sample = self.get_data(idx)
            self.cache_dict[idx] = sample
            return sample

    def get_data(self, idx):
        self.image_shape = (cam["height"], cam["width"])

        cam_id = self.view_lists[idx]
        cam = self.cams[cam_id]

        K = cam["K"]

        img = cv2.imread(self.img_lists[idx])
        # msk = np.load(self.msk_lists[idx])
        msk = cv2.imread(self.msk_lists[idx], -1) / 255.0
        # img = cv2.undistort(img, ori_K, D)
        # msk = cv2.undistort(msk, ori_K, D)

        if self.downscale > 1:
            img = cv2.resize(
                img, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
            )
            msk = cv2.resize(
                msk, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
            )

        img = (img[..., :3] / 255).astype(np.float32)
        msk = msk.astype(np.float32)
        # msk = np.ones_like(msk)
        # apply mask
        if len(msk.shape) > 2:
            msk = msk[:, :, 0]
        if self.split == "train":
            bg_color = np.random.rand(*img.shape).astype(np.float32)
            img = img * msk[..., None] + (1 - msk[..., None]) * bg_color
        else:
            bg_color = np.ones_like(img).astype(np.float32)
            img = img * msk[..., None] + (1 - msk[..., None])

        rays_o_, rays_d_ = make_rays(K, cam["c2w"], cam["height"], cam["width"])

        if self.split == "train":
            (msk, img, rays_o, rays_d, bg_color) = self.sampler.sample(
                msk, img, rays_o_, rays_d_, bg_color
            )
        else:
            rays_o = self.rays_o.reshape(-1, 3)
            rays_d = self.rays_d.reshape(-1, 3)
            img = img.reshape(-1, 3)
            msk = msk.reshape(-1)

        datum = {
            # NeRF
            "rgb": img.astype(np.float32),
            "rays_o": rays_o,
            "rays_d": rays_d,
            # SMPL parameters
            "betas": self.smpl_params["betas"][0],
            "global_orient": self.smpl_params["global_orient"][idx],
            "body_pose": self.smpl_params["body_pose"][idx],
            "transl": self.smpl_params["transl"][idx],
            # auxiliary
            "alpha": msk,
            "bg_color": bg_color,
            "idx": idx,
        }

        if self.near is not None and self.far is not None:
            datum["near"] = np.ones_like(rays_d[..., 0]) * self.near
            datum["far"] = np.ones_like(rays_d[..., 0]) * self.far
        else:
            # distance from camera (0, 0, 0) to midhip
            # TODO: we could replace it with bbox in the canonical space
            dist = np.sqrt(np.square(self.smpl_params["transl"][idx]).sum(-1))
            datum["near"] = np.ones_like(rays_d[..., 0]) * (dist - 1)
            datum["far"] = np.ones_like(rays_d[..., 0]) * (dist + 1)

        return datum


class MonoCapDataModule(pl.LightningDataModule):
    def __init__(self, opt, **kwargs):
        super().__init__()

        data_dir = Path(hydra.utils.to_absolute_path(opt.dataroot))
        for split in ("train", "val", "test"):
            dataset = MonoCapDataset(data_dir, opt.subject, split, opt.get(split))
            setattr(self, f"{split}set", dataset)
        self.opt = opt

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(
                self.trainset,
                shuffle=True,
                num_workers=self.opt.train.num_workers,
                persistent_workers=True and self.opt.train.num_workers > 0,
                pin_memory=True,
                batch_size=1,
            )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valset"):
            return DataLoader(
                self.valset,
                shuffle=False,
                num_workers=self.opt.val.num_workers,
                persistent_workers=True and self.opt.val.num_workers > 0,
                pin_memory=True,
                batch_size=1,
            )
        else:
            return super().test_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testset"):
            return DataLoader(
                self.testset,
                shuffle=False,
                num_workers=self.opt.test.num_workers,
                persistent_workers=True and self.opt.test.num_workers > 0,
                pin_memory=True,
                batch_size=1,
            )
        else:
            return super().test_dataloader()
