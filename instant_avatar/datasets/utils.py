import torch
import numpy as np

import json
import glob
import os
import copy


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def transl_image(img_info, rgb, mask):
    h, w, ori_h, ori_w = img_info

    x = min(ori_w, w)
    y = min(ori_h, h)

    new_rgb = np.zeros([h, w, 3])
    new_mask = np.zeros([h, w])

    new_rgb[:y, :x, :] = rgb[:y, :x, :]
    if len(mask.shape) > 2:
        new_mask[:y, :x] = mask[:y, :x, 0]
    else:
        new_mask[:y, :x] = mask[:y, :x]

    return new_rgb, new_mask


def load_annots(data_dir, view_id):
    annot = np.load(f"{data_dir}/annots.npy", allow_pickle=True)

    dat = annot.tolist()
    cams = dat["cams"]

    K = cams["K"][view_id]
    D = cams["D"][view_id]

    R = cams["R"][view_id]
    T = cams["T"][view_id] * 1e-3

    cam_data = {}
    cam_data["K"] = K
    cam_data["D"] = D
    cam_data["R"] = R
    cam_data["T"] = T
    cam_data["width"] = 1285 #1024
    cam_data["height"] = 940 #1024

    return cam_data


def load_smpl_param_from_npy(npy_file):
    dat = np.load(npy_file, allow_pickle=True)

    dat = dat.tolist()

    poses = dat["poses"][0]
    rh = dat["Rh"][0]
    th = dat["Th"][0]
    shapes = dat["shapes"][0]

    smpl_data = {}

    smpl_data["poses"] = poses[3:]

    smpl_data["gl_rot"] = rh

    smpl_data["gl_transl"] = th

    smpl_data["shapes"] = shapes

    return smpl_data


def load_smpl_param_from_json(json_file):
    with open(json_file) as fp:
        data = json.load(fp)
        fp.close()

    data = data[0]
    poses = np.array(data["poses"][0])
    rh = np.array(data["Rh"][0])
    th = np.array(data["Th"][0])
    shapes = np.array(data["shapes"][0])

    smpl_data = {}

    smpl_data["poses"] = poses[3:]

    smpl_data["gl_rot"] = rh

    smpl_data["gl_transl"] = th

    smpl_data["shapes"] = shapes

    return smpl_data


def load_smpl_param(param_path, fids, json_flag=True):
    # smpl_params = dict(np.load(str(path)))
    shapes = []
    poses = []
    gl_rot = []
    gl_transl = []

    for fid in fids:
        if json_flag:
            smpl_dir = f"{param_path}/{fid:06d}.json"
            smpl_data = load_smpl_param_from_json(smpl_dir)
        else:
            smpl_dir = f"{param_path}/{fid}.npy"
            smpl_data = load_smpl_param_from_npy(smpl_dir)

        shape = smpl_data["shapes"]
        pose = smpl_data["poses"]
        rot = smpl_data["gl_rot"]
        t = smpl_data["gl_transl"]


        shapes.append(shape)
        poses.append(pose)
        gl_rot.append(rot)
        gl_transl.append(t)

    np_betas = np.asarray(shapes).reshape(-1, 10)
    np_poses = np.asarray(poses).reshape(-1, 69)

    np_gl_rot = np.asarray(gl_rot).reshape(-1, 3)
    np_gl_transl = np.asarray(gl_transl).reshape(-1, 3)

    return {
        "betas": np_betas[0].astype(np.float32).reshape(1, 10),
        "body_pose": np_poses.astype(np.float32),
        "global_orient": np_gl_rot.astype(np.float32),
        "transl": np_gl_transl.astype(np.float32),
    }


def get_view_data(root_dir, cam_id, start, end, skip, json_flag=True):
    img_lists = sorted(glob.glob(f"{root_dir}/images/{cam_id:02d}/*.jpg"))[
        start:end:skip
    ]

    msk_lists = sorted(glob.glob(f"{root_dir}/mask/{cam_id:02d}/*.png"))[start:end:skip]


    if len(img_lists) < 1:
        img_lists = sorted(glob.glob(f"{root_dir}/images/{cam_id:03d}/*.jpg"))[
         start:end:skip
        ]

        msk_lists = sorted(glob.glob(f"{root_dir}/mask/{cam_id:03d}/*.png"))[start:end:skip]


    fids = []
    for file in img_lists:
        fid = int(os.path.basename(file).split(".")[0])
        fids.append(fid)

    if json_flag:
        param_path = f"{root_dir}/smpl_params_standard"
    else:
        param_path = f"{root_dir}/params"

    smpl_params = load_smpl_param(param_path, fids, json_flag=json_flag)

    view_lists = [cam_id for i in range(len(img_lists))]

    return (img_lists, msk_lists, smpl_params, view_lists)


def combine_smpl_params(data):
    rot = []
    pose = []
    transl = []

    for item in data:
        rot.append(item["global_orient"])
        pose.append(item["body_pose"])
        transl.append(item["transl"])

    #
    return {
        "betas": data[0]["betas"],
        "body_pose": np.concatenate(pose, 0),
        "global_orient": np.concatenate(rot, 0),
        "transl": np.concatenate(transl, 0),
    }


def get_camera_params(root, vids, downscale, processed=False):
    cams = {}

    for vid in vids:
        tmp_data = {}
        camera = load_annots(root, vid)
        K = camera["K"]

        ori_K = copy.copy(K)
        D = camera["D"]
        w2c = np.eye(4)

        w2c[:3, :3] = camera["R"]
        w2c[:3, 3] = camera["T"].reshape(-1)
        c2w = np.linalg.inv(w2c)

        cx = K[0, 2]
        cy = K[1, 2]

        new_w = int(cx * 2)
        new_h = int(cy * 2)


        if processed:
            width = new_w
            height = new_h
        else:
            width = camera["width"]
            height = camera["height"]

        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        if downscale > 1:
            width = int(width / downscale)
            height = int(height / downscale)
            K[:2] /= downscale

        new_K = K
        focal_x = K[0, 0]
        focal_y = K[1, 1]

        tmp_data["K"] = new_K
        tmp_data["ori_K"] = ori_K
        tmp_data["D"] = D
        tmp_data["R"] = R
        tmp_data["T"] = T
        tmp_data["width"] = width
        tmp_data["height"] = height

        tmp_data["new_w"] = new_w
        tmp_data["new_h"] = new_h

        tmp_data["c2w"] = c2w

        cams[vid] = tmp_data

    return cams


def get_ray_directions(H, W):
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    xy = np.stack([x, y, np.ones_like(x)], axis=-1)
    return xy


def make_rays(K, c2w, H, W):
    xy = get_ray_directions(H, W).reshape(-1, 3).astype(np.float32)
    d_c = xy @ np.linalg.inv(K).T
    d_w = d_c @ c2w[:3, :3].T
    d_w = d_w / np.linalg.norm(d_w, axis=1, keepdims=True)
    o_w = np.tile(c2w[:3, 3], (len(d_w), 1))
    o_w = o_w.reshape(H, W, 3)
    d_w = d_w.reshape(H, W, 3)
    return o_w.astype(np.float32), d_w.astype(np.float32)
