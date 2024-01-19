import glob
import os
import cv2
import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn

# from torch.cuda.amp import custom_fwd
# from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.utils.data import DataLoader

from instant_avatar.metric.image_utils import psnr
from instant_avatar.metric.lpipsPyTorch import lpips
from instant_avatar.metric.loss_utils import ssim
from tqdm import tqdm
import numpy as np
import json


@hydra.main(config_path="./confs", config_name="SNARF_NGP_refine")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    model = model.cuda()
    model.eval()

    checkpoints = sorted(glob.glob("checkpoints/*.ckpt"))
    print("Resume from", checkpoints[-1])
    checkpoint = torch.load(checkpoints[-1])
    model.load_state_dict(checkpoint["state_dict"])

    dataloader = DataLoader(
        datamodule.testset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    ssims = []
    psnrs = []
    lpipss = []

    result = {}

    with torch.inference_mode():
        for i, batch in tqdm(enumerate(dataloader)):
            new_batch = {}

            for k in batch.keys():
                if k == "image_name" or k == "nvdiff_pkg":
                    continue
                new_batch[k] = batch[k].cuda()

            height = batch["height"]
            width = batch["width"]

            render, _, alpha, _ = model.render_image_fast(batch, (height, width))
            # render [1,c,h,w]

            render = torch.clamp(render, 0.0, 1.0).permute(0, 3, 1, 2).cuda() * 255
            gt_img = batch["rgb"][0].permute(2, 0, 1).unsqueeze(0)[:, :3, :, :].cuda()
            gt_img = torch.clamp(gt_img, 0.0, 1.0) * 255

            ssims.append(ssim(render, gt_img))
            psnrs.append(psnr(render, gt_img))
            lpipss.append(lpips(render, gt_img, net_type="vgg"))

            if 0:
                render_img = render.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                gt_img_ = gt_img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

                res_img = cv2.hconcat([render_img[0], gt_img_[0]])
                cv2.imwrite(f"{i:04d}_test.png", res_img)

        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")

        result["ssim"] = f"{torch.tensor(ssims).mean():.5f}"
        result["psnr"] = f"{torch.tensor(psnrs).mean():.5f}"
        result["lpips"] = f"{torch.tensor(lpipss).mean():.5f}"

    metric_dir = "metric"
    os.makedirs(metric_dir, exist_ok=True)

    with open(metric_dir + "/results.json", "w") as fp:
        json.dump(result, fp, indent=True)


if __name__ == "__main__":
    main()
