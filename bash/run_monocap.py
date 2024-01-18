import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
dataset = "monocap"

seqs = [
    "lan_images620_1300",
    "marc_images35000_36200",
    "olek_images0812",
    "vlad_images1011",
]

base_name = "instant_avatar_bl"

cfg_name = "SNARF_NGP"

for seq in seqs:
    experiment = f"{base_name}_{seq}"
    cmd_str = f"python train.py --config-name {cfg_name} dataset={dataset}/{seq} experiment={experiment} train.max_epochs=200"

    cmd_str1 = f"python metric_infer.py --config-name {cfg_name} dataset={dataset}/{seq} experiment={experiment} train.max_epochs=200"

    os.system(cmd_str)

    # os.system(cmd_str1)
