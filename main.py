import os, random, datetime
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# uv run tensorboard --logdir logs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import Image

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_imgs(image_dir):
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names
    img_paths = [os.path.join(image_dir, n) for n in img_names]
    N_imgs = len(img_paths)

    mask = np.linspace(0, img_names.shape[0]-1, N_TRAIN, endpoint=False).astype(int)
    img_names = img_names[mask]
    img_paths = np.array(img_paths)[mask].tolist()

    img_list = []
    for p in img_paths:
        img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
        img_list.append(img)
    img_list = np.stack(img_list)  # (N, H, W, 3)
    img_list = torch.from_numpy(img_list).float() / 255  # (N, H, W, 3) torch.float32
    H, W = img_list.shape[1], img_list.shape[2]

    results = {
        'imgs': img_list,  # (N, H, W, 3) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'H': H,
        'W': W,
    }
    print(f"{image_dir} H={H} W={W}")
    return results

N_TRAIN = 5 # how many images to use as train images

scenes = ["llff_flower", "fortress", "room"] # 
train_imgs = []
for sc in scenes:
    image_dir = os.path.join('nerfmm_colab_data', sc)
    image_data = load_imgs(image_dir)
    imgs = image_data['imgs']  # (N, H, W, 3) torch.float32

    H = image_data['H']
    W = image_data['W']
    train_imgs.append(imgs)

from sparsenerf.networks import UNet
from sparsenerf.train import SparseNeRFScene
from datetime import datetime

class RayParameters():
    def __init__(self, max_epochs):
        self.NEAR, self.FAR = 0.0, 1.0  # ndc near far
        self.N_SAMPLE = 128  # samples per ray
        self.POS_ENC_FREQ = 10  # positional encoding freq for location
        self.DIR_ENC_FREQ = 4   # positional encoding freq for direction
        self.POS_ENC_REG_ITERS_p = 0.0
        self.DIR_ENC_REG_ITERS_p = 0.0
        self.POS_ENC_REG_ITERS = self.POS_ENC_REG_ITERS_p * max_epochs
        self.DIR_ENC_REG_ITERS = self.DIR_ENC_REG_ITERS_p * max_epochs

RUN_NAME = f"backproj-UNet-mask-enc-multi-5-images"
#RUN_NAME =  "test-useless-" + datetime.now().strftime("%d-%m-%H:%M:%S")
N_EPOCH = 1000*10  # set to 1000 to get slightly better results. we use 10K epoch in our paper.
EVAL_INTERVAL = 250*10  # render an image to visualise for every this interval.
ray_params = RayParameters(N_EPOCH)

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

image_enc = UNet(3, out_channels=4).to(device)
#nn.init.kaiming_normal_(image_enc.weight, mode='fan_out', nonlinearity='relu')
nerfScenes = []
for i in range(len(scenes)):
    nerfScenes.append(SparseNeRFScene(
        image_enc,
        image_enc.out_channels,
        N_EPOCH, EVAL_INTERVAL,
        ray_params,
        train_imgs[i], scenes[i],
        RUN_NAME,
        use_views=True,
        use_im_encoder=True,
        train_im_encoder=True,
        n_views=3,
        mask_current_view=True,
        use_views_after_N_epochs=100,
        device=device
    ))
#torch.autograd.set_detect_anomaly(True)
print(RUN_NAME)
for epoch_i in tqdm(range(N_EPOCH)):
    for i in range(len(scenes)):
        nerfScenes[i].train_epoch()

for i in range(len(scenes)):
    if nerfScenes[i].use_im_encoder:
        nerfScenes[i].save_encoded_ims()

for i in range(len(scenes)):
    print(f"{i+1}/{len(scenes)}")
    nerfScenes[i].generate_results(1)