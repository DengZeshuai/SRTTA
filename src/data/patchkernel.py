import os
import glob
import random
import math
import time
import torch
try:
    import imageio.v2 as imageio
except:
    import imageio
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import utils.basicsr_degradations as degradation


class PatchKernel(data.Dataset):
    def __init__(self, args, name=None, train=True, task_idx=0):
        self.args = args
        self.scale = max(args.scale)
        self.task_idx = task_idx
        self.train = train
        self.kernel_range = args.kernel_range  # kernel size ranges from 7 to 21
        self.kernel_list = args.kernel_list
        self.kernel_prob = args.kernel_prob
        self.blur_sigma = args.blur_sigma
        self.betag_range = args.betag_range
        self.betap_range = args.betap_range
    
    def set_image_path(self, image_path, image=None):
        self.image_path = image_path
        if image is not None:
            self.image = image
        else:
            self.image = imageio.imread(image_path)[:, :, :3] 
    
    def set_image(self, image):
        if isinstance(image, torch.Tensor):
            self.image = util.tensor2single3(image)
        else:
            self.image = image
    
    def set_task(self, task_idx):
        self.task_idx = task_idx
    
    def __len__(self):
        return self.args.batch_size

    def __getitem__(self, idx):
        patch = self.get_patch(self.image, self.args.patch_size, self.scale)
        patch = self.augment(patch)
        if 1 in self.task_idx:
            kernel_size = random.choice(self.kernel_range)
            kernel = degradation.random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
            kernel = torch.FloatTensor(kernel)
        else:
            kernel = torch.zeros(1)
        
        patch = util.single2tensor3(patch)

        return patch, kernel

    def get_patch(self, img, patch_size=64, scale=2):
        h, w = img.shape[:2]
        rw = random.randrange(0, w - patch_size  + 1)
        rh = random.randrange(0, h - patch_size + 1)
        
        patch = img[rh:rh + patch_size, rw:rw + patch_size, :].copy()
        return patch

    def augment(self, img_in, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)
            
            return img

        return _augment(img_in)