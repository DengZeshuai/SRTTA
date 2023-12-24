import os
import glob
import cv2
import random
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset

import utils.utils_tta as util


def trainTranform(img_size=224):
    transform = transforms.Compose([
        transforms.RandomCrop(size=img_size),
        transforms.RandomApply([
            transforms.RandomRotation(
                90, resample=False, expand=False, center=None)
        ], p=0.8)
    ])

    return transform


class DIV2KMD(Dataset):
    def __init__(self, base_path, img_size=224, train=True, cache=False):
        super(DIV2KMD, self).__init__()
        self.img_size = img_size
        self.single_type = util.get_corruptions()
        self.mixed_types = util.get_mixed_corruptions()
        self.corruptions = self.single_type + self.mixed_types
        self.classes = ["blur", "noise", "jpeg"]
        if train:
            self.img_paths = list(sorted([os.path.join(base_path, p) for p in os.listdir(base_path)]))
            self.img_paths = self.img_paths[0:800]
            self.transform = trainTranform(img_size)
        else:
            self.img_paths = []
            self.labels = []
            for cor in os.listdir(base_path):
                img_names = sorted(glob.glob(os.path.join(base_path, cor, 'X2', '*.png')))
                for img_name in img_names:
                    img_label = []
                    self.img_paths.append(f"{img_name}")
                    if "blur" in cor.lower():
                        cor_cls = "blur"
                        img_label.append(self.classes.index(cor_cls)) # class 0
                    if "noise" in cor.lower():
                        cor_cls = "noise"
                        img_label.append(self.classes.index(cor_cls)) # class 1
                    if "jpeg" in cor.lower():
                        cor_cls = "jpeg"
                        img_label.append(self.classes.index(cor_cls)) # class 2
                    self.labels.append(img_label)

        self.train = train
        self.cache = cache
        if self.cache:
            print("caching images...")
            self.imgs = [cv2.imread(path)[:, :, [2, 1, 0]] for path in tqdm(self.img_paths)]

    def degrade_img(self, img, cor_type):
        """"preprocess high-resolution images with random degradation"""
        img = img.copy()
        if not self.train: return img

        if cor_type in self.single_type:
            img = util.preprocess_img(img, scale=2, corruption=cor_type)
        else:
            operators = ['blur', 'down', 'noise', 'jpeg']
            for op in operators:
                if op == 'blur' and op in cor_type.lower():
                    degradation_type = random.choice(
                        ['GaussianBlur', 'DefocusBlur', 'GlassBlur'])
                    img = util.preprocess_img(img, scale=1, corruption=degradation_type)
                if op == 'down':
                    # downsampling image using bicubic interpolation
                    img = util.preprocess_img(img, scale=2, corruption='Original')
                if op == 'noise' and op in cor_type.lower():
                    degradation_type = random.choice(
                        ['GaussianNoise', 'PoissonNoise', 'ImpulseNoise', 'SpeckleNoise'])
                    img = util.preprocess_img(img, scale=1, corruption=degradation_type)
                if op == 'jpeg' and op in cor_type.lower():
                    degradation_type = random.choice(['JPEG'])
                    img = util.preprocess_img(img, scale=1, corruption=degradation_type)
        return img

    def __getitem__(self, idx):
        if self.cache:
            img = self.imgs[idx].copy()
        else:
            try:
                img = cv2.imread(self.img_paths[idx])[:, :, [2, 1, 0]]
            except:
                print(1)
        
        img = np.ascontiguousarray(img).astype(np.float32)
        selected_types = random.choices(self.corruptions, k=1)[0]
        if self.train:
            # convert numpy array to tensor, HWC -> CHW
            input_img = torch.from_numpy(img).permute(2, 0, 1)
            # randomly crop images
            input_img = self.transform(input_img)
            # degrada images
            input_img = self.degrade_img(input_img.permute(1, 2, 0).numpy(), selected_types)
            # convert numpy array to tensor, normalize to [0-1]
            input_img = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.
            
            if "origin" in selected_types.lower():
                # label for clean image
                label = torch.zeros(3).float()
            else:
                label = torch.zeros(3).float()
                for idx, c in enumerate(self.classes):
                    if c in selected_types.lower():
                        label[idx] = 1.
        else:
            input_img = torch.from_numpy(img).permute(2,0,1) / 255.
            if len(self.labels[idx]) == 0:
                label = torch.zeros(3).float()
            else:
                label = torch.nn.functional.one_hot(torch.tensor(
                    [p for p in self.labels[idx]]), len(self.classes)).sum(0).bool().float()
        return input_img, label, self.img_paths[idx]
        
    def __len__(self):
        return len(self.img_paths)