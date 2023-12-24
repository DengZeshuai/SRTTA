import os
import glob
import cv2
import math
import numpy as np
import torch
import random

from tqdm import tqdm
import argparse
import imageio

from multiprocessing import Pool

import sys
sys.path.insert(1, '..')
import utils.utils_image as util
import utils.utils_blindsr as blindsr
import utils.utils_blindsr_plus as blindsr_plus

def get_corruptions():
    """ Original: bicubic downsample """
    corruptions = ['Original', 'GaussianBlur', 'DefocusBlur', 'GlassBlur', 
                   'GaussianNoise', 'PoissonNoise', 'ImpulseNoise', 'SpeckleNoise', 'JPEG']
    return corruptions


def preprocess_img(img, scale=2, corruption='Original'):
    corruptions = get_corruptions()
    corr_idx = corruptions.index(corruption)

    img = util.uint2single(img)
    img_lr = None
    if corr_idx == 0:
        img_lr = blindsr.bicubic_degradation(img, scale)

    if corruption.lower().find('blur') >= 0:
        img_blur = img
        if corr_idx == 1:
            img_blur = blindsr.add_blur(img_blur, scale)
        if corr_idx == 2:
            img_blur = blindsr_plus.add_defocus_blur(img_blur)
        if corr_idx == 3:
            img_blur = blindsr_plus.add_glass_blur(img_blur)
        img_lr = blindsr.bicubic_degradation(img_blur, scale)
    
    if corruption.lower().find('noise') >= 0:
        img_lr = blindsr.bicubic_degradation(img, scale)
        if corr_idx == 4:
            img_lr = blindsr.add_Gaussian_noise(img_lr, noise_level1=2, noise_level2=25)
        
        if corr_idx == 5:
            img_lr = blindsr_plus.add_scale_Poisson_noise(img_lr)

        if corr_idx == 6:
            img_lr = blindsr_plus.add_impluse_noise(img_lr)

        if corr_idx == 7:
            img_lr = blindsr.add_speckle_noise(img_lr)
        
    if corr_idx == 8:
        img_lr = blindsr.bicubic_degradation(img, scale)
        img_lr = blindsr.add_JPEG_noise(img_lr)
    
    if img_lr is None: # defalut bicubic
        print("The corruption are not support, default bicubic")
        img_lr = blindsr.bicubic_degradation(img, scale)
    
    # signle2uint
    img_lr = util.single2uint(img_lr) # 0-1 to 0-255

    return img_lr

def process(img_path, save_dir, scale=2, corruption='Original'):
    img_hr = imageio.imread(img_path)
    if img_hr.shape[2] > 3: img_hr = img_hr[:,:,:3]
    img_lr = preprocess_img(img_hr, scale, corruption=corruption)
    basename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(save_dir, basename + '.png')
    imageio.imwrite(save_path, img_lr)
    
    return basename, corruption
    

def get_image_paths(input_dir):
    img_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    return img_paths

def main(args):
    img_paths = get_image_paths(args.input_dir)
    corruptions = get_corruptions()
    
    if len(img_paths) < len(corruptions) * args.n_per_corruption:
        repeat = len(corruptions) * args.n_per_corruption // len(img_paths) + 1
        img_paths = img_paths * repeat

    if args.n_workers > 1:
        print(f'Read images with multiprocessing, #thread: {args.n_workers} ...')
        pool = Pool(args.n_workers)
        
        pbar = tqdm(total=len(img_paths)*len(args.scale), unit='image', ncols=100)
        
        def callback(args):
            """get the image data and update pbar."""
            basename, corruption = args
            pbar.update(1)
            pbar.set_description(f'Processing {basename} with {corruption} ...')
    
    for s in args.scale:
        start_idx = 0
        for corruption in corruptions:
            if args.debug and corruption != args.corruption:
                continue

            end_idx = start_idx + args.n_per_corruption
            assert end_idx < len(img_paths)
            save_dir = os.path.join(args.output_dir, corruption, 'X{}'.format(s))
            if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

            for img_path in img_paths[start_idx:end_idx]:
                if args.n_workers > 1:
                        pool.apply_async(
                            process,
                            args=(img_path, save_dir, s, corruption),
                            callback=callback
                        )
                else:
                    print("Processing {} with {} ...".format(os.path.basename(img_path), corruption))
                    process(img_path, save_dir, s, corruption)
    
    if args.n_workers > 1:
        pool.close()
        pool.join()
        pbar.close()
    
    print(f'\nFinish processing.')
            
            
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/mnt/cephfs/home/dengzeshuai/data/sr/DIV2KRK/gt_rename/', help='Input folder')
    parser.add_argument('--output_dir', type=str, default='/mnt/cephfs/home/dengzeshuai/data/sr/DIV2KRK/corruptions/', help='Output folder')
    parser.add_argument('--scale', type=str, default='2', help='super resolution scale')
    parser.add_argument('--debug', action='store_true', help='set this option to debugs the code')
    parser.add_argument('--corruption', type=str, default='ImpluseNoise', help='the type of ImpluseNoise for debugging')
    parser.add_argument('--n_per_corruption', type=int, default=100, help='number of aux head to train')
    parser.add_argument('--n_workers', type=int, default=12, help='number of workers to process image')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproduce')
    args = parser.parse_args()
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))

    set_seed(args.seed)

    main(args)
