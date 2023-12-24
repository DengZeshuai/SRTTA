import os
import glob
import time
try:
    import imageio.v2 as imageio
except:
    import imageio
from tqdm import tqdm
import numpy as np
import logging
import random
import torch

import utils.utils_image as util
import utils.utils_blindsr as blindsr
import utils.utils_blindsr_plus as blindsr_plus

def get_corruptions():
    """ Original: bicubic downsample """
    corruptions = ['Original', 'GaussianBlur', 'DefocusBlur', 'GlassBlur', 
                   'GaussianNoise', 'PoissonNoise', 'ImpulseNoise', 'SpeckleNoise', 'JPEG']
    return corruptions

def get_mixed_corruptions():
    """ Original: bicubic downsample """
    corruptions = ["BlurJPEG", "BlurNoise", "NoiseJPEG", "BlurNoiseJPEG"]
    return corruptions

def preprocess_img(img, scale=2, corruption='Original'):
    corruptions = get_corruptions()
    corr_idx = corruptions.index(corruption)

    img = util.uint2single(img)
    img_lr = None
    if corr_idx == 0:
        if scale != 1:
            img_lr = blindsr.bicubic_degradation(img, scale)
        else:
            img_lr = img

    if corruption.lower().find('blur') >= 0:
        if corr_idx == 1:
            img_blur = blindsr.add_blur(img, scale)
        if corr_idx == 2:
            img_blur = blindsr_plus.add_defocus_blur(img)
        if corr_idx == 3:
            img_blur = blindsr_plus.add_glass_blur(img)
        
        if scale != 1: 
            img_lr = blindsr.bicubic_degradation(img_blur, scale)
        else:
            img_lr = img_blur
    
    if corruption.lower().find('noise') >= 0:
        if scale != 1: 
            img_lr = blindsr.bicubic_degradation(img, scale)
        else:
            img_lr = img

        if corr_idx == 4:
            img_lr = blindsr.add_Gaussian_noise(img_lr, noise_level1=2, noise_level2=25)
        
        if corr_idx == 5:
            img_lr = blindsr_plus.add_scale_Poisson_noise(img_lr)

        if corr_idx == 6:
            img_lr = blindsr_plus.add_impluse_noise(img_lr)

        if corr_idx == 7:
            img_lr = blindsr.add_speckle_noise(img_lr)
        
    if corruption.lower().find('jpeg') >= 0:
        if scale != 1: 
            img_lr = blindsr.bicubic_degradation(img, scale)
        else:
            img_lr = img
        img_lr = blindsr.add_JPEG_noise(img_lr)
    
    if img_lr is None: # defalut bicubic
        print("The corruption are not support, default bicubic")
        if scale != 1: 
            img_lr = blindsr.bicubic_degradation(img, scale)
        else:
            img_lr = img
    
    # signle2uint
    img_lr = util.single2uint(img_lr) # 0-1 to 0-255

    return img_lr





def read_img(img_path):
    img = imageio.imread(img_path)
    # some images have 4 channels
    if img.shape[2] > 3: img = img[:, :, :3]
    return img

def write_img(save_path, image):
    imageio.imwrite(save_path, image)


def get_paths(input_dir, target_dir):
    if not input_dir == target_dir:
        lr_paths = sorted(glob.glob(os.path.join(input_dir, "*")))
        gt_paths = sorted(glob.glob(os.path.join(target_dir, "*")))
        # check whether the lr path is corresponding to the gt path
        for lr_path, gt_path in zip(lr_paths, gt_paths):
            lr_name = os.path.basename(lr_path)
            gt_name = os.path.basename(gt_path)
            assert lr_name == gt_name
    else:
        scale = max(args.scale) if isinstance(args.scale, list) else scale
        lr_paths = sorted(glob.glob(os.path.join(input_dir, "*_LR{}.png".format(scale))))
        gt_paths = sorted(glob.glob(os.path.join(target_dir, "*_HR.png")))
        for lr_path, gt_path in zip(lr_paths, gt_paths):
            lr_name = os.path.basename(lr_path)
            gt_name = os.path.basename(gt_path).replace('_HR', '_LR{}'.format(scale))
            assert lr_name == gt_name
    
    return lr_paths, gt_paths


def test_one(args, model, lr_img, hr_img, sr_img=None,return_sr=False):
    model.eval()
    if not isinstance(lr_img, torch.Tensor):
        lr_img = util.single2tensor4(lr_img)
    if not args.cpu: lr_img = lr_img.cuda()
    
    if sr_img is None:
        with torch.no_grad():
            sr_img = model.sr_forward(lr_img)
    
    sr_img = util.quantize(sr_img.cpu().squeeze(0).permute(1, 2, 0), args.rgb_range)
    sr_img_y = util.rgb2ycbcr(sr_img.numpy() / 255.) * 255. # normalize to 0-1, aviod round() operation
    
    if isinstance(hr_img, torch.Tensor):
        hr_img = hr_img.squeeze().permute(1, 2, 0).numpy()
    hr_img_y = util.rgb2ycbcr(hr_img / 255.) * 255. # normalize to 0-1, aviod round() operation

    psnr_, ssim_ = util.calc_psnr_ssim(sr_img_y, hr_img_y)

    if return_sr:
        return psnr_, ssim_, sr_img
    else:
        return psnr_, ssim_


def test_all(args, model, test_loader, origin_model=None, cls_model=None, return_sr=False):
    model_update = model
    psnrs, ssims, srs = [], [], []
    for idx, (lr, gt, _) in tqdm(enumerate(test_loader), ncols=80, total=len(test_loader)):
        if cls_model:
            cls_pred = cls_model(lr[:,[2,1,0]].cuda()/255.)
            if (cls_pred.sigmoid() > 0.5).sum().item() == 0:
                # directly using pretrained model to upscale clean images
                model = origin_model
            else:
                model = model_update

        if return_sr:
            psnr_, ssim_, sr = test_one(args, model, lr, gt, return_sr=True)
            srs.append(sr)
        else:
            psnr_, ssim_ = test_one(args, model, lr, gt)
        psnrs.append(psnr_)
        ssims.append(ssim_)

    if return_sr:
        return psnrs, ssims, srs
    else:
        return psnrs, ssims

def merge_list(psnrs, ssims):
    """return 1D array with results of [mean, 0, 1, ..., N]"""
    str_list = []
    for psnr, ssim in zip(psnrs, ssims):
        psnr_ssim = "{:.3f}/{:.4f}".format(psnr, ssim)
        str_list.append(psnr_ssim)
    str_list.insert(0, "{:.3f}/{:.4f}".format(np.mean(psnrs), np.mean(ssims)))
    return str_list

def transform_tensor(img, op, undo=False):
    """
    params:
        img: BxHxWxC, [0-255], Tensor
        op: transform operator
    """
    if undo:
        if op.find('t') >= 0: # first rotate back
            img = img.permute((0, 1, 3, 2)).contiguous()
    
    if op.find('v') >= 0:
        img = torch.flip(img, dims=[3]).contiguous()
    if op.find('h') >= 0:
        img = torch.flip(img, dims=[2]).contiguous()
    
    if not undo:
        if op.find('t') >= 0: # rotate in the last
            img = img.permute((0, 1, 3, 2)).contiguous()

    return img

def augment_transform(img_in, undo=False):
    """ transform the input tensor 8 times
    params:
        img: BxHxWxC, [0-255], Tensor
    """
    img_outs = []
    tran_ops = ['', 'v', 'h', 't', 'vh', 'vt', 'ht', 'vht'] # augment
    for op in tran_ops:
        img_outs.append(transform_tensor(img_in, op, undo))
    return img_outs, tran_ops


def load_model(pre_train, model):
    print(pre_train)
    assert os.path.exists(pre_train)
    if os.path.exists(pre_train):
        if "baseline" in pre_train.lower():
            state_dict = torch.load(pre_train)
        else:
            try:
                state_dict = torch.load(pre_train)['model']
            except:
                state_dict = torch.load(pre_train)
        state_dict={k.replace("upsample.","upsampler."):v for k,v  in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    return model


def get_logger(save_path):
    logger = logging.getLogger()
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(save_path)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console.setFormatter(formatter)
    logger.disabled = False
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 