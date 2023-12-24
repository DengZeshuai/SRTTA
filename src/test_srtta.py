import os
import cv2
import data
import torch
import argparse
import numpy as np
from tqdm import tqdm
from model.edsr import EDSR
from model.classifier import Classifier
import utils.utils_tta as util
from utils.utils_tta import test_all, get_corruptions

def run(args):
    if isinstance(args.scale,int):
        args.scale=[args.scale]
    args.data_test = args.data_test.split('+')
    val_dataloader = data.Data(args)
    test_loaders = val_dataloader.loader_test
    model = EDSR(args).cuda()
    corruptions = []
    if "single" in args.corruptions:
        corruptions = corruptions + get_corruptions()[1:]
    if "multi" in args.corruptions:
        corruptions = corruptions + ["BlurJPEG","BlurNoise","NoiseJPEG","BlurNoiseJPEG"]
    origin_model = EDSR(args).cuda()
    origin_model = util.load_model(args.base_model, origin_model)
    if args.cls_model:
        cls_model=Classifier().eval().cuda()
        state_dict = torch.load(args.cls_model)
        cls_model.load_state_dict(state_dict,strict=True)
    else:
        cls_model = None
    corruption_psnrs = {}
    corruption_ssims = {}
    corruption_srs = {}
    log_txt = {}
    for corruption in tqdm(corruptions):
        model = util.load_model(f"{args.pre_train}/state_{corruption}_last.pt", model)
        for _, t_data in enumerate(test_loaders):
            data_name = t_data.dataset.name
            if data_name not in corruption_psnrs:
                corruption_psnrs[data_name] = []
            if data_name not in corruption_ssims:
                corruption_ssims[data_name] = []
            if data_name not in corruption_srs:
                corruption_srs[data_name] = {}
            if data_name not in log_txt:
                log_txt[data_name] = ""
            if data_name == "Set5":
                psnrs, ssims, srs = test_all(args, model, t_data, return_sr=True)
                # record metric
                res_txt=f"{data_name} : PSNR = {np.mean(psnrs)} SSIM = {np.mean(ssims)}"
                log_txt[data_name] += res_txt + "\n"
                print(res_txt)
            else:
                t_data.dataset.set_corruption(corruption)
                psnrs, ssims,srs = test_all(args, model, t_data, 
                    origin_model=origin_model, cls_model=cls_model, return_sr=True)
                # record metric
                res_txt = f"{data_name} - {corruption} : PSNR = {np.mean(psnrs)} SSIM = {np.mean(ssims)}"
                log_txt[data_name] += res_txt + "\n"
                print(res_txt)
            corruption_psnrs[data_name].append(np.mean(psnrs))
            corruption_ssims[data_name].append(np.mean(ssims))
            corruption_srs[data_name][corruption]=srs

    print("finish testing.")
    for _, t_data in enumerate(test_loaders):
        data_name = t_data.dataset.name
        print(f"------------------Results of {data_name}---------------------")
        log_txt[data_name] += f"{data_name} - AVERAGE : PSNR = {np.mean(corruption_psnrs[data_name])} SSIM = {np.mean(corruption_ssims[data_name])}"
        print(log_txt[data_name])
    if args.save_dir:
        print("start saving results...")
        for _, t_data in enumerate(tqdm(test_loaders)):
            data_name = t_data.dataset.name
            for corruption in corruptions:
                os.makedirs(f"{args.save_dir}/{corruption}",exist_ok=True)
                img_names=[img_name.split("/")[-1] for img_name in t_data.dataset.images_lr[0]]
                for idx,sr in enumerate(corruption_srs[data_name][corruption]):
                    cv2.imwrite(f"{args.save_dir}/{corruption}/{img_names[idx]}",sr.numpy().astype(np.uint8)[:,:,[2,1,0]])
        print("finish saving results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_train', type=str, default='checkpoints/SRTTA/srtta_lifelong_x2.pt', 
                        help='pre-trained model directory')
    parser.add_argument('--dir_data', type=str, default='/chenzhuokun/datasets/', help='dataset directory')
    parser.add_argument('--base_model', type=str, default='checkpoints/EDSR_baseline_x2.pt', help='path to base model')
    parser.add_argument('--cls_model', type=str, default=None, help='path to cls model')
    parser.add_argument('--corruptions', type=str,default="single", help='multi-corruption')
    parser.add_argument('--multi-corruption', action='store_true', help='multi-corruption',default=True)
    parser.add_argument('--tta_data', type=str, default='DIV2KC', help='test dataset name')
    parser.add_argument('--data_train', type=str, default='PatchKernel', help='train dataset name')
    parser.add_argument('--data_test', type=str, default='DIV2KC', help='test dataset name')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--save_dir', type=str, default="", help='train dataset name')
    parser.add_argument('--debug', action='store_true', help='set this option to debugs the code')
    parser.add_argument('--model', default='EDSR', help='model name')
    parser.add_argument('--ext', type=str, default='img', help='dataset file extension')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--test_only', action='store_true', help='set this option to test the model',default=True)
    parser.add_argument('--n_threads', type=int, default=12, help='number of threads for data loading')
    parser.add_argument('--n_resblocks', type=int, default=16, help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
    parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
    parser.add_argument('--scale', type=int, default=[2], help='super resolution scale')
    args = parser.parse_args()
    run(args)