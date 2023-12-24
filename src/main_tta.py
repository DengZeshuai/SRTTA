import os
import glob
import argparse
from datetime import datetime
import torch
import random
import torch.optim as optim
from torch.utils.data import dataloader
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from copy import deepcopy
from model.classifier import Classifier
import srtta
import data
from data.div2kc import DIV2KC
from data.div2kmc import DIV2KMC
import utils.utils_tta as util
from model.edsr import EDSR

import logging
logger = logging.getLogger(__name__)


def init_logger(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    log_dest = "{}_{}.txt".format(
        os.path.splitext(args.log_file)[0], current_time)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, log_dest)),
            logging.StreamHandler()
    ])

    logger.info(args)


def setup_srtta(args, model, cls_model=None):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = srtta.configure_model(args, model)
    params, param_names = srtta.collect_params(model)
    optimizer = optim.Adam(params, args.lr, args.betas)
    srtta_model = srtta.SRTTA(args, model, optimizer, cls_model=cls_model)
    # logger.info(f"params for adaptation: %s", param_names)

    return srtta_model

def create_metric_file(args, img_paths, corruption='GaussianBlur'):
    # log file 
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    log_file = os.path.join(args.save_dir, "{}_{}".format(corruption, args.metric_file))
    header = ['adapted_img', 'iteration', 'Original', 'mean']
    header += [os.path.basename(p) for p in img_paths] # use image_name as column index
    metric_frame = pd.DataFrame(columns=header)
 
    if os.path.exists(log_file) and not args.reset:
        print("The log file exists, don't overwrite it.")
        return None # assure not to save the different results into the same dir
    metric_frame.to_csv(log_file, mode='a', index=False, header=header)

    return log_file, header

def record_metric(log_file, header, img_name, iteration, ori_psnr, ori_ssim, psnrs, ssims):
    # logger for test model on all image
    ori_metric = "{:.3f}/{:.4f}".format(ori_psnr, ori_ssim)
    metric_array = util.merge_list(psnrs, ssims)
    metric_array = [img_name, iteration, ori_metric] + metric_array # insert iteration and loss 
    metric_array = [metric_array] # two-dimentional list for new row dataFrame
    if len(header) != len(metric_array[0]):
        header = header[:len(metric_array[0])]
    metric_frame = pd.DataFrame(metric_array, index=None, columns=header)
    metric_frame.to_csv(log_file, mode='a', index=False, header=False)

def main(args):
    init_logger(args)

    if args.corruption is None:
        assert args.tta_data in ["DIV2KC","DIV2KMC"]
        if args.tta_data == "DIV2KC":
            corruptions = util.get_corruptions()[1:] # maybe affect the reproduction 
        else:
            corruptions = ["BlurJPEG", "BlurNoise", "NoiseJPEG", "BlurNoiseJPEG"]
    else:
        corruptions = args.corruption
    
    logger.info("Corruptions: {}".format(corruptions))

    #### init test data ###
    val_dataloader = data.Data(args)
    trainset = val_dataloader.loader_train.dataset.datasets[0]
    train_loader = val_dataloader.loader_train
    test_loaders = val_dataloader.loader_test # contain two test set
    
    #### init tta data ###
    if args.tta_data=="DIV2KC":
        tta_data = DIV2KC(args, name="DIV2KC", train=False)
    else:
        tta_data = DIV2KMC(args, name="DIV2KMC", train=False)
    tta_dataloader = dataloader.DataLoader(
                    tta_data, batch_size=1, shuffle=True,  # shuffle=True, 
                    pin_memory=not args.cpu, num_workers=args.n_threads)
    
    # config model
    model = EDSR(args)
    model = util.load_model(args.pre_train, model)
    origin_model = deepcopy(model)
    cls_model=Classifier().eval()
    state_dict = torch.load(args.classifier)
    cls_model.load_state_dict(state_dict,strict=True)

    if not args.cpu: 
        model = model.cuda()
        origin_model = origin_model.cuda()
        cls_model = cls_model.cuda()

    srtta_model = setup_srtta(args, model, cls_model=cls_model)
    
    if args.resume is not None:
        finished_corruption, finished_img_iter = srtta_model.resume(args.resume)
        start_idx = corruptions.index(finished_corruption)
        corruptions = corruptions[start_idx:]

    for idx, corruption in enumerate(corruptions):
        if args.params_reset and idx != 0:
            srtta_model.reset_parameters()

        # evaluate the model before test-time adaptation 
        ori_psnr, ori_ssim = 0, 0
        for _, t_data in enumerate(test_loaders):
            if args.resume is not None and finished_img_iter >= 0: break
            data_name = t_data.dataset.name
            if data_name in ["DIV2KC","DIV2KMC"]:
                # set corruption type for val dataset
                t_data.dataset.set_corruption(corruption)

            psnrs, ssims = util.test_all(args, model, t_data)
        
            logger.info("Original PSNR/SSIM: {:.3f}/{:.4f} on {} for {} data".format(
                np.mean(psnrs), np.mean(ssims), data_name, corruption))
            if data_name == "Set5":
                ori_psnr, ori_ssim = np.mean(psnrs), np.mean(ssims)
            else:
                tta_psnrs, tta_ssims = psnrs, ssims

        metric_file, header = create_metric_file(args, tta_data.get_img_paths(), corruption)
        record_metric(metric_file, header, -1, -1, ori_psnr, ori_ssim, tta_psnrs, tta_ssims)
        
        # compute fisher to select the important params
        if args.fisher_restore:
            srtta_model.compute_fisher(test_loaders)
        
        if corruption.lower() == 'original': continue # do not adapt for clean data
        
        # set corruption for tta and val dataset
        tta_data.set_corruption(corruption)
        
        if args.resume is not None and finished_img_iter == len(tta_dataloader):
            logger.info(f"Corruption: {corruption} have been adapted, skip to next ")
            finished_img_iter = -1
            continue

        for iter_idx, (img_lr, img_gt, filename) in enumerate(tta_dataloader):
            if args.resume is not None and finished_img_iter >= 0:
                if iter_idx + 1 <= finished_img_iter:
                    continue
                else:
                    finished_img_iter = -1

            trainset.set_image(img_lr)

            # adaptation 
            img_out = srtta_model(train_loader, img_lr, img_gt, filename[0], corruption)

            if args.save_results:   
                img_out = util.quantize(img_out)
                img_out = img_out.byte().permute(1, 2, 0).cpu()
                util.write_img(os.path.join(args.save_dir, filename[0] + '.png'), img_out.numpy())

            # test results
            if iter_idx == len(tta_dataloader) - 1 or iter_idx % args.test_interval == 0:
                ori_psnr, ori_ssim = 0, 0
                for _, t_data in enumerate(test_loaders):
                    data_name = t_data.dataset.name
                    if data_name == "Set5":
                        psnrs, ssims = util.test_all(args, model, t_data)
                    else:
                        psnrs, ssims = util.test_all(args, model, t_data, origin_model=origin_model, cls_model=cls_model)
                    
                    logger.info("Adapted PSNR/SSIM: {:.3f}/{:.4f} on {}-{} for {} data".format(
                        np.mean(psnrs), np.mean(ssims), data_name, filename[0], corruption))
                    if data_name == "Set5":
                        ori_psnr, ori_ssim = np.mean(psnrs), np.mean(ssims)
                    else:
                        tta_psnrs, tta_ssims = psnrs, ssims
                
                # record metirc
                record_metric(metric_file, header, filename[0], args.iterations, ori_psnr, ori_ssim, tta_psnrs, tta_ssims)
                
            # save model params
            srtta_model.save(corruption, iter_idx)

    logger.info("Finish Test-time Adaptation....")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='../experiment/save/', help='path to gt image for metric computing')
    parser.add_argument('--metric_file', type=str, default='tta_metrics.csv', help='path to the file for recording metric')
    parser.add_argument('--log_file', type=str, default='logger.txt', help='path to gt image for metric computing')
    parser.add_argument('--reset', action='store_true', help='reset the adapting')
    parser.add_argument('--exp_name', type=str, default='debug', help='exp name')
    
    # model options
    parser.add_argument('--model', default='EDSR', help='model name')
    parser.add_argument('--pre_train', type=str, default='checkpoints/EDSR_baseline_x2.pt', help='pre-trained model directory')
    parser.add_argument('--classifier', type=str, default='checkpoints/classifier.pt', help='pre-trained model directory')
    parser.add_argument('--n_resblocks', type=int, default=16, help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
    parser.add_argument('--scale', type=str, default='2', help='super resolution scale')
    parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
    parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')

    # data options
    parser.add_argument('--dir_data', type=str, default='../datasets/', help='dataset directory')
    parser.add_argument('--data_train', type=str, default='PatchKernel', help='train dataset name')
    parser.add_argument('--data_test', type=str, default='DIV2KC+Set5', help='test dataset name')
    parser.add_argument('--tta_data', type=str, default='DIV2KC', help='test dataset name')
    parser.add_argument('--corruption', type=str, default=None, help='the type of corruption data')
    parser.add_argument('--ext', type=str, default='img', help='dataset file extension')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--patch_size', type=int, default=96, help='output patch size')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    
    # training options
    parser.add_argument('--n_fixed_blocks', type=int, default=0, help='the number of last resblocks that do not update during tta')
    parser.add_argument('--iterations', type=int, default=10, help='the number of iterations to adapt on each test image')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
    parser.add_argument('--seed', type=int, default=1004, help='random seed')
    parser.add_argument('--save_results', action='store_true', help='save output results')
    parser.add_argument('--params_reset', action='store_true', help='flag to reset the parameters for each domain data')
    parser.add_argument('--fisher_restore', action='store_true', help='flag to stochastically restore from the original model')
    parser.add_argument('--fisher_ratio', type=float, default=0.3, help='threshold of stochastic restoration')
    parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
    parser.add_argument('--n_threads', type=int, default=12, help='number of threads for data loading')
    parser.add_argument('--resume', type=str, default=None, help='path for resume from specific checkpoint')
    parser.add_argument('--test_interval', type=int, default=100, help='number of interval for computing metric')
    parser.add_argument('--multi-corruption', action='store_true', help='multi-corruption')
    parser.add_argument('--teacher_weight', type=float, default=1, help='the weight of the teacher degradation loss')
    
    args = parser.parse_args()
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    args.data_train = args.data_train.split('+')
    args.data_test = args.data_test.split('+')
    if args.corruption is not None:
        args.corruption = args.corruption.split('+')

    ### hypyer-parameters for random degradation, TODO
    args.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
    args.kernel_list = ['iso', 'aniso']
    args.kernel_prob = [0.5, 0.5]
    args.blur_sigma = [0.2, 3]
    args.betag_range = [0.5, 4]
    args.betap_range = [1, 2]
    args.noise_range = [1, 30]
    args.jpeg_range = [30, 95]
    print(args)

    date_str = time.strftime("%Y%m%d", time.localtime())
    args.save_dir = f"../experiment/{date_str}_{args.exp_name}/"

    util.set_seed(args.seed)
    main(args)