import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from copy import deepcopy
import utils.utils_tta as utils_tta
import utils.utils_image as util

import utils.utils_blindsr_plus as blindsr_plus
import utils.basicsr_degradations as degradations
from utils.diffjpeg import DiffJPEG


import logging
logger = logging.getLogger(__name__)

def configure_model(args, model):
    """Configure model for use with tta."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()

    train_params = ['body'] # only update the body

    for k, v in model.named_parameters():
        prefix, block_index = k.split('.')[:2]
        if prefix in train_params:
            
            logger.info('train params: {}'.format(k))
            v.requires_grad = True
        else:
            logger.info('freezing params: {}'.format(k))
            v.requires_grad = False # fix the other layers
    
    return model

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True: #isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    # print(nm, np)
    return params, names

def compute_loss(pred, target, eps=1e-3):
    """ L1 Charbonnier loss """
    return torch.sqrt(((pred - target)**2) + eps).mean()

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def create_ema_model(model):
    """Copy the model and optimizer states for resetting after adaptation."""
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return ema_model

class SRTTA():
    def __init__(self, args, model, optimizer, fisher=None, cls_model=None):
        self.args = args
        self.model = model
        self.origin_model = model
        self.fisher = fisher
        self.cls_model = cls_model
        self.optimizer = optimizer
        
        self.compute_loss = compute_loss
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

        self.noise_range = args.noise_range
        self.jpeg_range = args.jpeg_range
        self.jpeger = DiffJPEG(differentiable=False)
        self.fishers = {}
        if not self.args.cpu:
            self.jpeger = self.jpeger.cuda()

        # teacher model, this do not need the gradient
        self.model_teacher = create_ema_model(self.model)

    def __call__(self, train_loader, img_lr, img_gt, img_name, corruption=None):
        # test-time adaptation

        if self.cls_model:
            classes = ["origin", "blur", "noise", "jpeg"]
            with torch.no_grad():
                input_img = img_lr.clone()
                input_img = input_img[:, [2,1,0]].cuda() / 255.
                cls_pred = self.cls_model(input_img)
                degradation_types = [classes[cls_pred.argmax(-1).item()]]

            corruptions=[]
            for dtype_ in degradation_types:
                if dtype_.lower().find("noise") >= 0:
                    corruption = 'GaussianNoise'
                elif dtype_.lower().find("jpeg") >= 0:
                    corruption = "JPEG"
                elif dtype_.lower().find("blur") >= 0:
                    corruption = 'GaussianBlur'
                elif dtype_.lower().find("origin") >= 0:
                    corruption = "Original"
                corruptions.append(corruption)
        else:
            corruptions = [corruption]

        logger.info(f"use {','.join(corruptions)} for {img_name}")
        for iteration in range(self.args.iterations):
            self.test_time_adaptation(train_loader, corruptions)
           
            # test results
            if not isinstance(img_lr, torch.Tensor):
                img_lr = util.single2tensor4(img_lr)
            if not self.args.cpu: img_lr = img_lr.cuda()
            
            self.model.eval()
            with torch.no_grad():
                sr_img = self.model.sr_forward(img_lr)

            # record metric
            with torch.no_grad():
                psnr, ssim = utils_tta.test_one(self.args, self.model, img_lr, img_gt, sr_img=sr_img)
            
            if iteration == self.args.iterations - 1:
                logger.info("Adapted PSNR/SSIM: {:.3f}/{:.4f} on {} with {} iters".format(
                    psnr, ssim, img_name, iteration, corruption))

        return sr_img

    def test_time_adaptation(self, train_loader, corruption=None):
        total_loss = 0
        self.model.train()
        self.optimizer.zero_grad()

        if corruption is not None:
            task_idxs = []
            for corruption_ in corruption:
                if corruption_.lower().find('blur') >= 0:
                    task_idx = 1
                elif corruption_.lower().find('noise') >= 0:
                    task_idx = 2
                elif corruption_.lower().find('jpeg') >= 0:
                    task_idx = 3
                else:
                    return 0
                task_idxs.append(task_idx)
        train_loader.dataset.datasets[0].set_task(task_idxs)
        loader_iter = iter(train_loader)
        img_label, kernel = next(loader_iter)
        if not self.args.cpu: 
            img_label = img_label.cuda()
            if kernel.dim() > 2: kernel = kernel.cuda()
        with torch.no_grad():
            img_in = self.preprocess(img_label, kernel, task_idxs).contiguous()
            out_tea_gt = self.model_teacher(img_label, aux_forward=True)
        
        with torch.no_grad():
            out_student_gt = self.model(img_label, aux_forward=True)
        out_student = self.model(img_in, aux_forward=True)
        # compute student loss
        loss = self.compute_loss(out_student, out_student_gt.detach()) 
        # compute teacher loss
        if self.args.teacher_weight > 0:
            loss += self.args.teacher_weight * self.compute_loss(out_student, out_tea_gt)
        loss.backward()
        total_loss += loss.item()

        self.optimizer.step()

        if self.args.fisher_restore:
            self.fisher_restoration()

        return total_loss

    def preprocess(self, img_gt, kernel, task_idx=1):
        # convert to [0-1]
        img_in = img_gt / 255.
        # add blur
        if task_idx==1 or (isinstance(task_idx,list) and 1 in task_idx):
            img_in = blindsr_plus.filter2D(img_in, kernel)
        # add noise
        if task_idx==2 or (isinstance(task_idx,list) and 2 in task_idx):
            img_in = degradations.random_add_gaussian_noise_pt(
                    img_in, sigma_range=self.noise_range, clip=False, rounds=False, gray_prob=0.4)
        # add jpge
        if task_idx==3 or (isinstance(task_idx,list) and 3 in task_idx):
            jpeg_p = img_in.new_zeros(img_in.size(0)).uniform_(*self.jpeg_range)
            img_in = self.jpeger(img_in, quality=jpeg_p).contiguous()
        
        # convert to [0-255]
        img_in = (torch.clamp(img_in, 0, 1) * 255.0).round()

        return img_in.detach()

    def fisher_restoration(self):
        """Restore the important params back to original model"""
        for nm, m  in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    # fishers[name]: [fisher, mask]
                    mask = self.fishers[f"{nm}.{npp}"][-1]
                    with torch.no_grad():
                        p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)

    def reset_parameters(self):
        """Restore the model and optimizer states from copies."""
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def compute_fisher(self, test_loaders):
        if len(self.fishers) > 0: return self.fishers

        fishers = {}
        fisher_optimizer = optim.Adam(self.model.parameters())
        for idx_data, t_data in enumerate(test_loaders):
            if t_data.dataset.name != "Set5": continue
            for idx, (img_lr, _, filename) in enumerate(t_data, start=1):
                if not self.args.cpu: img_lr = img_lr.cuda()
                
                fisher_optimizer.zero_grad()
                tran_imgs, tran_ops = utils_tta.augment_transform(img_lr)
                tran_imgs.reverse() # the last item is img_lr
                tran_ops.reverse() 
                
                # compute consistent loss
                sr_imgs = []
                for idx, (tran_img, op) in enumerate(zip(tran_imgs, tran_ops), start=1):
                    if idx < len(tran_imgs):
                        with torch.no_grad():
                            sr_img = self.model(tran_img)
                            sr_img = utils_tta.transform_tensor(sr_img, op, undo=True)
                            sr_imgs.append(sr_img)
                    else:
                        sr_img = self.model(tran_img)
                        sr_imgs.append(sr_img)
                # with torch.no_grad():
                sr_pseudo = torch.cat(sr_imgs, dim=0).mean(dim=0, keepdim=True).detach()
                loss = self.compute_loss(sr_imgs[-1], sr_pseudo)
                loss.backward()

            # computer fisher
            for name, param in self.model.named_parameters():
                if param.grad is not None:                   
                    if idx_data > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if idx_data == len(t_data):
                        fisher = fisher / idx_data
                    fishers.update({name: fisher})
        
        # computer mask based on the fisher
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                fisher = fishers[name].flatten() # TODO: check whether flatten is reverse
                _, mask_idx = torch.topk(fisher, k=int(len(fisher) * self.args.fisher_ratio))
                mask = param.new_zeros(param.shape).flatten() # ensure the mask and p are in the save devide 
                mask[mask_idx] = 1
                mask = mask.view(param.shape)
                self.fishers.update({name: [fisher, mask]})

        # self.fishers = fishers
        fisher_optimizer.zero_grad()

        return fisher

    def resume(self, resume_path):
        if resume_path is not None:
            resume_state = torch.load(resume_path)
            load_model_and_optimizer(self.model, self.optimizer, 
                resume_state['model'], resume_state['optimizer'])
            self.model_state = resume_state['ori_model']
            self.optimizer_state = resume_state['ori_optimizer']
            corruption = resume_state['corruption']
            iter_idx = resume_state['iter_idx']
        
        return corruption, iter_idx

    def save(self, corruption='GaussianBlur', iter_idx=0):
        state = {}
        state['model'] = self.model.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        state['ori_model'] = self.model_state
        state['ori_optimizer'] = self.optimizer_state
        state['corruption'] = corruption
        state['iter_idx'] = iter_idx

        save_path = os.path.join(self.args.save_dir, "state_{}_last.pt".format(corruption))
        torch.save(state, save_path)