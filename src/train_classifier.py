import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.utils_tta import set_seed
from data.div2kmd import DIV2KMD
from model.classifier import Classifier
import utils.utils_tta as util
import warnings
warnings.filterwarnings("ignore")

def train_classifier(args, classifier, logger, writer,device="cuda"):
    classifier.train()

    train_dataset = DIV2KMD(base_path=args.training_dir,cache=args.cache)
    test_dataset = DIV2KMD(base_path=args.test_dir,train=False, cache=args.cache)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.total_epoch,eta_min=args.lr * args.lf)
    train_dataloader = DataLoader(train_dataset, 
                            batch_size=args.bs, num_workers=args.worker, shuffle=True)
    test_dataloader = DataLoader(test_dataset, 
                            batch_size=1, num_workers=args.worker, shuffle=False)
    best_acc = 0
    eval_record_threds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for epoch in range(args.total_epoch):
        pbar = tqdm(train_dataloader, total=len(train_dataset)//args.bs + 1, ncols=100)
        loss_sum = 0
        idx_iter = 0
        for imgs,labels,paths in pbar:
            imgs = imgs.cuda() # batch[0] = batch[0][0][0]
            labels = labels.cuda() # # batch[1]

            optimizer.zero_grad()
            pred = classifier(imgs)
            loss = classifier.computeLoss(pred, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
            loss_sum += loss
            pbar.set_postfix({"epoch": f"{epoch}", "loss": f"{loss:.2f}"})
            idx_iter += 1
        writer.add_scalar('train/train_loss', loss_sum/len(pbar), epoch)
        
        if epoch % 20 == 0 or epoch == args.total_epoch - 1:
            right_dict = {}
            for eval_record_thred in eval_record_threds:
                right_dict[eval_record_thred] = 0
            classifier.eval()
            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataset),ncols=100)
            sum = 0
            loss_ce_sum = 0
            for _, (imgs, labels,paths) in pbar:
                imgs = imgs.cuda()
                labels = labels.cuda()
                with torch.no_grad():
                    pred = classifier(imgs)
                    loss_ce_sum += classifier.computeLoss(pred, labels)
                    pred = pred.sigmoid()
                    pred = pred[0]
                    for eval_record_thred in eval_record_threds:
                        right_dict[eval_record_thred]+=((labels[0].float().to(device)!=(((pred)>eval_record_thred))).sum().item()==0)
                    sum += imgs.shape[0]
                pbar.set_postfix({"epoch": f"{epoch}","mode": "eval"})
            acc05 = float(right_dict[eval_record_thred]) / float(sum)
            if acc05 > best_acc:
                best_acc=acc05
                torch.save(classifier.state_dict(),os.path.join(args.save_dir, "best.pt"))
            torch.save(classifier.state_dict(),os.path.join(args.save_dir, "last.pt"))
            
            log_txt = f"Epoch {epoch}"
            for eval_record_thred in eval_record_threds:
                acc = float(right_dict[eval_record_thred])/float(sum)
                log_txt += f", acc-{eval_record_thred:.1f}={acc:.4f}"
            logger.info(log_txt)
            writer.add_scalar('val/acc', acc, epoch)
            writer.add_scalar('val/loss_ce', loss_ce_sum/len(pbar), epoch)
            classifier.train()
    return classifier

def evaluate_classifier(args, classifier, device="cuda"):
    classifier.eval()
    test_dataset = DIV2KMD(base_path=args.test_dir,train=False, cache=args.cache)
    test_dataloader = DataLoader(test_dataset, 
                            batch_size=1, num_workers=args.worker, shuffle=False)
    pbar=tqdm(enumerate(test_dataloader), total=len(test_dataset),ncols=100)
    right=0
    sum=0
    time_all=0
    for _,(imgs, labels,paths) in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            start_inference=time.time()
            pred=classifier(imgs)
            time_all+=time.time()-start_inference
            pred_right=((labels[0].to(device)!=(pred.sigmoid().mean(0)>0.5)).sum().item()==0)
            right+=pred_right
            sum+=imgs.shape[0]
        pbar.set_postfix({"epoch": f"{0}","mode": "eval"})
    acc=float(right)/float(sum)
    print(f"accuracy = {acc}")
    print(f"avg time : {time_all/sum}")

def main(args):
    set_seed(args.seed)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    logger = util.get_logger(f"{args.save_dir}/log.txt")
    writer = SummaryWriter(args.save_dir)
    
    classifier = Classifier().cuda()
    
    classifier = train_classifier(args, classifier, logger, writer)
    
    evaluate_classifier(args, classifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dir', type=str, default='../datasets/DIV2K/DIV2K_train_HR', help='dataset directory')
    parser.add_argument('--test_dir', type=str, default='../datasets/DIV2KC/corruptions', help='dataset directory')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lf', type=float, default=0.01, help='lr final=lr*lf')
    parser.add_argument('--worker', type=int, default=8, help='number of workers')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproduction')
    parser.add_argument('--total_epoch', type=int, default=300, help='total training epoch')
    parser.add_argument('--save_dir', type=str, default='../expriment/degradation_classifier/', help='exp name')
    parser.add_argument('--train_dtypes', type=str, default='single+multi', help='degradation types for preparing training data')
    parser.add_argument('--cache', action="store_true", help='cache images')
    parser.add_argument('--img_size', type=int, default=224, help='the size of training images')
    parser.add_argument('--shuffle',  action="store_true", help='shuffle the order of precoss images')
    args = parser.parse_args()

    main(args)
