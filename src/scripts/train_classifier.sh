CUDA_VISIBLE_DEVICES=0 python train_classifier.py --bs 256 --total_epoch 400 \
--save_dir experiments/degradation_classifier \
--lr 0.001 --cache --lf 0.001 --train_dtypes single+multi --img_size 224