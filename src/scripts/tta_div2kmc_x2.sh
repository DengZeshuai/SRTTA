CUDA_VISIBLE_DEVICES=1 python main_tta.py --exp_name SRTTA_lifelong_DIV2KMC_x2 --lr 5e-5 \
--iterations 10 --batch_size 32 --patch_size 64 --pre_train checkpoints/EDSR_baseline_x2.pt \
--teacher_weight 1 --fisher_restore --fisher_ratio 0.5  --params_reset \
--scale 2 --tta_data DIV2KMC --data_test DIV2KMC+Set5

# For lifelong setting, we first adapt to DIV2K-C, and then use the adapted model to adapt to DIV2K-MC
CUDA_VISIBLE_DEVICES=1 python main_tta.py --exp_name SRTTA_reset_DIV2KMC_x2 --lr 5e-5 \
--iterations 10 --batch_size 32 --patch_size 64 --pre_train $PATH_TO_ADAPTED_CHECKPOINTS_IN_DIV2KC \
--teacher_weight 1 --fisher_restore --fisher_ratio 0.5  \
--scale 2 --tta_data DIV2KMC --data_test DIV2KMC+Set5