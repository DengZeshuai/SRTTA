CUDA_VISIBLE_DEVICES=1 python main_tta.py --exp_name SRTTA_reset_DIV2KC_x2 --lr 5e-5 \
--iterations 10 --batch_size 32 --patch_size 64 --pre_train checkpoints/EDSR_baseline_x2.pt \
--teacher_weight 1 --fisher_restore --fisher_ratio 0.5 --params_reset \
--scale 2 --tta_data DIV2KC

CUDA_VISIBLE_DEVICES=1 python main_tta.py --exp_name SRTTA_lifelong_DIV2KC_x2 --lr 5e-5 \
--iterations 10 --batch_size 32 --patch_size 64 --pre_train checkpoints/EDSR_baseline_x2.pt \
--teacher_weight 1 --fisher_restore --fisher_ratio 0.5  \
--scale 2 --tta_data DIV2KC