CUDA_VISIBLE_DEVICES=4 python main_tta.py --exp_name tta_x4_parameter_reset --lr 5e-5 \
--iterations 10 --batch_size 32 --patch_size 64 --pre_train checkpoints/EDSR_baseline_x4.pt \
--teacher_weight 1 --fisher_restore --fisher_ratio 0.5  --params_reset \
--scale 4 --tta_data DIV2KC

CUDA_VISIBLE_DEVICES=4 python main_tta.py --exp_name tta_x4_lifelong --lr 5e-5 \
--iterations 10 --batch_size 32 --patch_size 64 --pre_train checkpoints/EDSR_baseline_x4.pt \
--teacher_weight 1 --fisher_restore --fisher_ratio 0.5  \
--scale 4 --tta_data DIV2KC