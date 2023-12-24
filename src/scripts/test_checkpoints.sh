export CUDA_VISIBLE_DEVICES=6
# x2 lifelong
python test_srtta.py --test_only --base_model checkpoints/EDSR_baseline_x2.pt --dir_data ../datasets \
--pre_train checkpoints/reproduce/SRTTA_lifelong_DIV2KC_x2 \
--cls_model checkpoints/classifier.pt

# x2 parameter-reset
python test_srtta.py --test_only --base_model checkpoints/EDSR_baseline_x2.pt --dir_data ../datasets \
--pre_train checkpoints/reproduce/SRTTA_reset_DIV2KC_x2 \
--cls_model checkpoints/classifier.pt

# x2 lifelong multi-corruptions
python test_srtta.py --test_only --base_model checkpoints/EDSR_baseline_x2.pt --dir_data ../datasets \
--pre_train checkpoints/reproduce/SRTTA_lifelong_DIV2KMC_x2 --corruptions multi \
--cls_model checkpoints/classifier.pt --tta_data DIV2KMC --data_test DIV2KMC

# x2 parameter-reset multi-corruptions
python test_srtta.py --test_only --base_model checkpoints/EDSR_baseline_x2.pt --dir_data ../datasets \
--pre_train checkpoints/reproduce/SRTTA_reset_DIV2KMC_x2 --corruptions multi \
--cls_model checkpoints/classifier.pt --tta_data DIV2KMC --data_test DIV2KMC

# x4 lifelong
python test_srtta.py --test_only --base_model checkpoints/EDSR_baseline_x4.pt --dir_data ../datasets \
--pre_train checkpoints/reproduce/SRTTA_lifelong_DIV2KC_x4 \
--scale 4 --cls_model checkpoints/classifier.pt

# x4 parameter-reset
python test_srtta.py --test_only --base_model checkpoints/EDSR_baseline_x4.pt --dir_data ../datasets \
--pre_train checkpoints/reproduce/SRTTA_reset_DIV2KC_x4 \
--scale 4 --cls_model checkpoints/classifier.pt