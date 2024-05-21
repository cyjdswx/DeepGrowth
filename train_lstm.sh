#!/bin/bash

python train_lstm.py --name VS_whole64_convlstmlarge_correct_fold4 --fold 4 --batchSize 2 --latent_dim 256 \
    --dataset_dir /exports/lkeb-hpc/ychen/01_data/08_VS_followup/longitudinal_t1ce_whole_cropped64_05/ \
    --latent_code_regularization --lambda_ll 1e-4 
    #--pretrained /exports/lkeb-hpc/ychen/02_pythonProject/15_tumor_growth/checkpoints/VS_inrsdf_down_siren_pretrain
    #--pretrained /exports/lkeb-hpc/ychen/02_pythonProject/15_tumor_growth/checkpoints/VS_inrsiren_tensor_pretrain_entire
    #--dataset_dir /exports/lkeb-hpc/ychen/01_data/08_VS_followup/longitudinal_t1ce_crossval_more_largerbbox/ \
    #--dataset_dir /exports/lkeb-hpc/ychen/01_data/08_VS_followup/longitudinal_t1ce_crossval_more_90p_ex/ \
    #--dataset_dir /exports/lkeb-hpc/ychen/01_data/08_VS_followup/longitudinal_t1ce_crossval_more_largerbbox/ \
    #--pretrained /exports/lkeb-hpc/ychen/02_pythonProject/15_tumor_growth/checkpoints/VS_inr_pretrained_tensor_l1_50p_fold2/
