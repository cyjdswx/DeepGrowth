#!/bin/bash

python train_longpntnet.py --name VS_deepgrowth --fold 0 --batchSize 8 --latent_dim 256 \
    --dataset_dir /exports/lkeb-hpc/ychen/01_data/08_VS_followup/longitudinal_t1ce_whole_cropped64_05/ \
    --latent_code_regularization --lambda_ll 1e-2 \
    --pretrained /exports/lkeb-hpc/ychen/02_pythonProject/15_tumor_growth/checkpoints/VS_whole64x4_sdf_05_pretrained
    #--pretrained /exports/lkeb-hpc/ychen/02_pythonProject/15_tumor_growth/checkpoints/VS_whole64_sdf_x1_64_pretrained
    #--pretrained /exports/lkeb-hpc/ychen/02_pythonProject/15_tumor_growth/checkpoints/VS_whole64x8_sdf_pretrained
    #--pretrained /exports/lkeb-hpc/ychen/02_pythonProject/15_tumor_growth/checkpoints/VS_inrsiren_tensor_pretrain_entire
    #--dataset_dir /exports/lkeb-hpc/ychen/01_data/08_VS_followup/longitudinal_t1ce_crossval_more_largerbbox/ \
    #--dataset_dir /exports/lkeb-hpc/ychen/01_data/08_VS_followup/longitudinal_t1ce_crossval_more_90p_ex/ \
    #--dataset_dir /exports/lkeb-hpc/ychen/01_data/08_VS_followup/longitudinal_t1ce_crossval_more_largerbbox/ \
    #--pretrained /exports/lkeb-hpc/ychen/02_pythonProject/15_tumor_growth/checkpoints/VS_inr_pretrained_tensor_l1_50p_fold2/
