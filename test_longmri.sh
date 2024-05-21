#!/bin/bash

python test_convlstm.py --name VS_whole64_convlstmlarge_fold0 --fold 0 --hr_depth 5 --hr_width 64 --latent_dim 256  --phase test \
    --results_dir /exports/lkeb-hpc/ychen/03_result/04_longMri/VS_whole64_convlstmlarge_fold0
