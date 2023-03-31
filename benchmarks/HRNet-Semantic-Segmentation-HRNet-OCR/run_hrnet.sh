#!/bin/bash
export PYTHONPATH=/home/usl/Code/Peng/data_collection/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/:$PYTHONPATH
echo $PYTHONPATH
torchrun tools/train.py --cfg experiments/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-3_wd5e-4_bs_12_epoch484.yaml --local_rank 0
