#!/bin/bash
export PYTHONPATH=/home/usl/Code/PengJiang/RELLIS-3D/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/:$PYTHONPATH
python tools/test.py --cfg experiments/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-3_wd5e-4_bs_12_epoch484.yaml \
                     --data-cfg  /home/furkan/RELLIS-3D/benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml \
                     DATASET.TEST_SET test.lst \
                     OUTPUT_DIR output \
                     TEST.MODEL_FILE /home/furkan/RELLIS-3D/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/pretrained_models/hrnet_best/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/best.pth
