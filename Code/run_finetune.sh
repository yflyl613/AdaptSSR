#!/bin/sh

export PYTHONPATH=$PYTHONPATH:./

pretrain_method="AdaptSSR"
py_file=./finetune_classification.py
finetune_task='age'
exp_name="${pretrain_method}_${finetune_task}"
pretrain_model_path="../model_all/${pretrain_method}_Pretrain/epoch-1.pt"
python -u ${py_file} --exp_name ${exp_name} --finetune_task ${finetune_task} --pretrain_model_path ${pretrain_model_path} --pretrain_method ${pretrain_method}
