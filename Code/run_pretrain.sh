#!/bin/sh

export PYTHONPATH=$PYTHONPATH:./

# First pre-train the user model with the MLM task
pretrain_method="MLM"
exp_name=${pretrain_method}_Pretrain
py_file=./${pretrain_method}/pretrain.py
python -u ${py_file} --exp_name ${exp_name} --epoch 20

# Then pre-train the user model with the self-supervised ranking task
pretrain_method="AdaptSSR"
exp_name=${pretrain_method}_Pretrain
py_file=./${pretrain_method}/pretrain.py
pretrain_model_path="../model_all/MLM_Pretrain/epoch-20.pt"
python -u ${py_file} --exp_name ${exp_name} --epoch 20 --pretrain_model_path ${pretrain_model_path}