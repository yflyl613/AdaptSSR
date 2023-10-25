# AdaptSSR
The source code and data for our paper "AdaptSSR: Pre-training User Model with Augmentation-Adaptive Self-Supervised Ranking" in NeurIPS 2023.

## Requirements
- PyTorch == 1.12.1
- pickle
- tqdm

## Get Started
- **Prepare Data**
  - Download the [Tecent Transfer Learning (TTL) dataset](https://drive.google.com/file/d/1imhHUsivh6oMEtEW-RwVc4OsDqn-xOaP/view) and put it under `Data/`.
  - Run the command `bash prepare_data.sh` under `Code/`. The script will unzip and preprocess the TTL dataset for experiements.
- **Run Experiments**
  - `Code/run_pretrain.sh` is the script for user model pre-training. You can modify the value of hyper-parameters to change the setting of experiments. Please refer to `Code/AdaptSSR/pretrain.py` for more options.
  - `Code/run_finetune.sh` is the script for fine-tuning the pre-trained user model on downstream tasks. You can modify the value of `finetune_task` to select the downstream task. Please refer to `Code/finetune_classification.py` for more options.

