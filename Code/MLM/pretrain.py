# %%
import argparse
import os
import random
import torch
import numpy as np
import pickle
import math
import time
import logging
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import Queue, Process

from MLM.dataset import PretrainDataset
from MLM.model import PretrainModel
from utils import dump_args

assert "1.12.1" in torch.__version__

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default=None)
parser.add_argument("--base_model", type=str, default="BERT4Rec")
parser.add_argument("--data_dir", type=str, default="../Data")
parser.add_argument("--model_dir", type=str, default="../model_all")
parser.add_argument("--log_dir", type=str, default="../log_all")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--log_step", type=int, default=1000)
parser.add_argument("--max_len", type=int, default=150)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=2)
parser.add_argument("--pooling", type=str, default=None)
parser.add_argument("--trm_dropout", type=float, default=0.5)
parser.add_argument("--mask_ratio", type=float, default=0.3)

args = parser.parse_args()
args.model_dir = os.path.join(args.model_dir, args.exp_name)
args.log_path = os.path.join(args.log_dir, f"{args.exp_name}.txt")
os.makedirs(args.log_dir, exist_ok=True)

format = "[%(asctime)s] %(levelname)s - %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    filename=args.log_path,
    filemode="w",
    format=format,
    level=logging.INFO,
    datefmt=datefmt,
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter(format, datefmt))
logger.addHandler(console_handler)

logging.warning(f"Running experiment {args.exp_name}")

if os.path.exists(args.model_dir):
    logging.warning("Remove existing model dir in 10s")
    time.sleep(10)
    shutil.rmtree(args.model_dir)
os.makedirs(args.model_dir, exist_ok=True)

with open(os.path.join(args.data_dir, "item_map.pkl"), "rb") as f:
    item_map = pickle.load(f)
args.n_items = len(item_map)

dump_args(args)

# %%
def train(args, log_queue):
    total_behavior_path = os.path.join(args.data_dir, "pretrain_remap.pkl")
    if os.path.exists(total_behavior_path):
        with open(total_behavior_path, "rb") as f:
            total_behavior = pickle.load(f)
        log_queue.put(f"Load total behavior from {total_behavior_path}")
    else:
        with open(os.path.join(args.data_dir, "pretrain_remap.csv")) as f:
            total_lines = f.readlines()
        total_behavior = []
        for line in tqdm(total_lines):
            behavior = [int(x) for x in line.strip("\n").split(",") if x != "0"]
            behavior = behavior + [0] * (args.max_len - len(behavior))
            total_behavior.append(behavior)
        total_behavior = np.array(total_behavior, dtype=np.int32)
        with open(total_behavior_path, "wb") as f:
            pickle.dump(total_behavior, f, protocol=pickle.HIGHEST_PROTOCOL)
        log_queue.put(f"Dump total behavior to {total_behavior_path}")

    for epoch in range(args.epoch):
        train_cache_path = os.path.join(args.data_dir, "pretrain_train", f"{epoch}.pkl")
        if not os.path.exists(train_cache_path):
            train_file_path = os.path.join(
                args.data_dir, "pretrain_train", f"{epoch}.csv"
            )
            with open(train_file_path, "r") as f:
                train_lines = f.readlines()
            train_uid = []
            for line in tqdm(train_lines):
                train_uid.append(int(line.strip("\n")))
            train_uid = np.array(train_uid, dtype=np.int32)
            with open(train_cache_path, "wb") as f:
                pickle.dump(train_uid, f, protocol=pickle.HIGHEST_PROTOCOL)
            log_queue.put(f"Dump training data to {train_cache_path}")

    train_cache_path = os.path.join(args.data_dir, "pretrain_train", "0.pkl")
    with open(train_cache_path, "rb") as f:
        train_uid = pickle.load(f)

    num_sample = len(train_uid)
    log_queue.put(
        f"training sample: {num_sample}, {math.ceil(num_sample / args.batch_size)} batch"
    )

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainModel(args, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    log_queue.put(str(model))

    step = 0
    pbar = None
    for epoch in range(args.epoch):
        logging.warning(f"===== Epoch {epoch + 1} =====")
        train_data_cache_path = os.path.join(
            args.data_dir, "pretrain_train", f"{epoch}.pkl"
        )
        log_queue.put(f"Load training data from {train_data_cache_path}")
        train_ds = PretrainDataset(total_behavior, train_data_cache_path, args)
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=collate_fn,
        )
        if pbar is None:
            pbar = tqdm(
                total=len(train_dl), position=0, leave=True, desc="Train".rjust(5)
            )
        else:
            pbar.reset()
        for behavior, mask_idx in train_dl:
            step += 1
            behavior = behavior.to(device, non_blocking=True)
            mlm_loss, mlm_hit = model(behavior, mask_idx)
            mlm_acc = mlm_hit / len(mask_idx[0])
            optimizer.zero_grad()
            mlm_loss.backward()
            optimizer.step()
            pbar.update(1)

            if step % args.log_step == 0:
                log_queue.put(
                    f"Step: {step}, MLM_Loss: {mlm_loss:.8f}, MLM_Acc: {mlm_acc:.8f}"
                )

        model_path = os.path.join(args.model_dir, f"epoch-{epoch + 1}.pt")
        torch.save(model.state_dict(), model_path)
        log_queue.put(f"Model saved at {model_path}")
        pbar.refresh()
    pbar.close()


# %%
def evaluate(args, log_queue):
    total_behavior_path = os.path.join(args.data_dir, "pretrain_remap.pkl")
    while not os.path.exists(total_behavior_path):
        time.sleep(30)
    with open(total_behavior_path, "rb") as f:
        total_behavior = pickle.load(f)
    log_queue.put(f"Load total behavior from {total_behavior_path}")

    val_cache_path = os.path.join(args.data_dir, "pretrain_val.pkl")
    if os.path.exists(val_cache_path):
        with open(val_cache_path, "rb") as f:
            val_uid = pickle.load(f)
        log_queue.put(f"Load validation data from {val_cache_path}")
    else:
        with open(os.path.join(args.data_dir, "pretrain_val.csv"), "r") as f:
            val_lines = f.readlines()
        val_uid = []
        for line in tqdm(val_lines):
            val_uid.append(int(line.strip("\n")))
        val_uid = np.array(val_uid, dtype=np.int32)
        with open(val_cache_path, "wb") as f:
            pickle.dump(val_uid, f, protocol=pickle.HIGHEST_PROTOCOL)
        log_queue.put(f"Dump validation data to {val_cache_path}")

    log_queue.put(
        f"validation sample: {len(val_uid)}, {math.ceil(len(val_uid) / args.batch_size)} batch"
    )

    val_ds = PretrainDataset(total_behavior, val_cache_path, args)
    val_dl = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainModel(args, device)
    model = model.to(device)

    model.eval()
    torch.set_grad_enabled(False)

    val_pbar = tqdm(total=len(val_dl), position=1, leave=True, desc="Val".rjust(5))
    best_val_acc = -1
    checked_ckpts = []
    while len(checked_ckpts) < args.epoch:
        current_ckpts = os.listdir(args.model_dir)
        current_ckpts.sort(key=lambda x: int(x.split(".")[0].split("-")[-1]))
        if len(checked_ckpts) == len(current_ckpts):
            time.sleep(30)
        else:
            for ckpt in current_ckpts:
                if ckpt not in checked_ckpts:
                    checked_ckpts.append(ckpt)
                    epoch = int(ckpt.split(".")[0].split("-")[-1])
                    ckpt_path = os.path.join(args.model_dir, ckpt)
                    while True:
                        try:
                            log_queue.put(f"Loading ckpt from {ckpt_path}")
                            model.load_state_dict(
                                torch.load(ckpt_path, map_location="cpu")
                            )
                            break
                        except:
                            time.sleep(10)
                    val_pbar.reset()

                    # val
                    total_num, total_hit = 0, 0
                    for behavior, mask_idx in val_dl:
                        behavior = behavior.to(device, non_blocking=True)
                        mlm_hit = model(behavior, mask_idx, False)
                        total_num += len(mask_idx[0])
                        total_hit += mlm_hit
                        val_pbar.update(1)
                    val_acc = total_hit / total_num
                    val_pbar.refresh()

                    log_queue.put(f"[Val] Epoch: {epoch}, MLM_Acc: {val_acc:.8f}")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_ckpt_epoch = epoch

    log_queue.put(f"Best ckpt epoch: {best_ckpt_epoch}, Acc: {best_val_acc:.8f}")
    val_pbar.close()


# %%
def log(log_queue):
    while True:
        log_data = log_queue.get()
        if log_data is None:
            break
        else:
            logging.info(log_data)


# %%
def collate_fn(data):
    batch_behavior = []
    mask_idx_i, mask_idx_j = [], []
    for i, (behavior, mask_idx) in enumerate(data):
        batch_behavior.append(behavior)
        mask_idx_i.extend([i] * len(mask_idx))
        mask_idx_j.extend(mask_idx)
    batch_behavior = torch.LongTensor(np.array(batch_behavior))
    batch_mask_idx = [mask_idx_i, mask_idx_j]
    return batch_behavior, batch_mask_idx


# %%
log_queue = Queue()
log_process = Process(target=log, args=(log_queue,))
train_process = Process(target=train, args=(args, log_queue))
eval_process = Process(target=evaluate, args=(args, log_queue))

log_process.start()
train_process.start()
eval_process.start()

train_process.join()
train_process.close()
eval_process.join()
eval_process.close()

log_queue.put(None)
log_process.join()
log_process.close()
