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

from AdaptSSR.dataset import PretrainDataset
from AdaptSSR.model import PretrainModel
from utils import dump_args, str2bool

assert "1.12.1" in torch.__version__

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default=None)
parser.add_argument("--base_model", type=str, default="BERT4Rec")
parser.add_argument("--data_dir", type=str, default="../Data")
parser.add_argument("--model_dir", type=str, default="../model_all")
parser.add_argument("--log_dir", type=str, default="../log_all")
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--log_step", type=int, default=1000)
parser.add_argument("--max_len", type=int, default=150)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=2)
parser.add_argument("--pooling", type=str, default="attn", choices=["attn"])
parser.add_argument("--num_pooling", type=int, default=1)
parser.add_argument("--trm_dropout", type=float, default=0.1)
parser.add_argument("--crop_ratio", type=float, default=0.4)
parser.add_argument("--mask_ratio", type=float, default=0.6)
parser.add_argument("--reorder_ratio", type=float, default=0.6)
parser.add_argument("--pretrain_model_path", type=str, default=None)
parser.add_argument("--reg_coef", type=float, default=1)
parser.add_argument("--cos_pooling", type=str2bool, default=False)

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

    if args.pretrain_model_path is not None:
        log_queue.put(f"Loading pre-trained model from {args.pretrain_model_path}")
        load_ckpt = torch.load(args.pretrain_model_path, map_location="cpu")
        model.load_state_dict(load_ckpt, strict=False)
    else:
        load_ckpt = {}

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    log_queue.put(str(model))

    load_param = "\n\t".join(
        [
            f"{k} {v.shape} {v.requires_grad}"
            for k, v in model.named_parameters()
            if k in load_ckpt
        ]
    )
    initialize_param = "\n\t".join(
        [
            f"{k} {v.shape} {v.requires_grad}"
            for k, v in model.named_parameters()
            if k not in load_ckpt
        ]
    )
    log_queue.put(f"Load parameters:\n\t{load_param}")
    log_queue.put(f"Initialize parameters:\n\t{initialize_param}")

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
        for item_seq_1, item_seq_2, mask_idx_1, mask_idx_2 in train_dl:
            step += 1
            item_seq_1 = item_seq_1.to(device, non_blocking=True)
            item_seq_2 = item_seq_2.to(device, non_blocking=True)
            (
                batch_loss,
                hit_pos_semi,
                hit_semi_neg,
            ) = model(item_seq_1, item_seq_2, mask_idx_1, mask_idx_2)
            bz = len(item_seq_1)
            acc_pos_semi = hit_pos_semi / (bz * 2)
            acc_semi_neg = hit_semi_neg / (bz * 2 * 4 * (bz - 1))
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            pbar.update(1)

            if step % args.log_step == 0:
                log_queue.put(
                    f"Step: {step}, Loss: {batch_loss:.5f}, Acc_pos_semi: {acc_pos_semi:.5f}, Acc_semi_neg: {acc_semi_neg:.5f}"
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
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainModel(args, device)
    model = model.to(device)

    model.eval()
    for m in model.base_model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
    torch.set_grad_enabled(False)

    val_pbar = tqdm(total=len(val_dl), position=1, leave=True, desc="Val".rjust(5))
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
                        except (RuntimeError, OSError):
                            time.sleep(10)
                    val_pbar.reset()

                    # val
                    total_num, total_hit_pos_semi, total_hit_semi_neg, total_cnt = (
                        0,
                        0,
                        0,
                        0,
                    )
                    for behavior_1, behavior_2, mask_idx_1, mask_idx_2 in val_dl:
                        behavior_1 = behavior_1.to(device, non_blocking=True)
                        behavior_2 = behavior_2.to(device, non_blocking=True)
                        _, hit_pos_semi, hit_semi_neg = model(
                            behavior_1, behavior_2, mask_idx_1, mask_idx_2
                        )
                        bz = len(behavior_1)
                        total_num += bz * 2
                        total_cnt += bz * 2 * 4 * (bz - 1)
                        total_hit_pos_semi += hit_pos_semi.detach().cpu()
                        total_hit_semi_neg += hit_semi_neg.detach().cpu()
                        val_pbar.update(1)

                    val_acc_pos_semi = total_hit_pos_semi / total_num
                    val_acc_semi_neg = total_hit_semi_neg / total_cnt
                    val_pbar.refresh()

                    log_queue.put(
                        f"[Val] Epoch: {epoch}, Acc_Pos_Semi: {val_acc_pos_semi:.5f}, Acc_Semi_Neg: {val_acc_semi_neg:.5f}"
                    )

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
    batch_behavior_1, batch_behavior_2 = [], []
    mask_idx_1_i, mask_idx_1_j = [], []
    mask_idx_2_i, mask_idx_2_j = [], []
    for i, (behavior_1, behavior_2, mask_idx_1, mask_idx_2) in enumerate(data):
        batch_behavior_1.append(behavior_1)
        batch_behavior_2.append(behavior_2)
        mask_idx_1_i.extend([i] * len(mask_idx_1))
        mask_idx_1_j.extend(mask_idx_1)
        mask_idx_2_i.extend([i] * len(mask_idx_2))
        mask_idx_2_j.extend(mask_idx_2)
    batch_behavior_1 = torch.LongTensor(np.array(batch_behavior_1))
    batch_behavior_2 = torch.LongTensor(np.array(batch_behavior_2))
    batch_mask_idx_1 = [mask_idx_1_i, mask_idx_1_j]
    batch_mask_idx_2 = [mask_idx_2_i, mask_idx_2_j]
    return batch_behavior_1, batch_behavior_2, batch_mask_idx_1, batch_mask_idx_2


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
