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

from dataset import FinetuneTopNDatasetTrain, FinetuneTopNDatasetVal
from topN_model import FinetuneModel
from utils import str2bool, dump_args, topN_metrics

assert "1.12.1" in torch.__version__

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default=None)
parser.add_argument("--base_model", type=str, default="BERT4Rec", choices=["BERT4Rec"])
parser.add_argument(
    "--finetune_task", type=str, default=None, choices=["click", "like"]
)
parser.add_argument("--data_dir", type=str, default="../Data")
parser.add_argument("--model_dir", type=str, default="../model_all")
parser.add_argument("--log_dir", type=str, default="../log_all")
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epoch", type=int, default=5)
parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--log_step", type=int, default=500)
parser.add_argument("--pretrain_method", type=str, default=None)
parser.add_argument("--pretrain_model_path", type=str, default=None)
parser.add_argument("--max_len", type=int, default=150)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=2)
parser.add_argument("--pooling", type=str, default="attn", choices=["attn"])
parser.add_argument("--trm_dropout", type=float, default=0.5)
parser.add_argument("--num_pooling", type=int, default=1)
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

if args.finetune_task == "click":
    args.n_labels = 17879
    args.lr = 5e-4
    args.batch_size = 512
elif args.finetune_task == "like":
    args.n_labels = 7539
    args.lr = 5e-4
    args.batch_size = 128

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

    train_cache_path = os.path.join(
        args.data_dir, args.finetune_task, f"finetune_train.pkl"
    )
    if os.path.exists(train_cache_path):
        with open(train_cache_path, "rb") as f:
            train_data = pickle.load(f)
            train_uid = train_data["uid"]
            train_target = train_data["target"]
            train_labels = train_data["labels"]
        log_queue.put(f"Load training data from {train_cache_path}")
    else:
        with open(
            os.path.join(args.data_dir, args.finetune_task, f"finetune_train.csv"),
            "r",
        ) as f:
            train_lines = f.readlines()
        train_uid, train_target, train_labels = [], [], []
        for line in tqdm(train_lines):
            uid, target, labels = line.strip("\n").split(" ")
            labels = set([int(x) for x in labels.split(",")])
            train_uid.append(int(uid))
            train_target.append(int(target))
            train_labels.append(labels)
        train_uid = np.array(train_uid, dtype=np.int32)
        train_target = np.array(train_target, dtype=np.int32)
        with open(train_cache_path, "wb") as f:
            pickle.dump(
                {"uid": train_uid, "target": train_target, "labels": train_labels},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        log_queue.put(f"Dump training data to {train_cache_path}")

    batch_per_epoch = math.ceil(len(train_target) / args.batch_size)
    log_queue.put(f"training sample: {len(train_target)}, {batch_per_epoch} batch")

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    train_ds = FinetuneTopNDatasetTrain(
        total_behavior, train_uid, train_target, train_labels, args.n_labels
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinetuneModel(args, device)

    if args.pretrain_model_path is not None:
        log_queue.put(f"Loading pre-trained model from {args.pretrain_model_path}")
        load_ckpt = torch.load(args.pretrain_model_path, map_location="cpu")
        model.base_model.load_state_dict(load_ckpt)
        tmp_ckpt = {}
        for k, v in load_ckpt.items():
            k = "base_model." + k
            tmp_ckpt[k] = v
        load_ckpt = tmp_ckpt
    else:
        load_ckpt = {}

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

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
    unexpected_param = "\n\t".join(
        [f"{k} {v.shape}" for k, v in load_ckpt.items() if k not in model.state_dict()]
    )
    log_queue.put(str(model))
    log_queue.put(f"Load parameters:\n\t{load_param}")
    log_queue.put(f"Initialize parameters:\n\t{initialize_param}")
    log_queue.put(f"Unexpected parameters:\n\t{unexpected_param}")

    step = 0
    pbar = tqdm(total=len(train_dl), position=0, leave=True, desc="Train".rjust(5))
    for epoch in range(args.epoch):
        log_queue.put(f"===== Epoch {epoch + 1} =====")
        pbar.reset()
        for behavior, label, neg in train_dl:
            step += 1
            behavior = behavior.to(device, non_blocking=True).long()
            label = label.to(device, non_blocking=True).long()
            neg = neg.to(device, non_blocking=True).long()
            batch_loss, batch_hit = model(behavior, label, neg)
            batch_acc = batch_hit / len(label)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            pbar.update(1)

            if step % args.log_step == 0:
                log_queue.put(
                    f"Step: {step}, Loss: {batch_loss:.5f}, Acc: {batch_acc:.5f}"
                )

        model_path = os.path.join(args.model_dir, f"epoch-{epoch + 1}.pt")
        torch.save(model.state_dict(), model_path)
        log_queue.put(f"Model saved at {model_path}")
        pbar.refresh()
    pbar.close()


# %%
def evaluate(args, log_queue):
    batch_size = 512

    total_behavior_path = os.path.join(args.data_dir, "pretrain_remap.pkl")
    while not os.path.exists(total_behavior_path):
        time.sleep(30)
    with open(total_behavior_path, "rb") as f:
        total_behavior = pickle.load(f)
    log_queue.put(f"Load total behavior from {total_behavior_path}")

    val_cache_path = os.path.join(args.data_dir, args.finetune_task, "finetune_val.pkl")
    test_cache_path = os.path.join(
        args.data_dir, args.finetune_task, "finetune_test.pkl"
    )
    if os.path.exists(val_cache_path):
        with open(val_cache_path, "rb") as f:
            val_data = pickle.load(f)
            val_uid = val_data["uid"]
            val_target = val_data["target"]
            val_labels = val_data["labels"]
        log_queue.put(f"Load validation data from {val_cache_path}")
    else:
        with open(
            os.path.join(args.data_dir, args.finetune_task, "finetune_val.csv"), "r"
        ) as f:
            val_lines = f.readlines()
        val_uid, val_target, val_labels = [], [], []
        for line in tqdm(val_lines):
            uid, target, labels = line.strip("\n").split(" ")
            labels = [int(x) for x in labels.split(",")]
            val_uid.append(int(uid))
            val_target.append(int(target))
            val_labels.append(labels)
        val_uid = np.array(val_uid, dtype=np.int32)
        val_target = np.array(val_target, dtype=np.int32)
        with open(val_cache_path, "wb") as f:
            pickle.dump(
                {"uid": val_uid, "target": val_target, "labels": val_labels},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        log_queue.put(f"Dump validation data to {val_cache_path}")

    if os.path.exists(test_cache_path):
        with open(test_cache_path, "rb") as f:
            test_data = pickle.load(f)
            test_uid = test_data["uid"]
            test_target = test_data["target"]
            test_labels = test_data["labels"]
        log_queue.put(f"Load test data from {test_cache_path}")
    else:
        with open(
            os.path.join(args.data_dir, args.finetune_task, "finetune_test.csv"), "r"
        ) as f:
            test_lines = f.readlines()
        test_uid, test_target, test_labels = [], [], []
        for line in tqdm(test_lines):
            uid, target, labels = line.strip("\n").split(" ")
            labels = [int(x) for x in labels.split(",")]
            test_uid.append(int(uid))
            test_target.append(int(target))
            test_labels.append(labels)
        test_uid = np.array(test_uid, dtype=np.int32)
        test_target = np.array(test_target, dtype=np.int32)
        with open(test_cache_path, "wb") as f:
            pickle.dump(
                {"uid": test_uid, "target": test_target, "labels": test_labels},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        log_queue.put(f"Dump test data to {test_cache_path}")

    log_queue.put(
        f"validation sample: {len(val_target)}, {math.ceil(len(val_target) / args.batch_size)} batch"
    )
    log_queue.put(
        f"test sample: {len(test_target)}, {math.ceil(len(test_target) / args.batch_size)} batch"
    )

    def collate_fn(data):
        batch_behavior, batch_target = [], []
        mask_idx_i, mask_idx_j = [], []
        for i, (behavior, target, labels) in enumerate(data):
            batch_behavior.append(behavior)
            batch_target.append(target)
            mask_idx_i.extend([i] * len(labels))
            mask_idx_j.extend(labels)
        batch_behavior = torch.LongTensor(np.array(batch_behavior))
        batch_target = torch.LongTensor(np.array(batch_target))
        batch_mask_idx = [mask_idx_i, mask_idx_j]
        return batch_behavior, batch_target, batch_mask_idx

    val_ds = FinetuneTopNDatasetVal(total_behavior, val_uid, val_target, val_labels)
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_ds = FinetuneTopNDatasetVal(total_behavior, test_uid, test_target, test_labels)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinetuneModel(args, device)
    model = model.to(device)

    model.eval()
    torch.set_grad_enabled(False)

    val_pbar = tqdm(total=len(val_dl), position=1, leave=True, desc="Val".rjust(5))
    best_val_ndcg10 = -1
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
                    total_mrr5, total_mrr10, total_mrr20 = 0, 0, 0
                    total_hr5, total_hr10, total_hr20 = 0, 0, 0
                    total_ndcg5, total_ndcg10, total_ndcg20 = 0, 0, 0
                    for behavior, label, mask in val_dl:
                        behavior = behavior.to(device, non_blocking=True)
                        label = label.to(device, non_blocking=True)
                        batch_logits = model(behavior, mask=mask)
                        sorted_batch_logits = torch.argsort(
                            batch_logits, dim=-1, descending=True
                        )
                        hr5, mrr5, ndcg5 = topN_metrics(
                            sorted_batch_logits[:, :5], label
                        )
                        hr10, mrr10, ndcg10 = topN_metrics(
                            sorted_batch_logits[:, :10], label
                        )
                        hr20, mrr20, ndcg20 = topN_metrics(
                            sorted_batch_logits[:, :20], label
                        )
                        total_mrr5 += mrr5
                        total_mrr10 += mrr10
                        total_mrr20 += mrr20
                        total_hr5 += hr5
                        total_hr10 += hr10
                        total_hr20 += hr20
                        total_ndcg5 += ndcg5
                        total_ndcg10 += ndcg10
                        total_ndcg20 += ndcg20
                        val_pbar.update(1)
                    total_sample = len(val_target)
                    val_mrr5 = total_mrr5 / total_sample
                    val_mrr10 = total_mrr10 / total_sample
                    val_mrr20 = total_mrr20 / total_sample
                    val_hr5 = total_hr5 / total_sample
                    val_hr10 = total_hr10 / total_sample
                    val_hr20 = total_hr20 / total_sample
                    val_ndcg5 = total_ndcg5 / total_sample
                    val_ndcg10 = total_ndcg10 / total_sample
                    val_ndcg20 = total_ndcg20 / total_sample
                    val_pbar.refresh()

                    log_queue.put(
                        f"[Val] Epoch: {epoch}, MRR@5: {val_mrr5:.5f}, MRR@10: {val_mrr10:.5f}, MRR@20: {val_mrr20:.5f}, HR@5: {val_hr5:.5f}, HR@10: {val_hr10:.5f}, HR@20: {val_hr20:.5f}, NDCG@5: {val_ndcg5:.5f}, NDCG@10: {val_ndcg10:.5f}, NDCG@20: {val_ndcg20:.5f}"
                    )

                    if val_ndcg10 > best_val_ndcg10:
                        best_val_ndcg10 = val_ndcg10
                        best_ckpt_epoch = epoch

    best_ckpt_path = os.path.join(args.model_dir, f"epoch-{best_ckpt_epoch}.pt")
    while True:
        try:
            log_queue.put(f"Loading best ckpt from {best_ckpt_path}")
            model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
            break
        except OSError:
            time.sleep(10)

    # test
    total_mrr5, total_mrr10, total_mrr20 = 0, 0, 0
    total_hr5, total_hr10, total_hr20 = 0, 0, 0
    total_ndcg5, total_ndcg10, total_ndcg20 = 0, 0, 0
    for behavior, label, mask in tqdm(
        test_dl, desc="Test".rjust(5), position=2, leave=True
    ):
        behavior = behavior.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        batch_logits = model(behavior, mask=mask)
        sorted_batch_logits = torch.argsort(batch_logits, dim=-1, descending=True)
        hr5, mrr5, ndcg5 = topN_metrics(sorted_batch_logits[:, :5], label)
        hr10, mrr10, ndcg10 = topN_metrics(sorted_batch_logits[:, :10], label)
        hr20, mrr20, ndcg20 = topN_metrics(sorted_batch_logits[:, :20], label)
        total_mrr5 += mrr5
        total_mrr10 += mrr10
        total_mrr20 += mrr20
        total_hr5 += hr5
        total_hr10 += hr10
        total_hr20 += hr20
        total_ndcg5 += ndcg5
        total_ndcg10 += ndcg10
        total_ndcg20 += ndcg20

    total_sample = len(test_target)
    test_mrr5 = total_mrr5 / total_sample
    test_mrr10 = total_mrr10 / total_sample
    test_mrr20 = total_mrr20 / total_sample
    test_hr5 = total_hr5 / total_sample
    test_hr10 = total_hr10 / total_sample
    test_hr20 = total_hr20 / total_sample
    test_ndcg5 = total_ndcg5 / total_sample
    test_ndcg10 = total_ndcg10 / total_sample
    test_ndcg20 = total_ndcg20 / total_sample

    log_queue.put(
        f"Best ckpt epoch: {best_ckpt_epoch}, MRR@5: {test_mrr5:.5f}, MRR@10: {test_mrr10:.5f}, MRR@20: {test_mrr20:.5f}, HR@5: {test_hr5:.5f}, HR@10: {test_hr10:.5f}, HR@20: {test_hr20:.5f}, NDCG@5: {test_ndcg5:.5f}, NDCG@10: {test_ndcg10:.5f}, NDCG@20: {test_ndcg20:.5f}"
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
