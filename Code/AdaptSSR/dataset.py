import math
import pickle
import numpy as np
import random
from copy import copy
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, total_behavior, uid_path, args):
        super().__init__()
        self.rng = random.Random(1901)
        self.total_behavior = total_behavior

        with open(uid_path, "rb") as f:
            self.uid = pickle.load(f)

        self.max_len = args.max_len
        self.crop_ratio = args.crop_ratio
        self.mask_ratio = args.mask_ratio
        self.reorder_ratio = args.reorder_ratio

    def __getitem__(self, idx):
        uid = self.uid[idx]
        behavior = self.total_behavior[uid]
        seq_len = (behavior != 0).sum()
        behavior_1 = behavior[:seq_len].tolist()
        behavior_2 = copy(behavior_1)
        candidate_func = [self.crop, self.reorder]
        if seq_len > 1:
            candidate_func.append(self.mask)
        aug_func_1 = self.rng.choice(candidate_func)
        aug_func_2 = self.rng.choice(candidate_func)
        item_seq_1, mask_idx_1 = aug_func_1(behavior_1, seq_len)
        item_seq_2, mask_idx_2 = aug_func_2(behavior_2, seq_len)
        return item_seq_1, item_seq_2, mask_idx_1, mask_idx_2

    def crop(self, x, seq_len):
        crop_length = math.ceil(self.crop_ratio * seq_len)
        start_idx = self.rng.randint(0, seq_len - crop_length)
        return self.pad(x[start_idx : start_idx + crop_length]), []

    def mask(self, x, seq_len):
        mask_num = math.ceil(self.mask_ratio * seq_len)
        if mask_num == seq_len:
            mask_num = math.floor(self.mask_ratio * seq_len)
        mask_idx = self.rng.sample(range(seq_len), mask_num)
        return self.pad(x), mask_idx

    def reorder(self, x, seq_len):
        reorder_length = math.ceil(self.reorder_ratio * seq_len)
        start_idx = self.rng.randint(0, seq_len - reorder_length)
        tmp = x[start_idx : start_idx + reorder_length]
        self.rng.shuffle(tmp)
        new_x = x[:start_idx] + tmp + x[start_idx + reorder_length :]
        return self.pad(new_x), []

    def pad(self, x):
        return np.array(x + [0] * (self.max_len - len(x)))

    def __len__(self):
        return len(self.uid)
