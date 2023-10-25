import math
import pickle
import random
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, total_behavior, uid_path, args):
        super().__init__()
        self.rng = random.Random(2101)
        self.total_behavior = total_behavior
        self.max_len = args.max_len
        self.mask_ratio = args.mask_ratio

        with open(uid_path, "rb") as f:
            self.uid = pickle.load(f)

    def __getitem__(self, idx):
        uid = self.uid[idx]
        behavior = self.total_behavior[uid]
        seq_len = (behavior != 0).sum()
        mask_num = math.ceil(seq_len * self.mask_ratio)
        if mask_num == seq_len:
            mask_num = math.floor(seq_len * self.mask_ratio)
        mask_idx = self.rng.sample(range(seq_len), mask_num)
        return behavior, mask_idx

    def __len__(self):
        return len(self.uid)
