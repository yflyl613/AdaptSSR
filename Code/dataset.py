import random
from torch.utils.data import Dataset


class FinetuneDataset(Dataset):
    def __init__(self, total_behavior, uid, labels):
        super().__init__()
        self.total_behavior = total_behavior
        self.uid = uid
        self.labels = labels

    def __getitem__(self, idx):
        uid = self.uid[idx]
        label = self.labels[idx]
        return (self.total_behavior[uid], label)

    def __len__(self):
        return len(self.labels)


class FinetuneTopNDatasetTrain(Dataset):
    def __init__(self, total_behavior, uid, target, labels, n_labels):
        super().__init__()
        self.total_behavior = total_behavior
        self.uid = uid
        self.target = target
        self.labels = labels
        self.total_labels = list(range(n_labels))
        self.rng = random.Random(1534)

    def __getitem__(self, idx):
        uid = self.uid[idx]
        behavior = self.total_behavior[uid]
        target = self.target[idx]
        labels = self.labels[idx]
        neg = self.rng.choice(self.total_labels)
        while neg in labels:
            neg = self.rng.choice(self.total_labels)
        return (behavior, target, neg)

    def __len__(self):
        return len(self.uid)


class FinetuneTopNDatasetVal(Dataset):
    def __init__(self, total_behavior, uid, target, labels):
        super().__init__()
        self.total_behavior = total_behavior
        self.uid = uid
        self.target = target
        self.labels = labels

    def __getitem__(self, idx):
        uid = self.uid[idx]
        behavior = self.total_behavior[uid]
        target = self.target[idx]
        labels = self.labels[idx].copy()
        labels.remove(target)
        return (behavior, target, labels)

    def __len__(self):
        return len(self.uid)
