import importlib
import torch.nn as nn
import torch.nn.functional as F

from models import *
from utils import hit_cnt


class FinetuneModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device

        if args.pretrain_method is not None:
            module = importlib.import_module(f"{args.pretrain_method}.model")
            self.base_model = module.PretrainModel(args, device)
        else:
            self.base_model = eval(args.base_model)(args, device)

        self.classifier = nn.Sequential(
            nn.Linear(args.emb_dim * args.num_pooling, args.emb_dim),
            nn.ReLU(),
            nn.Linear(args.emb_dim, args.n_labels),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, labels):
        user_repr, reg_loss = self.base_model.get_ur(item_seq=item_seq)
        logits = self.classifier(user_repr)
        batch_loss = F.cross_entropy(logits, labels)
        batch_hit = hit_cnt(logits, labels)

        if reg_loss is not None:
            batch_loss = batch_loss + self.args.reg_coef * reg_loss

        return batch_loss, batch_hit
