import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *


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

    def forward(self, item_seq, labels=None, negs=None, mask=None):
        """
        item_seq: (batch_size, seq_len)
        labels: (batch_size,)
        negs: (batch_size,)
        """
        user_repr, reg_loss = self.base_model.get_ur(item_seq=item_seq)
        logits = self.classifier(user_repr)
        if labels is None:
            logits[mask] = torch.finfo(torch.float).min  # mask positive labels
            return logits
        else:
            labels_score = torch.gather(
                logits, dim=1, index=labels.unsqueeze(dim=-1)
            ).reshape(-1)
            negs_score = torch.gather(
                logits, dim=1, index=negs.unsqueeze(dim=-1)
            ).reshape(-1)
            batch_loss = -F.logsigmoid(labels_score - negs_score).mean()
            batch_hit = (labels_score > negs_score).sum()

            if reg_loss is not None:
                batch_loss = batch_loss + self.args.reg_coef * reg_loss

            return batch_loss, batch_hit
