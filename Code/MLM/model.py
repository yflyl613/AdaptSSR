import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")
from models import *


class MLMHead(nn.Module):
    def __init__(self, args, decoder_weight):
        super().__init__()
        self.decoder_weight = decoder_weight
        self.bias = nn.Parameter(torch.zeros(args.n_items))

    def forward(self, hidden_states):
        """
        hidden_states: bz, seq_len, emb_dim
        return: bz, seq_len, n_items
        """
        predictions = F.linear(
            hidden_states, weight=self.decoder_weight, bias=self.bias
        )
        return predictions


class PretrainModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.base_model = eval(args.base_model)(args, device)
        self.mlm_head = MLMHead(args, self.base_model.item_embedding.weight)
        self.mask_token = nn.Parameter(
            torch.empty((1, args.emb_dim), dtype=torch.float).normal_(mean=0, std=0.02)
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

    def forward(self, item_seq, mask_index, compute_loss=True):
        """
        item_seq: (bs, seq_len)
        mask_index: [list of mask_index_i, list of mask_index_j]
        """
        item_embs = self.base_model.item_embedding(item_seq)
        item_embs[mask_index] = self.mask_token
        hidden_states, _, _ = self.base_model(item_seq, item_embs=item_embs)
        mask_hidden_states = hidden_states[mask_index]  # num_mask, emb_dim
        mask_logits = self.mlm_head(mask_hidden_states)  # num_mask, n_items
        mask_label = item_seq[mask_index]  # num_mask
        mask_preds = torch.argmax(mask_logits, dim=-1)
        mlm_hit = (mask_preds == mask_label).sum()
        if compute_loss:
            mlm_loss = F.cross_entropy(mask_logits, mask_label)
            return mlm_loss, mlm_hit
        else:
            return mlm_hit

    def get_ur(self, **kwargs):
        return self.base_model.get_ur(**kwargs)
