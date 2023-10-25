import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")
from models import *


def remove_diag(x):
    n = x.size(0)
    return x.reshape(-1)[:-1].reshape(n - 1, n + 1)[:, 1:].reshape(n, -1)


class PretrainModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.base_model = eval(args.base_model)(args, device)
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

    def forward(self, item_seq_1, item_seq_2, mask_idx_1, mask_idx_2):
        """
        item_seq_1: (batch_size, seq_len)
        item_seq_2: (batch_size, seq_len)
        """
        bz = item_seq_1.size(0)

        item_embs_1 = self.base_model.item_embedding(item_seq_1)
        item_embs_2 = self.base_model.item_embedding(item_seq_2)
        item_embs_1[mask_idx_1] = self.mask_token
        item_embs_2[mask_idx_2] = self.mask_token

        total_item_seq = torch.cat(
            [item_seq_1, item_seq_1, item_seq_2, item_seq_2], dim=0
        )
        total_item_embs = torch.cat(
            [item_embs_1, item_embs_1, item_embs_2, item_embs_2], dim=0
        )
        _, total_ur, reg_loss = self.base_model(
            total_item_seq, item_embs=total_item_embs
        )

        total_ur = F.normalize(total_ur, p=2, dim=-1)

        anchor_1 = total_ur[:bz]
        pos_1 = total_ur[bz : 2 * bz]
        anchor_2 = total_ur[2 * bz : 3 * bz]
        pos_2 = total_ur[3 * bz :]

        # loss related with anchor_1
        sim_anchor_pos_1 = (anchor_1 * pos_1).sum(
            dim=-1, keepdim=True
        ) / self.args.num_pooling
        sim_anchor_semi_1_1 = (anchor_1 * anchor_2).sum(dim=-1, keepdim=True)
        sim_anchor_semi_1_2 = (anchor_1 * pos_2).sum(dim=-1, keepdim=True)
        sim_anchor_semi_1 = (
            torch.cat([sim_anchor_semi_1_1, sim_anchor_semi_1_2], dim=-1)
            / self.args.num_pooling
        )

        matrix_anchor_anchor_1 = remove_diag(torch.matmul(anchor_1, anchor_1.T))
        matrix_anchor_pos_1 = remove_diag(torch.matmul(anchor_1, pos_1.T))
        matrix_anchor_semi_1_1 = remove_diag(torch.matmul(anchor_1, anchor_2.T))
        matrix_anchor_semi_1_2 = remove_diag(torch.matmul(anchor_1, pos_2.T))
        sim_semi_neg_1 = (
            torch.cat(
                [
                    matrix_anchor_anchor_1,
                    matrix_anchor_pos_1,
                    matrix_anchor_semi_1_1,
                    matrix_anchor_semi_1_2,
                ],
                dim=-1,
            )
            / self.args.num_pooling
        )

        diff_pos_semi_1 = (
            sim_anchor_pos_1.min(dim=-1).values - sim_anchor_semi_1.max(dim=-1).values
        )
        hit_pos_semi_1 = (sim_anchor_pos_1 > sim_anchor_semi_1).sum()
        diff_semi_neg_1 = (
            sim_anchor_semi_1.min(dim=-1).values - sim_semi_neg_1.max(dim=-1).values
        )
        hit_semi_neg_1 = (
            sim_anchor_semi_1.unsqueeze(dim=-1) > sim_semi_neg_1.unsqueeze(dim=1)
        ).sum()
        coef_1 = 1 - sim_anchor_semi_1.detach().mean()
        diff_1 = coef_1 * diff_pos_semi_1 + (1 - coef_1) * diff_semi_neg_1
        loss_1 = -F.logsigmoid(diff_1).mean()

        # loss related with anchor_2
        sim_anchor_pos_2 = (anchor_2 * pos_2).sum(
            dim=-1, keepdim=True
        ) / self.args.num_pooling
        sim_anchor_semi_2_1 = (anchor_2 * anchor_1).sum(dim=-1, keepdim=True)
        sim_anchor_semi_2_2 = (anchor_2 * pos_1).sum(dim=-1, keepdim=True)
        sim_anchor_semi_2 = (
            torch.cat([sim_anchor_semi_2_1, sim_anchor_semi_2_2], dim=-1)
            / self.args.num_pooling
        )

        matrix_anchor_anchor_2 = remove_diag(torch.matmul(anchor_2, anchor_2.T))
        matrix_anchor_pos_2 = remove_diag(torch.matmul(anchor_2, pos_2.T))
        matrix_anchor_semi_2_1 = remove_diag(torch.matmul(anchor_2, anchor_1.T))
        matrix_anchor_semi_2_2 = remove_diag(torch.matmul(anchor_2, pos_1.T))
        sim_semi_neg_2 = (
            torch.cat(
                [
                    matrix_anchor_anchor_2,
                    matrix_anchor_pos_2,
                    matrix_anchor_semi_2_1,
                    matrix_anchor_semi_2_2,
                ],
                dim=-1,
            )
            / self.args.num_pooling
        )

        diff_pos_semi_2 = (
            sim_anchor_pos_2.min(dim=-1).values - sim_anchor_semi_2.max(dim=-1).values
        )
        hit_pos_semi_2 = (sim_anchor_pos_2 > sim_anchor_semi_2).sum()
        diff_semi_neg_2 = (
            sim_anchor_semi_2.min(dim=-1).values - sim_semi_neg_2.max(dim=-1).values
        )
        hit_semi_neg_2 = (
            sim_anchor_semi_2.unsqueeze(dim=-1) > sim_semi_neg_2.unsqueeze(dim=1)
        ).sum()
        coef_2 = 1 - sim_anchor_semi_2.detach().mean()
        diff_2 = coef_2 * diff_pos_semi_2 + (1 - coef_2) * diff_semi_neg_2
        loss_2 = -F.logsigmoid(diff_2).mean()

        hit_pos_semi = (hit_pos_semi_1 + hit_pos_semi_2) / 2
        hit_semi_neg = (hit_semi_neg_1 + hit_semi_neg_2) / 2

        batch_loss = (loss_1 + loss_2) / 2

        if reg_loss is not None:
            batch_loss = batch_loss + self.args.reg_coef * reg_loss

        return (
            batch_loss,
            hit_pos_semi,
            hit_semi_neg,
        )

    def get_ur(self, **kwargs):
        return self.base_model.get_ur(**kwargs)
