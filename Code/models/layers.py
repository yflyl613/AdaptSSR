import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def remove_diag(x):
    n = x.size(0)
    return x.reshape(-1)[:-1].reshape(n - 1, n + 1)[:, 1:].reshape(n, -1)


class AttnPooling(nn.Module):
    def __init__(self, emb_dim, num_pooling, cos_pooling):
        super().__init__()
        self.query = nn.Parameter(
            torch.empty((num_pooling, emb_dim), dtype=torch.float).normal_(
                mean=0, std=0.02
            )
        )
        self.cos_pooling = cos_pooling

    def forward(self, input_tensor, mask=None):
        """
        input_tensor: batch_size, seq_len, emb_dim
        mask: batch_size, seq_len   # float.min for mask, 0 for not mask
        """
        if self.cos_pooling:
            norm_input = F.normalize(input_tensor, p=2, dim=-1)
            norm_query = F.normalize(self.query, p=2, dim=-1)
            attn_weights = torch.matmul(
                norm_input.unsqueeze(dim=1), norm_query.unsqueeze(dim=-1)
            ).squeeze(dim=-1)
        else:
            attn_weights = torch.matmul(
                input_tensor.unsqueeze(dim=1), self.query.unsqueeze(dim=-1)
            ).squeeze(dim=-1)
        attn_weights = F.softmax(
            attn_weights + mask.unsqueeze(dim=1), dim=-1
        )  # bz, K, seq
        output = torch.matmul(attn_weights, input_tensor)
        norm_query = F.normalize(self.query, p=2, dim=-1)
        corr = remove_diag(torch.matmul(norm_query, norm_query.T))
        reg_loss = corr.pow_(2).sum().sqrt()
        return output.reshape(output.size(0), -1), reg_loss


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        n_heads,
        hidden_dropout_prob=0.1,
        attn_dropout_prob=0.1,
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        if emb_dim % n_heads != 0:
            raise ValueError(
                f"emb_dim {emb_dim} must be divisible by n_heads {n_heads}"
            )

        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.emb_dim = emb_dim
        self.scale = math.sqrt(self.head_dim)

        self.WQ = nn.Linear(emb_dim, emb_dim)
        self.WK = nn.Linear(emb_dim, emb_dim)
        self.WV = nn.Linear(emb_dim, emb_dim)
        self.WO = nn.Linear(emb_dim, emb_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.layer_norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_score(self, x):
        """
        x: batch_size, seq_len, emb_dim
        return: batch_size, n_heads, seq_len, head_dim
        """
        new_x_shape = x.size()[:-1] + (self.n_heads, self.head_dim)
        x = x.view(new_x_shape)  # batch_size, seq_len, n_heads, head_dim
        return x.permute(0, 2, 1, 3)  # batch_size, n_heads, seq_len, head_dim

    def forward(self, input_tensor, mask=None):
        """
        input_tensor: batch_size, seq_len, emb_dim
        mask: batch_size, 1, seq_len or 1, seq_len   # 0 for mask, 1 for not mask
        return: batch_size, seq_len, emb_dim
        """
        input_query = self.WQ(input_tensor)
        input_key = self.WK(input_tensor)
        input_value = self.WV(input_tensor)

        query = self.transpose_for_score(input_query)
        key = self.transpose_for_score(input_key)
        value = self.transpose_for_score(input_value)

        attn_scores = (
            torch.matmul(query, key.transpose(-1, -2)) / self.scale
        )  # batch_size, n_heads, seq_len, prompt_length+seq_len
        attn_scores = attn_scores + mask
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(
            attn_probs, value
        )  # batch_size, n_heads, seq_len, head_dim
        context = context.permute(
            0, 2, 1, 3
        ).contiguous()  # batch_size, seq_len, n_heads, head_dim
        new_context_shape = context.size()[:-2] + (self.emb_dim,)
        context = context.view(new_context_shape)  # batch_size, seq_len, emb_dim
        output = self.WO(context)
        output = self.layer_norm(input_tensor + self.out_dropout(output))
        return output


class FFN(nn.Module):
    def __init__(
        self,
        emb_dim,
        inter_size,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        hidden_act="gelu",
    ):
        super().__init__()
        ACT2FN = {
            "relu": F.relu,
            "gelu": gelu,
            "tanh": F.tanh,
            "sigmoid": F.sigmoid,
        }
        self.dense_1 = nn.Linear(emb_dim, inter_size)
        self.inter_act = ACT2FN[hidden_act]
        self.dense_2 = nn.Linear(inter_size, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor):
        inter_hidden = self.inter_act(self.dense_1(input_tensor))
        hidden_states = self.dense_2(inter_hidden)
        output = self.layer_norm(input_tensor + self.dropout(hidden_states))
        return output


class TransformerLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        emb_dim,
        inter_size,
        hidden_dropout_prob=0.1,
        attn_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        hidden_act="gelu",
    ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            emb_dim=emb_dim,
            n_heads=n_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.ffn = FFN(
            emb_dim=emb_dim,
            inter_size=inter_size,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
        )

    def forward(self, input_tensor, mask=None):
        attn_output = self.self_attn(input_tensor, mask)
        ffn_output = self.ffn(attn_output)
        return ffn_output


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        emb_dim=64,
        inter_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        layer_norm_eps=1e-12,
        hidden_act="gelu",
    ):
        super().__init__()
        layer = TransformerLayer(
            n_heads=n_heads,
            emb_dim=emb_dim,
            inter_size=inter_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, input_tensor, mask=None):
        output = input_tensor
        for layer in self.layers:
            output = layer(output, mask)
        return output
