import torch
import torch.nn as nn
from models.layers import TransformerEncoder, AttnPooling


class BERT4Rec(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        self.item_embedding = nn.Embedding(args.n_items, args.emb_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(args.max_len, args.emb_dim)
        self.trm_encoder = TransformerEncoder(
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            emb_dim=args.emb_dim,
            inter_size=4 * args.emb_dim,
            hidden_dropout_prob=args.trm_dropout,
            attn_dropout_prob=args.trm_dropout,
        )
        self.layer_norm = nn.LayerNorm(args.emb_dim, eps=1e-12)
        self.dropout = nn.Dropout(args.trm_dropout)
        self.pooling = args.pooling
        if self.pooling == "attn":
            self.pooler = AttnPooling(args.emb_dim, args.num_pooling, args.cos_pooling)

    def forward(self, item_seq, item_embs=None):
        """
        item_seq: (batch_size, seq_len)
        item_embs: (batch_size, seq_len, emb_dim)
        return: (batch_size, seq_len, emb_dim), (batch_size, emb_dim)
        """
        batch_size, seq_len = item_seq.shape
        if item_embs is None:
            token_emb = self.item_embedding(item_seq)  # batch_size, seq_len, emb_dim
        else:
            token_emb = item_embs

        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        position_emb = self.pos_embedding(position_ids)
        position_emb = position_emb.unsqueeze(dim=0).expand(batch_size, -1, -1)
        input_emb = position_emb + token_emb  # batch_size, seq_len, emb_dim
        input_emb = self.dropout(self.layer_norm(input_emb))
        attn_mask, extended_mask = self.get_attn_mask(item_seq)
        hidden_states = self.trm_encoder(input_emb, extended_mask)
        if self.pooling:
            pooler_output, reg_loss = self.pooler(hidden_states, attn_mask)
        else:
            pooler_output, reg_loss = None, None
        return hidden_states, pooler_output, reg_loss

    def get_attn_mask(self, item_seq):
        """
        item_seq: (batch_size, seq_len)
        return: (batch_size, 1, seq_len, seq_len)
        """
        attn_mask = item_seq == 0  # True for mask, False for not mask
        attn_mask = attn_mask * torch.finfo(torch.float).min
        extended_attn_mask = attn_mask[:, None, None, :].expand(
            -1, -1, attn_mask.size(-1), -1
        )
        return attn_mask, extended_attn_mask

    def get_ur(self, **kwargs):
        _, user_repr, reg_loss = self.forward(item_seq=kwargs["item_seq"])
        return user_repr, reg_loss
