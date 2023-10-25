import argparse
import torch
import logging


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def dump_args(args):
    for arg in dir(args):
        if not arg.startswith("_"):
            logging.info(f"args[{arg}]={getattr(args, arg)}")


@torch.no_grad()
def hit_cnt(y_pred, y_true=None, mask=None):
    """
    if y_true is None:
        y_pred: (batch_size, seq_len, n_candidate)
        mask: (batch_size, seq_len)
    else:
        y_pred: (batch_size, n_labels) or (batch_size,)
        y_true: (batch_size,)
    """
    if y_true is None:
        return ((y_pred.argmax(dim=-1) == 0) * mask).sum()
    else:
        if len(y_pred.shape) == 1:
            return (y_true == y_pred).sum()
        else:
            return (y_true == y_pred.argmax(dim=-1)).sum()


@torch.no_grad()
def topN_metrics(sorted_pred, labels):
    """
    sorted_pred: (batch_size, k)
    labels: (batch_size,)
    """
    k = sorted_pred.shape[-1]
    hit_matrix = sorted_pred == labels.unsqueeze(dim=-1)  # batch_size, k
    hr = hit_matrix.any(dim=-1).sum()
    mrr_matrix = hit_matrix / (torch.arange(k, device=labels.device) + 1.0)
    ndcg_matrix = hit_matrix / torch.log2(torch.arange(k, device=labels.device) + 2.0)
    mrr = mrr_matrix.sum()
    ndcg = ndcg_matrix.sum()
    return hr, mrr, ndcg
