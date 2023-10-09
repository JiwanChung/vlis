import math

import torch
import torch.nn.functional as F


def sum_lprobs(lprobs):
    lprobs = torch.stack(lprobs, dim=0)
    return lprobs.logsumexp(dim=0)


def mean_lprobs(lprobs):
    x = sum_lprobs(lprobs)
    return x - math.log(len(lprobs))


def renormalize(logprob):
    return F.log_softmax(logprob, dim=-1)


def log_topk(tokenizer, val, k=5, do_exp: bool = False):
    if do_exp:
        val = val.exp()
    topk = val.topk(k)
    tokens = tokenizer.convert_ids_to_tokens(topk.indices)
    vs = topk.values.tolist()
    vs = [f'{v:.3f}' for v in vs]
    res = dict(zip(tokens, vs))
    return res
