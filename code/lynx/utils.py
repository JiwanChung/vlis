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


def guarded_mean(x, eps: float = 1e-4):
    m = ~torch.isnan(x)
    m = (m & (x > -math.inf)).to(x.dtype)
    z = m * x
    z[torch.isnan(z)] = 0
    mean = z.sum() / (m.sum() + eps)
    return mean


def guarded_std(x, eps: float = 1e-4):
    m = ~torch.isnan(x)
    m = (m & (x > -math.inf)).to(x.dtype)
    z = m * x
    z[torch.isnan(z)] = 0
    mean = z.sum() / (m.sum() + eps)
    z = m * x ** 2
    z[torch.isnan(z)] = 0
    vs = z.sum() / (m.sum() + eps)
    var = (vs - mean ** 2)
    return var.sqrt()


def get_ent(lprobs, raw: bool = False, eps: float = 1e-4):
    m = ~torch.isnan(lprobs)
    m = (m & (lprobs > -math.inf)).to(lprobs.dtype)
    z = - m * lprobs * lprobs.exp()
    z[torch.isnan(z)] = 0
    ent = z.sum() / (m.sum() + eps)

    if raw:
        return ent
    else:
        return f'{ent.item():.2f}'
