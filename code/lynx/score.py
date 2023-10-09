import math
from pprint import pprint

import torch
import torch.nn.functional as F
from transformers import (
    LlamaTokenizer, Blip2Processor
)

from tokenizer import TokenConverter
from utils import (
    mean_lprobs, renormalize, log_topk,
    guarded_mean, guarded_std, get_ent
)


class Scorer:
    def calc_score(self, lm, vm_good, vm_bads, step: int = 0, verbose: bool = False):
        _lm = lm[:, -1:, :]
        # do nothing and share llama tokenizer
        # _lm = self.converter.logits_lm_to_vm(_lm, vm_good.shape[-1], do_l_to_v=step == 0)
        _vm_bads = list(vm_bads.transpose(0, 1))
        _lm = _lm[:, -1]
        _vm_good = vm_good[:, -1]
        _vm_bads = [v[:, -1] for v in _vm_bads]

        if verbose:
            print(f'step: {step}')
        if step == 0:
            for inv_init in self.invalid_inits:
                _lm[..., inv_init] = -math.inf

        vocab_size = _lm.shape[-1]
        _lm = renormalize(_lm)
        _vm_good = renormalize(_vm_good[:, :vocab_size])
        _vm_bads = [renormalize(v[:, :vocab_size]) for v in _vm_bads]

        if self.hparams.run_baseline == 'lm':
            return _lm
        elif self.hparams.run_baseline == 'vm':
            return _vm_good
        elif self.hparams.run_baseline == 'ours_vm':
            _vm_bad = mean_lprobs(_vm_bads)
            _lm = _vm_bad
        elif self.hparams.run_baseline == 'ours_lm':
            _vm_bads = [_lm for _ in _vm_bads]

        return self._calc_score(_lm, _vm_good, _vm_bads, step, verbose=verbose)

    def adapt_sanity_mask(self, fluency_mask):
        sanity_mask = self.sanity_mask.clone().to(fluency_mask.device)
        sanity_mask = sanity_mask.to(fluency_mask.dtype)
        sanity_mask = sanity_mask[None, :].repeat(fluency_mask.shape[0], 1)
        if sanity_mask.shape[-1] != fluency_mask.shape[-1]:
            sanity_mask = sanity_mask[:, :fluency_mask.shape[-1]]
            if sanity_mask.shape[-1] < fluency_mask.shape[-1]:
                s2 = fluency_mask.clone()
                s2[:, :sanity_mask.shape[-1]] = sanity_mask
                sanity_mask = s2
        return sanity_mask

    def _calc_score(self, lm, vm_good, vm_bads, step: int = 0,
                    verbose: bool = False):
        vm_bad = mean_lprobs(vm_bads)

        vo = vm_good
        vx = vm_bad

        if self.hparams.fluency_threshold > 0:
            fluency_mask = math.log(self.hparams.fluency_threshold) <= lm
            mx = fluency_mask
            vo = torch.where(mx, vo, -math.inf)
            vx = torch.where(mx, vx, math.inf)  # this should be subtracted, hence positive inf

        rr = vo - vx

        if self.hparams.fluency_threshold > 0:
            vx = torch.where(mx, vx, -math.inf)  # revert for logging

        if self.hparams.visual_threshold > 0:
            visual_mask = math.log(self.hparams.visual_threshold) <= vo
            mx2 = visual_mask
            lm = torch.where(mx2, lm, -math.inf)

        fn = lm * self.hparams.fluency_alpha + rr
        fn[torch.isnan(fn)] = -math.inf
        if self.hparams.fluency_threshold > 0:
            fn = torch.where(mx, fn, -math.inf)

        # for logging
        if self.hparams.fluency_threshold > 0:
            rr = torch.where(mx, rr, -math.inf)
        if self.hparams.visual_threshold > 0:
            rr = torch.where(mx2, rr, -math.inf)

        if verbose:
            print('lm', log_topk(self.processor.tokenizer, lm[0].exp()), get_ent(lm[0]))
            print('vo', log_topk(self.processor.tokenizer, vo[0].exp()), get_ent(vo[0]))
            print('vx', log_topk(self.processor.tokenizer, vx[0].exp()), get_ent(vx[0]))
            print('rr', log_topk(self.processor.tokenizer, rr[0]))
            print('fn', log_topk(self.processor.tokenizer, fn[0]))
            topk = fn[0].topk(5).indices

            print('top_lm', lm[0].gather(0, topk).exp().tolist())
            print('top_vo', vo[0].gather(0, topk).exp().tolist())
            print('top_vx', vx[0].gather(0, topk).exp().tolist())
            print('top_rr', rr[0].gather(0, topk).tolist())
            print('rr_mean', guarded_mean(rr[0]).item(), guarded_mean(rr[0].gather(0, topk)).item())
            print('rr_std', guarded_std(rr[0]).item(), guarded_std(rr[0].gather(0, topk)).item())
            print('---')

        return fn

    '''
    def _calc_score(self, lm, vm_good, vm_bads, step: int = 0,
                    verbose: bool = False):
        vm_bad = mean_lprobs(vm_bads)
        fluency_mask = math.log(self.hparams.fluency_threshold) <= lm

        sanity_mask = self.adapt_sanity_mask(fluency_mask)
        mx = fluency_mask  # & sanity_mask

        vo = torch.where(mx, vm_good, -math.inf)
        vx = torch.where(mx, vm_bad, math.inf)  # this should be subtracted, hence positive inf
        rr = vo - vx
        rr = torch.where(mx, rr, -math.inf)
        fn = lm * self.hparams.fluency_alpha + rr
        fn = fn / (self.hparams.fluency_alpha + 1)
        fn[torch.isnan(fn)] = -math.inf
        fn = torch.where(mx, fn, -math.inf)

        if verbose:
            print('lm', log_topk(self.processor.tokenizer, lm[0]))
            print('_vo', log_topk(self.processor.tokenizer, vm_good[0]))
            print('_vx', log_topk(self.processor.tokenizer, vm_bad[0]))
            print('vo', log_topk(self.processor.tokenizer, vo[0]))
            vx2 = torch.where(mx, vm_bad, -math.inf)  # this should be subtracted, hence positive inf
            print('vx', log_topk(self.processor.tokenizer, vx2[0]))
            print('rr', log_topk(self.processor.tokenizer, rr[0]))
            print('fn', log_topk(self.processor.tokenizer, fn[0]))

        return fn
    '''


class TestScorer(Scorer):
    def __init__(self, lm_name: str, vm_name: str):
        self.processor = Blip2Processor.from_pretrained(vm_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(lm_name)
        self.tokenizer.bos_token = '<s>'
        self.tokenizer.eos_token = '</s>'
        self.tokenizer.unk_token = '<unk>'
        self.tokenizer.pad_token = '<unk>'

        self.converter = TokenConverter(self.tokenizer, self.processor.tokenizer)

    def test(self):
        xs = [
            'my name is Inigo Montoya. You killed my father. Prepare to die. </s> !!."".1!!!',
            ' the a my see an',
            "'' \' :\"",
        ]

        for x in xs:
            self._test(x)

    def _test(self, x):
        tl = self.tokenizer
        tv = self.processor.tokenizer

        x1 = tv(x, return_tensors='pt')['input_ids']
        x2 = tl(x, return_tensors='pt')['input_ids']
        x3 = self.converter.ids_vm_to_lm(x1)

        x1t = tv.batch_decode(x1)
        x2t = tl.batch_decode(x2)
        x3t = tl.batch_decode(x3)

        res_ids = {
            'lm': x1t,
            'vm': x2t,
            'vm->lm': x3t,
        }

        pprint(res_ids)

        y1 = F.one_hot(x2, len(tl)).float()
        y2 = F.one_hot(x1, len(tv))
        y3 = self.converter.logits_lm_to_vm(y1, y2.shape[-1])
        y1t = tl.batch_decode(y1.argmax(-1))
        y2t = tv.batch_decode(y2.argmax(-1))
        y3t = tv.batch_decode(y3.argmax(-1))

        res_logits = {
            'lm': y1t,
            'vm': y2t,
            'lm->vm': y3t,
        }

        pprint(res_logits)


if __name__ == '__main__':
    scorer = TestScorer("decapoda-research/llama-7b-hf", 'Salesforce/blip2-flan-t5-xl')
    scorer.test()
    import ipdb; ipdb.set_trace()  # XXX DEBUG
