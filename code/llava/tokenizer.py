import re

import torch


class TokenConverter:
    def __init__(self, tl, tv):
        self.tl = tl
        self.tv = tv

        vv = self.tv.get_vocab()
        vl = self.tl.get_vocab()
        self.vv = vv
        self.vl = vl
        self.rev_vv = {v: k for k, v in vv.items()}
        self.rev_vl = {v: k for k, v in vl.items()}
        self.shared = set(vv.keys()) & set(vl.keys())

        self.vocab_size = len(self.tv)
        self.lm_vocab_size = len(self.tl)
        self.build_sane_vocab('v')

        self.v_to_l = self.get_map(self.vv, self.tl, self.tv)
        self.l_to_v = self.get_map(self.vl, self.tv, self.tl)

    def get_map(self, vs, tk, tk2):
        out = {}
        for token, idx in vs.items():
            token2 = tk2.decode(idx)
            token3 = token.strip()
            for _token in [token, token2, token3]:
                x = tk.encode(_token, add_special_tokens=False)
                if len(x) == 1:
                    x = x[0]
                    out[idx] = x
        return out

    def build_sane_vocab(self, model: str = 'l'):
        tk = self.tv if model == 'v' else self.tl
        voc = self.vv if model == 'v' else self.vl
        self.sane_re = re.compile('[a-zA-Z0-9\.\)\(\]\[]+')
        self._oks = [' ', '.', ',', '!', ':', ';', '-', '_', '/', ' (', ')']  #, '\n']
        self.oks = self.cycle_tokens(tk, self._oks)
        self.init_t = self.get_subword_init(tk)
        self.oks = [*self.oks, self.init_t]
        self.sane_vocab = self.filter_vocab(voc, self.oks, flag=False)
        self.sane_vocab_ids = tk.convert_tokens_to_ids(self.sane_vocab)

    @staticmethod
    def get_subword_init(model, txt='had that'):
        vs = model.convert_ids_to_tokens(model.encode(txt, add_special_tokens=False))
        idx = len(txt.split()) - 1
        return vs[idx][0]

    def cycle_tokens(self, model, vs):
        vs = [self.cycle_token(model, v) for v in vs]
        vs = [v for v in vs if v is not None]
        return vs

    @staticmethod
    def cycle_token(model, token):
        res = model.convert_ids_to_tokens(
            model.encode(token, add_special_tokens=False)
        )
        return res[-1] if len(res) > 1 else None

    def filter_vocab(self, vs, oks=[], do_filter: bool = True, flag: bool = True):
        return {k: v for k, v in vs.items() if self._filter_vocab(k, oks, do_filter, flag)}

    def _filter_vocab(self, k, oks=[], do_filter: bool = True, flag: bool = True):
        res = True
        if do_filter:
            res = self.sane_re.search(k) is not None
        if flag:
            return res or (k in oks)
        else:
            return res and k not in oks

    def ids_vm_to_lm(self, vm):
        # vm shape? B l=1
        out = torch.full_like(vm, self.tl.pad_token_id)
        for b, row in enumerate(vm):
            for length, token in enumerate(row):
                vocab = token.item()
                '''
                if vocab in self.v_to_l:
                    out[b, length] = self.v_to_l[vocab]
                '''
                if self.rev_vv[vocab] in self.shared:
                    _v = self.vl[self.rev_vv[vocab]]
                    out[b, length] = _v
        return out

    def logits_lm_to_vm(self, lm, vocab_size, do_l_to_v: bool = False):
        # lm shape? B l=1 V
        out = torch.full([*lm.shape[:-1], vocab_size], self.tv.pad_token_id)
        out = out.to(device=lm.device, dtype=lm.dtype)
        for b, row in enumerate(lm):
            for length, token in enumerate(row):
                for vocab, val in enumerate(token):
                    if not do_l_to_v:
                        if self.rev_vl[vocab] in self.shared:
                            _v = self.vv[self.rev_vl[vocab]]
                            out[b, length, _v] = val.item()
                    else:
                        if vocab in self.l_to_v:
                            _vocab = self.l_to_v[vocab]
                            out[b, length, _vocab] = val.item()
        return out
