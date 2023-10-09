import copy
import inspect
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import transformers
from munch import Munch
from einops import rearrange, repeat
import torch
from peft import PeftModel
from transformers import (
    PreTrainedModel, LlamaModel,
    AutoModelForCausalLM, AutoTokenizer,
    LlamaTokenizer, LlamaForCausalLM, GenerationConfig,
)
from transformers.utils import ModelOutput

from _lynx import load_lynx
from tokenizer import TokenConverter
from utils import log_topk


@dataclass
class VLMOutput(ModelOutput):
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_outputs: Optional[Any] = None


def load_lm(lm_name: str = 'TheBloke/vicuna-7B-1.1-HF'):
    model = AutoModelForCausalLM.from_pretrained(
        lm_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    # tokenizer = LlamaTokenizer.from_pretrained(lm_name)
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.unk_token = '<unk>'
    tokenizer.pad_token = '<unk>'
    return model, tokenizer


class Model(PreTrainedModel):
    def __init__(self, lm_name: str, vm_name: str,
                 fluency_threshold: float = 0.0001,
                 fluency_alpha: float = 0.8,
                 visual_threshold: float = 0.0,
                 run_baseline: Optional[str] = None,
                 verbose: bool = False):
        self.hparams = Munch(dict(
            fluency_threshold=fluency_threshold,
            fluency_alpha=fluency_alpha,
            run_baseline=run_baseline,
            visual_threshold=visual_threshold,
            verbose=verbose
        ))

        vm, self.processor = load_lynx(vm_name)
        lm, self.tokenizer = load_lm(lm_name)

        super().__init__(config=lm.config)

        self.lm = lm
        self.vm = vm

        self.converter = TokenConverter(self.tokenizer, self.processor.tokenizer)
        self.sanity_mask = self.build_sanity_mask()

        self.invalid_inits = [self.tokenizer.encode(v, add_special_tokens=False)[0]
                              for v in ['I']]

    def build_sanity_mask(self):
        storage = torch.full([self.converter.vocab_size], 0)
        oks = self.converter.sane_vocab_ids
        oks = [v for v in oks if v < storage.shape[0]]
        oks = torch.tensor(oks)
        storage = storage.scatter(0, oks, 1)
        return storage.bool()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_input_ids_lm=None,
        decoder_attention_mask_lm=None,
        decoder_inputs_embeds_vm=None,
        inputs_embeds=None,
        **kwargs
    ):
        if len(input_ids.shape) == 2:
            input_ids = repeat(input_ids, 'b l -> b n l',
                               n=inputs_embeds.shape[1])

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, :, -1:]
            decoder_input_ids_lm = decoder_input_ids_lm[:, -1:]

        res = {
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": attention_mask,
            "decoder_inputs_embeds_vm": decoder_inputs_embeds_vm,
            "decoder_input_ids_lm": decoder_input_ids_lm,
            "decoder_attention_mask_lm": decoder_attention_mask_lm,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
        return res

    def _reorder_cache(self, past, beam_idx):
        past_vm, past_lm, num_imgs = past
        past_vm = self._reorder_cache_vm(past_vm, beam_idx, num_imgs)
        past_lm = self._reorder_cache_base(past_lm, beam_idx)
        return (past_vm, past_lm, num_imgs)

    def _format_cache_vm(self, x, beam_idx, num_imgs):
        x = rearrange(x, '(b n) l r c -> b n l r c', n=num_imgs)
        x = x.index_select(0, beam_idx)
        x = rearrange(x, 'b n l r c -> (b n) l r c')
        return x

    def _reorder_cache_vm(self, past, beam_idx, num_imgs):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(self._format_cache_vm(past_state, beam_idx, num_imgs)
                                     for past_state in layer_past),)
        return reordered_past

    @staticmethod
    def _reorder_cache_base(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def preprocess_vm(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds, attention_mask = self.embed_image_lynx(input_ids, attention_mask, pixel_values)
        return inputs_embeds, attention_mask

    def split_past(self, past: Tuple[Tuple[Tuple[torch.Tensor]]]):
        if past is None:
            return None, None, None
        past_vm, past_lm, num_imgs = past
        return past_vm, past_lm, num_imgs

    def merge_past(self, past_vm, past_lm, num_imgs):
        if past_vm is None:
            return None
        past = (past_vm, past_lm, num_imgs)
        return past

    '''
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.vm.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        model_kwargs.pop('attention_mask')

        return model_kwargs
    '''

    def get_step(self, past):
        if past is None:
            return 0
        else:
            return past[0][0].shape[-2]

    def embed_image_lynx(
        self,
        input_ids: torch.LongTensor = None,
        input_atts: torch.Tensor = None,
        images: Optional[torch.FloatTensor] = None,
    ):
        vision_input = images
        text_embeds = self.vm.embed_tokens(input_ids)

        if vision_input is not None:
            vision_embeds, vision_atts = self.vm.get_vision_embeds(vision_input)
            v2t_feats, v2t_atts = self.vm.bridge(vision_embeds=vision_embeds, vision_atts=vision_atts)

            inputs_embeds = torch.cat([v2t_feats, text_embeds], dim=1)
            attention_mask = torch.cat([v2t_atts, input_atts.to(v2t_atts.dtype)], dim=1)
        else:
            inputs_embeds = text_embeds
            attention_mask = input_atts

        return inputs_embeds, attention_mask

    def forward(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds_vm: Optional[torch.LongTensor] = None,
        decoder_input_ids_lm: Optional[torch.LongTensor] = None,
        decoder_attention_mask_lm: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ):
        return self._forward(
            decoder_input_ids,
            decoder_attention_mask,
            decoder_inputs_embeds_vm,
            decoder_input_ids_lm,
            decoder_attention_mask_lm,
            past_key_values,
            use_cache,
        )

    def _forward(
        self,
        decoder_input_ids_vm: Optional[torch.LongTensor] = None,
        decoder_attention_mask_vm: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds_vm: Optional[torch.LongTensor] = None,
        input_ids_lm: Optional[torch.LongTensor] = None,
        attention_mask_lm: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ):
        past_vm, past_lm, _ = self.split_past(past_key_values)

        vmask = decoder_attention_mask_vm
        if vmask is not None:
            vmask = rearrange(vmask, 'b n l -> (b n) l')

        if past_vm is None:
            vides = decoder_inputs_embeds_vm
            num_imgs = vides.shape[1]
            vides = rearrange(vides, 'b n l c -> (b n) l c')
            vm_inputs = dict(inputs_embeds=vides)
        else:
            vids = decoder_input_ids_vm
            num_imgs = vids.shape[1]
            vids = rearrange(vids, 'b n l -> (b n) l')
            vm_inputs = dict(input_ids=vids)

        vm_output = self.vm.LLM.forward(
            **vm_inputs,
            attention_mask=vmask,
            past_key_values=past_vm,  # store image info in past key values
            use_cache=use_cache,
            return_dict=True
        )
        # vm_output.logits = self.vm.LLM.lm_head(vm_output[0])
        vm_output.logits = vm_output[0]

        if past_lm is not None:  # not the first step
            input_ids_lm = decoder_input_ids_vm[:, 0]

        lm_output = self.lm(
            input_ids=input_ids_lm,
            attention_mask=attention_mask_lm,
            past_key_values=past_lm,
            use_cache=use_cache,
            return_dict=True
        )
        vm_logits = rearrange(vm_output.logits,
                              '(b n) l v -> b n l v', n=num_imgs)
        lm_logits = lm_output.logits

        step = self.get_step(past_vm)
        logits = self.calc_score(lm_logits, vm_logits[:, 0],
                                 vm_logits[:, 1:], step=step, verbose=self.hparams.verbose)
        logits = logits[:, None, :]  # fake length dim

        if self.hparams.verbose:
            print('fn', log_topk(self.processor.tokenizer, logits[0, 0]))

        past_vm = vm_output.past_key_values
        past_lm = lm_output.past_key_values
        past_key_values = self.merge_past(past_vm, past_lm, num_imgs)

        return VLMOutput(
            logits=logits,
            past_key_values=past_key_values,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values_good: torch.FloatTensor,
        pixel_values_bad: List[torch.FloatTensor],
        input_ids_vm: Optional[torch.LongTensor] = None,
        attention_mask_vm: Optional[torch.LongTensor] = None,
        input_ids_lm: Optional[torch.LongTensor] = None,
        attention_mask_lm: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids_vm (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask_vm (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        pixel_values = torch.stack([pixel_values_good, *pixel_values_bad], dim=1)
        num_imgs = pixel_values.shape[1]
        pixel_values = rearrange(pixel_values, 'b n c h w -> (b n) c h w')
        _input_ids_vm = repeat(input_ids_vm, 'b l -> (b n) l', n=num_imgs)
        _attention_mask_vm = repeat(attention_mask_vm, 'b l -> (b n) l', n=num_imgs)
        inputs_embeds_vm, attention_mask_vm = self.preprocess_vm(
            pixel_values,
            _input_ids_vm,
            _attention_mask_vm
        )
        inputs_embeds_vm = rearrange(inputs_embeds_vm, '(b n) l c -> b n l c', n=num_imgs)
        attention_mask_vm = rearrange(attention_mask_vm, '(b n) l -> b n l', n=num_imgs)

        outputs = PreTrainedModel.generate(
            self=self,
            inputs_embeds=inputs_embeds_vm,
            attention_mask=attention_mask_vm,
            decoder_input_ids_lm=input_ids_lm,
            decoder_attention_mask_lm=attention_mask_lm,
            decoder_inputs_embeds_vm=inputs_embeds_vm,
            decoder_start_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            **generate_kwargs,
        )

        return outputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            inc_mask = attention_mask.new_ones([*attention_mask.shape[:-1], 1])
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, inc_mask], dim=-1
            )
        # update decoder attention mask
        if "decoder_attention_mask_lm" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask_lm"]
            model_kwargs["decoder_attention_mask_lm"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

        return model_kwargs
