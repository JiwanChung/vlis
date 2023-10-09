import copy
import inspect
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import transformers
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

from munch import Munch
from einops import rearrange, repeat
import torch
from peft import PeftModel
from transformers import (
    PreTrainedModel,
    LlamaTokenizer, LlamaForCausalLM, GenerationConfig,
    Blip2Processor, Blip2ForConditionalGeneration,
)
from transformers.utils import ModelOutput

from tokenizer import TokenConverter


@dataclass
class VLMOutput(ModelOutput):
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_outputs: Optional[Any] = None


def load_lm(lm_name: str = 'TheBloke/vicuna-7B-1.1-HF'):
    model = LlamaForCausalLM.from_pretrained(
        lm_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(lm_name)
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.unk_token = '<unk>'
    tokenizer.pad_token = '<unk>'
    return model, tokenizer


def load_vm(vm_name: str = 'Salesforce/blip2-flan-t5-xl'):
    model = Blip2ForConditionalGeneration.from_pretrained(
        vm_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    processor = Blip2Processor.from_pretrained(vm_name)
    return model, processor


class Model(PreTrainedModel):
    def __init__(self, lm_name: str, vm_name: str,
                 fluency_threshold: float = 0.0001,
                 fluency_alpha: float = 0.8,
                 verbose: bool = False):
        self.hparams = Munch(dict(
            fluency_threshold=fluency_threshold,
            fluency_alpha=fluency_alpha,
            verbose=verbose
        ))

        lm, self.tokenizer = load_lm(lm_name)
        vm, self.processor = load_vm(vm_name)

        super().__init__(config=vm.config)

        self.lm = lm
        self.vm = vm

        self.converter = TokenConverter(self.tokenizer, self.processor.tokenizer)
        self.sanity_mask = self.build_sanity_mask()

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
        inputs_embeds=None,
        **kwargs
    ):
        if encoder_outputs is not None:
            h = encoder_outputs.last_hidden_state
            if len(input_ids.shape) < len(h.shape) - 1:
                input_ids = repeat(input_ids, 'b l -> b n l',
                                   n=h.shape[1])
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, :, -1:]
            decoder_input_ids_lm = decoder_input_ids_lm[:, -1:]

        res = {
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": attention_mask,
            "decoder_input_ids_lm": decoder_input_ids_lm,
            "decoder_attention_mask_lm": decoder_attention_mask_lm,
            "decoder_inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
        return res

    def _reorder_cache(self, past, beam_idx):
        past_vm, past_lm = past
        past_vm = self.vm._reorder_cache(past_vm, beam_idx)
        past_lm = self.lm._reorder_cache(past_lm, beam_idx)
        return (past_vm, past_lm)

    def preprocess_vm(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        vm = self.vm
        if hasattr(vm, "hf_device_map"):
            # preprocess for `accelerate`
            vm._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = vm.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = vm.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = vm.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = vm.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = vm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        return inputs_embeds, attention_mask

    def split_past(self, past: Tuple[Tuple[Tuple[torch.Tensor]]]):
        if past is None:
            return None, None
        past_vm, past_lm = past
        return past_vm, past_lm

    def merge_past(self, past_vm, past_lm):
        if past_vm is None:
            return None
        past = (past_vm, past_lm)
        return past

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

        num_imgs = encoder_kwargs['inputs_embeds'].shape[1]
        encoder_kwargs['inputs_embeds'] = rearrange(
            encoder_kwargs['inputs_embeds'], 'b n l c -> (b n) l c'
        )
        encoder_kwargs['attention_mask'] = rearrange(
            encoder_kwargs['attention_mask'], 'b n l -> (b n) l'
        )
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["encoder_outputs"].last_hidden_state = rearrange(
            model_kwargs["encoder_outputs"].last_hidden_state,
            '(b n) l c -> b n l c', n=num_imgs
        )
        model_kwargs.pop('inputs_embeds')
        model_kwargs.pop('attention_mask')

        return model_kwargs

    def get_step(self, past):
        if past is None:
            return 0
        else:
            return past[0][0].shape[-2]

    def forward(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids_lm: Optional[torch.LongTensor] = None,
        decoder_attention_mask_lm: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ):
        return self._forward(
            decoder_input_ids,
            decoder_attention_mask,
            decoder_input_ids_lm,
            decoder_attention_mask_lm,
            encoder_outputs,
            past_key_values,
            use_cache,
            decoder_inputs_embeds_vm=decoder_inputs_embeds
        )

    def _forward(
        self,
        decoder_input_ids_vm: Optional[torch.LongTensor] = None,
        decoder_attention_mask_vm: Optional[torch.LongTensor] = None,
        input_ids_lm: Optional[torch.LongTensor] = None,
        attention_mask_lm: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        decoder_inputs_embeds_vm: Optional[torch.Tensor] = None,
    ):
        past_vm, past_lm = self.split_past(past_key_values)

        vids = decoder_input_ids_vm
        vmask = decoder_attention_mask_vm
        num_imgs = vids.shape[1]
        vids = rearrange(vids, 'b n l -> (b n) l')
        if vmask is not None:
            vmask = rearrange(vmask, 'b n l -> (b n) l')
        _encoder_outputs = copy.deepcopy(encoder_outputs)
        _encoder_outputs.last_hidden_state = rearrange(
            _encoder_outputs.last_hidden_state,
            'b n l c -> (b n) l c'
        )
        vm_output = self.vm.language_model(
            decoder_input_ids=vids,
            decoder_attention_mask=vmask,
            encoder_outputs=_encoder_outputs,
            past_key_values=past_vm,
            use_cache=use_cache,
            return_dict=True
        )

        if past_lm is not None:  # not the first step
            input_ids_lm = self.converter.ids_vm_to_lm(decoder_input_ids_vm[:, 0])
            attention_mask_lm = decoder_attention_mask_vm
            if attention_mask_lm is not None:
                attention_mask_lm = attention_mask_vm[:, 0]

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

        past_vm = vm_output.past_key_values
        past_lm = lm_output.past_key_values
        past_key_values = self.merge_past(past_vm, past_lm)

        return VLMOutput(
            logits=logits,
            past_key_values=past_key_values,
            encoder_outputs=encoder_outputs,
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
            decoder_start_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            **generate_kwargs,
        )

        return outputs
