import os
import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers.image_processing_utils import BatchFeature
from transformers.models.blip_2.processing_blip_2 import Blip2Processor

from lynx_llm.models.lynx import LynxBase
from lynx_llm.dataset import get_image_transform
from lynx_llm.models.llms.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from lynx_llm.models.bridges.resampler import PerceiverResampler
from lynx_llm.models.llms.llama.tokenization_llama import LlamaTokenizer
from lynx_llm.models.vits.eva_vit import (
    VisionTransformer, interpolate_pos_embed
)


class LynxProcessor(Blip2Processor):
    def __init__(self, processor, tokenizer):
        self.image_processor = processor
        self.tokenizer = tokenizer


class ImageProcessor:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, return_tensors = None):
        images = [self.transforms(img) for img in images]
        images = torch.stack(images, 0)
        encoded_outputs = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)
        return encoded_outputs


DEFAULT_PAD_TOKEN = "[PAD]"
TOKEN_NONE_FLAG = "[NONE]"


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Lynx(LynxBase):
    def __init__(self, config=None, freeze_vit=True, freeze_llm=True, load_bridge=False):
        nn.Module.__init__(self)

        self.lm_name = 'lmsys/vicuna-7b-v1.1'
        self.root = Path(__file__).parent.parent.parent / 'data/lynx'

        self.freeze_vit = freeze_vit
        self.freeze_llm = freeze_llm

        self.init_params = []

        self.vision_encoder, missing_keys = self.build_vision_encoder(config, freeze_params=freeze_vit)
        self.update_init_params([f'vision_encoder.{k}' for k in missing_keys])

        self.tokenizer, num_new_tokens = build_tokenizer(self.lm_name,
                                                         use_left_pad=config['use_left_pad'])
        self.LLM, missing_keys = self.build_LLM(config, freeze_params=freeze_llm, num_new_tokens=num_new_tokens)
        self.update_init_params([f'LLM.{k}' for k in missing_keys])

        # Bridge: Vision2Text
        self.bridge, missing_keys = self.build_bridge(config, load_params=load_bridge)
        self.update_init_params([f'bridge.{k}' for k in missing_keys])

        # Video
        self.video_encoding = False
        '''
        self.video_encoding = config.get("video_encoding", "")
        if self.video_encoding:
            self.add_frame_pos = config['add_frame_pos']
            if self.add_frame_pos:
                self.absolute_frame_pos_embed = nn.Parameter(
                    torch.zeros(1, config['data']['num_frames'], 1, self.vision_width))
                trunc_normal_(self.absolute_frame_pos_embed, std=.02)
                self.update_init_params(['absolute_frame_pos_embed'])

            elif self.video_encoding == 'concate':
                # concate all video frames
                pass

            else:
                raise NotImplementedError(f"video_encoding == {self.video_encoding}")
        '''
        ckpt = str(self.root / Path(config['checkpoint']).name)
        self.load_pretrained(ckpt, config)

    def build_vision_encoder(self, config, freeze_params=True):
        """
        Args:
            load_params: False when building fine-tuning models
        """
        print(f"### Building ViT (Freeze: {freeze_params})", flush=True)

        if config['vision_encoder'] == 'eva_vit_1b':
            model, missing_keys = create_eva_vit_g(self.root,
                                                   config['image_res'],
                                                   config.get('drop_path_rate', 0.0),
                                                   load_params=True)
            # set attrs
            self.vision_width = model.embed_dim

        else:
            raise NotImplementedError("Vision Encoder: ", config['vision_encoder'])

        if freeze_params:

            assert len(missing_keys) == 0

            for name, param in model.named_parameters():
                param.requires_grad = False

            model = model.eval()
            model.train = disabled_train

        return model, missing_keys

    def build_LLM(self, config, freeze_params=True, num_new_tokens=0):
        print(f"### Building LLM (Freeze: {freeze_params})", flush=True)

        self.use_dec_only = True
        self.use_adapter = config.get('use_adapter', False)

        if config['LLM'] in ['vicuna-7b']:

            text_config = LlamaConfig.from_json_file(str(Path(__file__).parent / "llama_config.json"))

            text_config.use_flash_attn = config.get("use_flash_attn", False)

            text_config.use_adapter = self.use_adapter
            text_config.adapter_freq = config.get('adapter_freq', -1)
            text_config.freeze_params = freeze_params
            text_config.label_smoothing = config.get("label_smoothing", 0.0)

            model = LlamaForCausalLM.from_pretrained(
                self.lm_name, config=text_config,
                # load_in_8bit=True,
                # torch_dtype=torch.float16,
                # device_map="auto",
            )

            model.model.padding_idx = self.tokenizer.pad_token_id

            missing_keys = [n for n, _ in model.named_parameters() if 'adapter' in n]

            # set attrs
            self.text_width = model.config.hidden_size
            decoder_layers_attr_name = "model.layers"

        else:
            raise NotImplementedError("LLM: ", config['LLM'])

        if num_new_tokens > 0:
            print("### LLM Vocab Size: ", model.config.vocab_size, flush=True)
            print("### num_new_tokens: ", num_new_tokens, flush=True)
            vocab_size = model.config.vocab_size + num_new_tokens
            assert vocab_size == len(self.tokenizer)

            model.resize_token_embeddings(vocab_size)
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        if freeze_params:
            print("### Freeze LLM", flush=True)
            for name, param in model.named_parameters():
                if 'adapter' in name:
                    pass
                else:
                    param.requires_grad = False

        return model, missing_keys

    def build_bridge(self, config, load_params=True):
        """
        Bridge for Vision to Text
        """
        print("### Building Bridge", flush=True)
        missing_keys = []

        if config['bridge'] == 'resampler':

            model = PerceiverResampler(self.vision_width, self.text_width,
                                       depth=config["bridge_depth"], num_latents=config["num_bridge_tokens"])
            assert load_params is False, "no param to load for Resampler"
            missing_keys = [n for (n, p) in model.named_parameters()]
        else:
            raise NotImplementedError("Bridge: ", config['bridge'])

        if load_params:
            print("missing_keys: ", missing_keys, flush=True)

        return model, missing_keys


def build_tokenizer(LLM: str, use_left_pad: bool):
    num_new_tokens = 0

    if 'vicuna' in LLM:

        tokenizer = LlamaTokenizer.from_pretrained(LLM)

        if tokenizer.pad_token is None:
            num_new_tokens = tokenizer.add_special_tokens(
                {
                    "pad_token": DEFAULT_PAD_TOKEN,
                }
            )

    else:
        raise NotImplementedError("Tokenizer for LLM: ", LLM)

    print("-" * 40)
    print("### Vocab Size: ", len(tokenizer), flush=True)

    assert tokenizer.eos_token is not None
    assert tokenizer.pad_token is not None

    if tokenizer.bos_token is None:
        print("set bos_token to: ", TOKEN_NONE_FLAG, flush=True)
        tokenizer.bos_token = TOKEN_NONE_FLAG

    else:
        print("bos_token, ", tokenizer.bos_token)
        print("bos_token_id, ", tokenizer.bos_token_id)

    if use_left_pad:
        tokenizer.padding_side = "left"

    print("Left Pad: ", use_left_pad, flush=True)

    print("eos_token, ", tokenizer.eos_token)
    print("eos_token_id, ", tokenizer.eos_token_id)

    print("pad_token, ", tokenizer.pad_token)
    print("pad_token_id, ", tokenizer.pad_token_id)

    print("unk_token, ", tokenizer.unk_token, flush=True)
    print("unk_token_id, ", tokenizer.unk_token_id, flush=True)
    print("-" * 40)

    return tokenizer, num_new_tokens


def load_lynx(model_name, device: str = 'cuda'):
    config_path = Path(__file__).parent / 'lynx_llm/configs/LYNX.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    model = Lynx(config=config,
                 freeze_vit=config['freeze_vit'], freeze_llm=config['freeze_llm'],
                 load_bridge=False)
    model = model.to(device)

    for _, param in model.named_parameters():
        param.requires_grad = False

    model.eval()

    tokenizer = model.tokenizer
    _, img_transform = get_image_transform(config)
    image_processor = ImageProcessor(img_transform)

    return model, LynxProcessor(image_processor, tokenizer)


def create_eva_vit_g(root, img_size=224, drop_path_rate=0.4,
                     load_params=True, use_checkpoint=False, vision_feats_return_layer=-1):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=14,
        use_mean_pooling=False,
        embed_dim=1408,
        depth=39,
        num_heads=1408 // 88,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=use_checkpoint,
        vision_feats_return_layer=vision_feats_return_layer,
    )

    msg = None
    if load_params:
        path = str(root / "eva_vit_g.pth")
        assert os.path.exists(path), f'path does not exist: {path}'
        state_dict = torch.load(path, map_location="cpu")
        interpolate_pos_embed(model, state_dict)

        # rename for StableAttention
        for n in list(state_dict.keys()):
            if '.attn.qkv.weight' in n:
                weight = state_dict[n]

                q, k, v = torch.split(weight, weight.shape[0]//3, dim=0)

                new_n = n.replace('.attn.qkv.weight', '.attn.q_proj.weight').strip()
                state_dict[new_n] = q

                new_n = n.replace('.attn.qkv.weight', '.attn.k_proj.weight').strip()
                state_dict[new_n] = k

                new_n = n.replace('.attn.qkv.weight', '.attn.v_proj.weight').strip()
                state_dict[new_n] = v

                del state_dict[n]

            elif '.attn.v_bias' in n:
                new_n = n.replace('.attn.v_bias', '.attn.v_proj.bias').strip()
                state_dict[new_n] = state_dict[n]
                del state_dict[n]

            elif '.attn.q_bias' in n:
                new_n = n.replace('.attn.q_bias', '.attn.q_proj.bias').strip()
                state_dict[new_n] = state_dict[n]
                del state_dict[n]

        msg = model.load_state_dict(state_dict, strict=False)

    return model, msg.missing_keys
