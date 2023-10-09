import os
import requests
from io import BytesIO

from PIL import Image
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
)
from transformers.processing_utils import ProcessorMixin
from transformers.models.blip_2.processing_blip_2 import Blip2Processor

from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class LlavaProcessor(Blip2Processor):
    def __init__(self, processor, tokenizer):
        self.image_processor = processor
        self.tokenizer = tokenizer


def load_llava(model_name):
    disable_torch_init()
    model_name = os.path.expanduser(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "mpt" in model_name.lower():
        model = LlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt_multimodal"
    else:
        conv_mode = "multimodal"

    model.conv_mode = conv_mode

    if mm_use_im_start_end:
        model.image_prompt = '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        model.image_prompt = '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    return model, LlavaProcessor(image_processor, tokenizer)
