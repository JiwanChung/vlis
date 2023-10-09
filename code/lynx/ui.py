import gradio as gr
import platform
from typing import Tuple
import os
import sys
import time
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import torch
from simple_parsing import ArgumentParser
from PIL import Image
import numpy as np
import gradio as gr

from generate import GenerationModel


@dataclass
class Params:
    lm_name: str = 'lmsys/vicuna-7b-v1.5'
    vm_name: str = '../../data/lynx'

    max_length: int = 50
    temperature: float = 0  # greedy
    top_p: float = 0.9
    num_beams: int = 1

    fluency_alpha: float = 0.3
    fluency_threshold: float = 0.0001
    visual_threshold: float = 0

    demo_port: int = 8601


def get_args():
    parser = ArgumentParser()
    parser.add_arguments(Params, dest='config')
    args = parser.parse_args()
    args = args.config
    return args


def run(types, inp, instruction_lm, instruction_vm,
        prefix_lm, prefix_vm,
        baseline, max_length, temperature, top_p, num_beams, length_penalty,
        fluency_alpha, fluency_threshold, visual_threshold):
    print(f'instruction: {instruction_lm}')
    max_length = int(max_length)
    num_beams = int(num_beams)
    temperature = float(temperature)
    top_p = float(top_p)
    length_penalty = float(length_penalty)
    model.hparams.fluency_alpha = fluency_alpha
    model.hparams.fluency_threshold = fluency_threshold
    model.hparams.visual_threshold = float(visual_threshold)
    img = Image.fromarray(np.uint8(inp)).convert('RGB')
    if len(prefix_lm) == 0:
        prefix_lm = None
    if len(prefix_vm) == 0:
        prefix_vm = None

    if len(instruction_vm.strip()) == 0:
        instruction_vm = None
    if instruction_vm is None:
        instruction_vm = instruction_lm
    model.hparams.run_baseline = baseline
    result = model.run(img, instruction_lm, prefix_lm=prefix_lm,
                       instruction_vm=instruction_vm,
                       prefix_vm=prefix_vm,
                       num_beams=num_beams, max_new_tokens=max_length,
                       temperature=temperature, top_p=top_p,
                       length_penalty=length_penalty)
    print(f'res: {result}')
    desc = f'## {types}'
    if prefix_lm is None:
        prefix_lm = '<None>'
    if prefix_vm is None:
        prefix_vm = '<None>'
    return desc, instruction_lm, instruction_vm, prefix_lm, prefix_vm, result


args = get_args()
print('loading model')
model = GenerationModel(args.lm_name, args.vm_name, verbose=True)
desc = f'MCD {args.lm_name} {args.vm_name}'


if __name__ == "__main__":
    print(f"running from {platform.node()}")
    with gr.Blocks() as demo:
        gr.Markdown(desc)
        with gr.Row():
            with gr.Column(scale=1):
                inputs = [
                    gr.Textbox(label='types', value='', visible=False),
                    gr.Image(),
                    gr.Textbox(label='instruction', lines=3),
                    gr.Textbox(label='instruction_vm', lines=3),
                    gr.Textbox(label='prefix_lm', lines=1),
                    gr.Textbox(label='prefix_vm', lines=1),
                    gr.Radio(choices=['ours', 'lm', 'vm', 'ours_vm', 'ours_lm']),
                    gr.Slider(1, 300, value=args.max_length, label='max_length', step=1),
                    gr.Slider(0, 5, value=args.temperature, label='temperature'),
                    gr.Slider(0, 1, value=args.top_p, label='top_p'),
                    gr.Slider(1, 10, value=args.num_beams, label='num_beams', step=1),
                    gr.Slider(-5, 5, value=0, label='length_penalty'),
                    gr.Slider(0, 2, value=args.fluency_alpha, label='fluency_alpha'),
                    gr.Slider(0, 1, value=args.fluency_threshold, label='fluency_threshold'),
                    gr.Slider(0, 1, value=args.visual_threshold, label='visual_threshold'),
                ]
            with gr.Column(scale=1):
                outputs = [
                    # gr.Image(shape=(None, 512)),
                    gr.Markdown(label='Type'),
                    gr.Textbox(label='instruction', lines=3),
                    gr.Textbox(label='instruction_vm', lines=3),
                    gr.Textbox(label='prefix_lm', lines=1),
                    gr.Textbox(label='prefix_vm', lines=1),
                    gr.Textbox(label='generation', lines=5)
                ]
        with gr.Row():
            btn = gr.Button(value="Submit")
            btn.click(run, inputs=inputs, outputs=outputs, show_progress=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.demo_port
    )
