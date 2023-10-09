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
    lm_name: str = 'TheBloke/vicuna-7B-1.1-HF'
    vm_name: str = 'Salesforce/blip2-flan-t5-xl'

    max_length: int = 50
    temperature: float = 0  # greedy
    top_p: float = 0.9
    num_beams: int = 1

    fluency_alpha: float = 0.8
    fluency_threshold: float = 0.00001

    demo_port: int = 8601


def get_args():
    parser = ArgumentParser()
    parser.add_arguments(Params, dest='config')
    args = parser.parse_args()
    args = args.config
    return args


def run(types, inp, instruction, inputs, max_length, temperature, top_p, num_beams,
        fluency_alpha, fluency_threshold):
    print(f'instruction: {instruction}')
    print(f'inputs: {inputs}')
    max_length = int(max_length)
    num_beams = int(num_beams)
    temperature = float(temperature)
    top_p = float(top_p)
    model.hparams.fluency_alpha = fluency_alpha
    model.hparams.fluency_threshold = fluency_threshold
    if len(inputs.strip()) == 0:
        inputs = None
    img = Image.fromarray(np.uint8(inp)).convert('RGB')
    result = model.run(img, instruction, inputs,
                       num_beams=num_beams, max_new_tokens=max_length,
                       temperature=temperature, top_p=top_p)
    print(f'res: {result}')
    result = result.lower()
    desc = f'## {types} / {inputs}'
    return desc, instruction, inputs, result
    # return inp, instruction, inputs, result


args = get_args()
print('loading model')
model = GenerationModel(args.lm_name, args.vm_name, verbose=True)
desc = f'VLIS {args.lm_name} {args.vm_name}'


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
                    gr.Textbox(label='inputs', lines=1),
                    gr.Slider(1, 300, value=args.max_length, label='max_length', step=1),
                    gr.Slider(0, 2, value=args.temperature, label='temperature'),
                    gr.Slider(0, 1, value=args.top_p, label='top_p'),
                    gr.Slider(1, 10, value=args.num_beams, label='num_beams', step=1),
                    gr.Slider(0, 2, value=args.fluency_alpha, label='fluency_alpha'),
                    gr.Slider(0, 1, value=args.fluency_threshold, label='fluency_threshold'),
                ]
            with gr.Column(scale=1):
                outputs = [
                    gr.Markdown(label='Type'),
                    gr.Textbox(label='instruction', lines=3),
                    gr.Textbox(label='inputs', lines=1),
                    gr.Textbox(label='generation', lines=5)
                ]
        with gr.Row():
            btn = gr.Button(value="Submit")
            btn.click(run, inputs=inputs, outputs=outputs, show_progress=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.demo_port
    )
