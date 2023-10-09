## Introduction

Code for Alpaca version of Multimodal Contrastive Decoding framework.
The code may perform better or worse than OPT or T5 version of MCD depending on the task.
However, it generally shows better responsibility to natural language instructions.

MCD here uses Blip2-FLAN-T5 and Alpaca-LoRA.

*WARNING*: The code is confidential as of now. Do not share it with anyone outside of your project.
Please keep the confidentiality until the ICCV review process is completed.

## Installation

```
pip install -r requirements.txt
mkdir ../../data  # data folder for weights cache, etc.
```

*WARNING*: You need the master branch of the github `transformers` to run Llama models. This update is not yet integrated into a official release as of 23 May 2023.


## Testing

```
python test.py --lm_name 7b
```

*WARNING*: `7b` denotes Alpaca-LoRA with Llama-7B. We currently only support 7B sized backbone.
While Alpaca-LoRA weights for Llama 13B and 30B are available on huggingface, they are somewhat buggy as of now.


## Interactive Testing

```
python ui.py --lm_name 7b
```
