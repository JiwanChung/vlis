## Installation

```
pip install -r requirements.txt
mkdir ../../data  # data folder for weights cache, etc.
```

## Preparing Model Weights

1. Download LLaVA 13b v0 delta weights from [huggingface](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0) following the instructions of [github](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).

2. Download LLama weights as well following the [instructions](https://huggingface.co/docs/transformers/main/model_doc/llama).

3. Store both under `$REPO/data/models/llava` and run `bash ./get_weights.sh` to merge the delta weights.

Caveat: LLaVA v1.5 introduces breaking changes from v0. We are updating our code to support v1.5. In the meantime, please use LLaVA v0 with VLIS.

## Interactive Testing

```
python ui.py
```
