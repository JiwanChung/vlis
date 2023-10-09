## Installation

```
pip install -r requirements.txt
mkdir ../../data  # data folder for weights cache, etc.
```

## Preparing Model Weights

Follow the official guide [github](https://github.com/bytedance/lynx-llm) to download the required files.
You should store them under `$REPO/data/lynx` as:
- `$REPO/data/lynx/finetune_lynx.pt`
- `$REPO/data/lynx/eva_vit_g.pth`

## Interactive Testing

```
python ui.py
```
