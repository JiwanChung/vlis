root='../../data/models/llava'

python -m llava.model.apply_delta \
    --base $root/llama-13b \
    --target $root/LLaVA-13B-v0 \
    --delta liuhaotian/LLaVA-13b-delta-v0
