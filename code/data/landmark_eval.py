import json
from dataclasses import dataclass, field
from collections import defaultdict

from tqdm import tqdm
import numpy as np
from simple_parsing import ArgumentParser


@dataclass
class Params:
    path: str = '../../data/outputs/landmarks/what_is_this?.json'


def get_args(Config=Params):
    parser = ArgumentParser()
    parser.add_arguments(Config, dest='config')
    args = parser.parse_args()
    args = args.config
    return args


def get_acc(keys, key, txt):
    correct = False
    for kw in keys[key]:
        if kw.lower() in txt:
            correct = True
            break
    tried = False
    for xk in keys.keys():
        for kw in keys[xk]:
            if kw.lower() in txt:
                tried = True
                break
    return correct, tried


def main():
    args = get_args()
    with open(args.path) as f:
        hypos = json.load(f)

    with open('keys_landmarks.json') as f:
        keys = json.load(f)
    keys = {k: [k, *v] for k, v in keys.items()}

    outs = defaultdict(lambda: [])
    tries = defaultdict(lambda: [])
    for key in keys.keys():
        row = hypos[key]

        for model, out in row.items():
            flag = False
            txt = row[model].lower()
            flag, tried = get_acc(keys, key, txt)
            outs[model].append(flag)
            tries[model].append(tried)

    stats = {
        'acc': {k: f'{np.array(v).mean() * 100:.2f}%' for k, v in outs.items()},
        'tries': {k: f'{np.array(v).mean() * 100:.2f}%' for k, v in tries.items()}
    }
    print(stats)


if __name__ == '__main__':
    main()
