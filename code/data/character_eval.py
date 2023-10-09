import json
from dataclasses import dataclass, field
from collections import defaultdict

from tqdm import tqdm
import numpy as np
from simple_parsing import ArgumentParser


@dataclass
class Params:
    path: str = '../../data/outputs/characters/who_is_this?.json'


def get_args(Config=Params):
    parser = ArgumentParser()
    parser.add_arguments(Config, dest='config')
    args = parser.parse_args()
    args = args.config
    return args


def main():
    args = get_args()
    with open(args.path) as f:
        hypos = json.load(f)

    with open('keys_characters.json') as f:
        keys = json.load(f)

    outs = defaultdict(lambda: [])
    for key in keys.keys():
        if key in hypos:
            row = hypos[key]

            for model, out in row.items():
                flag = False
                txt = row[model].lower()
                for kw in [key, *keys[key]]:
                    if kw.lower() in txt:
                        flag = True
                        break
                outs[model].append(flag)
        else:
            raise Exception()

    stats = {k: (f'{np.array(v).mean() * 100:.2f}%', len(v)) for k, v in outs.items()}
    print(stats)


if __name__ == '__main__':
    main()
