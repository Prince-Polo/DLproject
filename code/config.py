import argparse
import os.path as osp
import yaml
import random
from easydict import EasyDict as edict
import numpy.random as npr
import torch
from utils import (
    edict_2_dict,
    check_and_create_dir,
    update)
import wandb
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="code/config/base.yaml")
    parser.add_argument("--experiment", type=str, default="conformal_0.5_dist_pixel_100_kernel201")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--log_dir', metavar='DIR', default="output")

    parser.add_argument("--mode", type=str, choices=["word", "svg", "jpg", "png"], required=True,
                        help="choose among word, svg, jpg, or png modes")
    
    # for word mode
    parser.add_argument("--font", type=str, default="none", help="font name")
    parser.add_argument("--word", type=str, default="none", help="the text to work on")
    parser.add_argument("--optimized_letter", type=str, default="none", help="the letter in the word to optimize")

    # for svg mode
    parser.add_argument("--svg_path", type=str, default="none", help="path to input SVG")

    # for jpg mode
    parser.add_argument("--jpg_path", type=str, default="none", help="path to input JPG")
    # for png mode
    parser.add_argument("--png_path", type=str, default="none", help="path to input PNG")

    # shared
    parser.add_argument("--semantic_concept", type=str, help="the semantic concept to insert")
    parser.add_argument("--prompt_suffix", type=str, default="minimal flat 2d vector. lineal color."
                                                             " trending on artstation")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_user", type=str, default="none")
    parser.add_argument("--color",type=bool, default=False, help="whether to use color or not")
    parser.add_argument("--color_prompt", type=str, default=None, help="the color prompt to use")
    # parser.add_argument('--font', type=str, default="none", help="font name")
    # parser.add_argument('--semantic_concept', type=str, help="the semantic concept to insert")
    # parser.add_argument('--word', type=str, default="none", help="the text to work on")
    # parser.add_argument('--prompt_suffix', type=str, default="minimal flat 2d vector. lineal color."
    #                                                          " trending on artstation")
    # parser.add_argument('--optimized_letter', type=str, default="none", help="the letter in the word to optimize")
    # parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--use_wandb', type=int, default=0)
    # parser.add_argument('--wandb_user', type=str, default="none")

    args = parser.parse_args()
    with open('TOKEN', 'r') as f:
        setattr(args, 'token', f.read().replace('\n', ''))

    cfg = edict()
    cfg.mode = args.mode
    cfg.config = args.config
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.semantic_concept = args.semantic_concept
    cfg.caption = f"a {args.semantic_concept}. {args.prompt_suffix}"
    cfg.batch_size = args.batch_size
    cfg.token = args.token
    cfg.use_wandb = args.use_wandb
    cfg.wandb_user = args.wandb_user
    cfg.log_dir = f"{args.log_dir}/{args.experiment}"
    cfg.color = args.color
    cfg.color_prompt = args.color_prompt

    if cfg.mode == 'word':
        cfg.font = args.font
        cfg.word = cfg.semantic_concept if args.word == "none" else args.word
        if " " in cfg.word:
            raise ValueError("No spaces are allowed in word mode.")
        cfg.log_dir = f"{args.log_dir}/{args.experiment}_{cfg.word}"
        if args.optimized_letter not in cfg.word:
            raise ValueError("The optimized letter must be part of the word.")
        cfg.optimized_letter = args.optimized_letter
        cfg.letter = f"{args.font}_{args.optimized_letter}_scaled"
        cfg.target = f"code/data/init/{cfg.letter}"
    elif cfg.mode == 'svg':
        if args.svg_path == "none":
            raise ValueError("You must specify the path.")
        cfg.svg_path = f"{args.svg_path}.svg"
        cfg.word = osp.splitext(osp.basename(cfg.svg_path))[0]
        cfg.log_dir = f"{args.log_dir}/{args.experiment}_{cfg.word}"
        cfg.letter = "svg_shape"
        cfg.target = f"{args.svg_path}_scaled"  # the initial image to be deformed
    elif cfg.mode == 'jpg':
        if args.jpg_path == "none":
            raise ValueError("You must specify the path.")
        cfg.jpg_path = f"{args.jpg_path}.jpg"
        cfg.word = osp.splitext(osp.basename(cfg.jpg_path))[0]
        cfg.log_dir = f"{args.log_dir}/{args.experiment}_{cfg.word}"
        cfg.letter = "jpg_shape"
        cfg.target = f"{args.jpg_path}_scaled"  # the initial image to be deformed
    elif cfg.mode == 'png':
        if args.png_path == "none":
            raise ValueError("You must specify the path.")
        cfg.png_path = f"{args.png_path}.png"
        cfg.word = osp.splitext(osp.basename(cfg.png_path))[0]
        cfg.log_dir = f"{args.log_dir}/{args.experiment}_{cfg.word}"
        cfg.letter = "png_shape"
        cfg.target = f"{args.png_path}_scaled"  # the initial image to be deformed
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    return cfg


def set_config():
    print("Setting up configuration...")

    cfg_arg = parse_args()
    with open(cfg_arg.config, 'r') as f:
        cfg_full = yaml.load(f, Loader=yaml.FullLoader)

    # recursively traverse parent_config pointers in the config dicts
    cfg_key = cfg_arg.experiment
    cfgs = [cfg_arg]
    while cfg_key:
        cfgs.append(cfg_full[cfg_key])
        cfg_key = cfgs[-1].get('parent_config', 'baseline')

    # allowing children configs to override their parents
    cfg = edict()
    for options in reversed(cfgs):
        update(cfg, options)
    del cfgs

    # set experiment dir
    signature = ""
    if cfg.mode == 'word':
        signature = f"font_{cfg.letter}_concept_{cfg.semantic_concept}_seed_{cfg.seed}"
        cfg.experiment_dir = osp.join(cfg.log_dir, cfg.font, signature)
    elif cfg.mode == 'svg':
        signature = f"svg_{osp.basename(cfg.svg_path)}_concept_{cfg.semantic_concept}_seed_{cfg.seed}"
        cfg.experiment_dir = osp.join(cfg.log_dir, "svg", signature)
    elif cfg.mode == 'jpg':
        signature = f"jpg_{osp.basename(cfg.jpg_path)}_concept_{cfg.semantic_concept}_seed_{cfg.seed}"
        cfg.experiment_dir = osp.join(cfg.log_dir, "jpg", signature)
    elif cfg.mode == 'png':
        signature = f"png_{osp.basename(cfg.png_path)}_concept_{cfg.semantic_concept}_seed_{cfg.seed}"
        cfg.experiment_dir = osp.join(cfg.log_dir, "png", signature)
    # signature = f"{cfg.letter}_concept_{cfg.semantic_concept}_seed_{cfg.seed}"
    # cfg.experiment_dir = \
    #     osp.join(cfg.log_dir, cfg.font, signature)
    configfile = osp.join(cfg.experiment_dir, 'config.yaml')
    print('Config:', cfg)

    # create experiment dir and save config
    check_and_create_dir(configfile)
    with open(osp.join(configfile), 'w') as f:
        yaml.dump(edict_2_dict(cfg), f)

    if cfg.use_wandb:
        wandb.init(project="Word-As-Image", entity=cfg.wandb_user,
                   config=cfg, name=f"{signature}", id=wandb.util.generate_id())

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = False
    else:
        assert False

    return cfg



# utils_config.py
from types import SimpleNamespace
import yaml
import os.path as osp

def generate_cfg_dict(args_dict):
    args = SimpleNamespace(**args_dict)
    cfg = {}

    cfg["mode"] = args.mode
    cfg["config"] = args.config
    cfg["experiment"] = args.experiment
    cfg["seed"] = args.seed
    cfg["semantic_concept"] = args.semantic_concept
    cfg["caption"] = f"a {args.semantic_concept}. {args.prompt_suffix}"
    cfg["batch_size"] = args.batch_size
    cfg["token"] = args.token
    cfg["use_wandb"] = args.use_wandb
    cfg["wandb_user"] = args.wandb_user
    cfg["log_dir"] = f"{args.log_dir}/{args.experiment}"
    cfg["color"] = args.color
    cfg["color_prompt"] = args.color_prompt

    if args.mode == 'word':
        cfg["font"] = args.font
        cfg["word"] = args.semantic_concept if args.word == "none" else args.word
        if " " in cfg["word"]:
            raise ValueError("No spaces are allowed in word mode.")
        cfg["log_dir"] = f"{args.log_dir}/{args.experiment}_{cfg['word']}"
        if args.optimized_letter not in cfg["word"]:
            raise ValueError("The optimized letter must be part of the word.")
        cfg["optimized_letter"] = args.optimized_letter
        cfg["letter"] = f"{args.font}_{args.optimized_letter}_scaled"
        cfg["target"] = f"code/data/init/{cfg['letter']}"
    elif args.mode == 'svg':
        if args.svg_path == "none":
            raise ValueError("You must specify the path.")
        cfg["svg_path"] = f"{args.svg_path}.svg"
        cfg["word"] = osp.splitext(osp.basename(cfg["svg_path"]))[0]
        cfg["log_dir"] = f"{args.log_dir}/{args.experiment}_{cfg['word']}"
        cfg["letter"] = "svg_shape"
        cfg["target"] = f"{args.svg_path}_scaled"
    elif args.mode == 'jpg':
        if args.jpg_path == "none":
            raise ValueError("You must specify the path.")
        cfg["jpg_path"] = f"{args.jpg_path}.jpg"
        cfg["word"] = osp.splitext(osp.basename(cfg["jpg_path"]))[0]
        cfg["log_dir"] = f"{args.log_dir}/{args.experiment}_{cfg['word']}"
        cfg["letter"] = "jpg_shape"
        cfg["target"] = f"{args.jpg_path}_scaled"
    elif args.mode == 'png':
        if args.png_path == "none":
            raise ValueError("You must specify the path.")
        cfg["png_path"] = f"{args.png_path}.png"
        cfg["word"] = osp.splitext(osp.basename(cfg["png_path"]))[0]
        cfg["log_dir"] = f"{args.log_dir}/{args.experiment}_{cfg['word']}"
        cfg["letter"] = "png_shape"
        cfg["target"] = f"{args.png_path}_scaled"
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    return cfg
