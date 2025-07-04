from typing import Mapping
import os
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import save_svg
from losses import SDSLoss, ToneLoss, ConformalLoss_word, ConformalLoss_svg
from config import set_config
from torchvision.transforms.functional import to_pil_image
from utils import (
    check_and_create_dir,
    get_data_augs,
    save_image,
    preprocess_word,
    preprocess_svg,
    preprocess_img,
    learning_rate_decay,
    combine_word,
    create_video,
    replace_letter_with_png)
import wandb
import warnings
from color import paint
warnings.filterwarnings("ignore")

pydiffvg.set_print_timing(False)
gamma = 1.0


def init_shapes(svg_path, trainable: Mapping[str, bool]):

    svg = f'{svg_path}.svg'
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg)

    parameters = edict()

    # path points
    if trainable.point:
        parameters.point = []
        for path in shapes_init:
            path.points.requires_grad = True
            parameters.point.append(path.points)

    return shapes_init, shape_groups_init, parameters


if __name__ == "__main__":
    
    cfg = set_config()

    # use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    print("preprocessing")
    if cfg.mode == "word":
        preprocess_word(cfg.font, cfg.word, cfg.optimized_letter, cfg.level_of_cc)
    elif cfg.mode == "svg":
        preprocess_svg(cfg.svg_path)
    elif cfg.mode == "jpg":
        preprocess_img(cfg.jpg_path, cfg.mode)
    elif cfg.mode == "png":
        preprocess_img(cfg.png_path, cfg.mode)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    if cfg.loss.use_sds_loss:
        sds_loss = SDSLoss(cfg, device)

    h, w = cfg.render_size, cfg.render_size

    data_augs = get_data_augs(cfg.cut_size)

    render = pydiffvg.RenderFunction.apply

    # initialize shape
    print('initializing shape')
    shapes, shape_groups, parameters = init_shapes(svg_path=cfg.target, trainable=cfg.trainable)

    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
               torch.ones(img_init.shape[0], img_init.shape[1], 3, device=device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]
    if cfg.use_wandb:
        plt.imshow(img_init.detach().cpu())
        wandb.log({"init": wandb.Image(plt)}, step=0)
        plt.close()

    if cfg.loss.tone.use_tone_loss:
        tone_loss = ToneLoss(cfg)
        tone_loss.set_image_init(img_init)

    if cfg.save.init:
        print('saving init')
        filename = os.path.join(
            cfg.experiment_dir, "svg-init", "init.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(filename, w, h, shapes, shape_groups)

    num_iter = cfg.num_iter
    pg = [{'params': parameters["point"], 'lr': cfg.lr_base["point"]}]
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)

    if cfg.loss.conformal.use_conformal_loss:
        if cfg.mode == "word":
            conformal_loss = ConformalLoss_word(parameters, device, cfg.optimized_letter, shape_groups)
        elif cfg.mode == "svg" or cfg.mode == "jpg" or cfg.mode == "png":
            conformal_loss = ConformalLoss_svg(parameters, device, shape_groups)

    lr_lambda = lambda step: learning_rate_decay(step, cfg.lr.lr_init, cfg.lr.lr_final, num_iter,
                                                 lr_delay_steps=cfg.lr.lr_delay_steps,
                                                 lr_delay_mult=cfg.lr.lr_delay_mult) / cfg.lr.lr_init

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda, last_epoch=-1)  # lr.base * lrlambda_f

    print("start training")
    # training loop
    t_range = tqdm(range(num_iter))
    for step in t_range:
        if cfg.use_wandb:
            wandb.log({"learning_rate": optim.param_groups[0]['lr']}, step=step)
        optim.zero_grad()

        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        img = render(w, h, 2, 2, step, None, *scene_args)

        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        if cfg.save.video and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1):
            save_image(img, os.path.join(cfg.experiment_dir, "video-png", f"iter{step:04d}.png"), gamma)
            filename = os.path.join(
                cfg.experiment_dir, "video-svg", f"iter{step:04d}.svg")
            check_and_create_dir(filename)
            save_svg.save_svg(
                filename, w, h, shapes, shape_groups)
            if cfg.use_wandb:
                plt.imshow(img.detach().cpu())
                wandb.log({"img": wandb.Image(plt)}, step=step)
                plt.close()

        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        x = x.repeat(cfg.batch_size, 1, 1, 1)
        x_aug = data_augs.forward(x)

        # compute diffusion loss per pixel
        loss = sds_loss(x_aug)
        if cfg.use_wandb:
            wandb.log({"sds_loss": loss.item()}, step=step)

        if cfg.loss.tone.use_tone_loss:
            tone_loss_res = tone_loss(x, step)
            if cfg.use_wandb:
                wandb.log({"dist_loss": tone_loss_res}, step=step)
            loss = loss + tone_loss_res

        if cfg.loss.conformal.use_conformal_loss:
            loss_angles = conformal_loss()
            loss_angles = cfg.loss.conformal.angeles_w * loss_angles
            if cfg.use_wandb:
                wandb.log({"loss_angles": loss_angles}, step=step)
            loss = loss + loss_angles

        t_range.set_postfix({'loss': loss.item()})
        loss.backward()
        optim.step()
        scheduler.step()

    filename = os.path.join(
        cfg.experiment_dir, "output-svg", "output.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(
        filename, w, h, shapes, shape_groups)

    if cfg.mode == "word":
        combine_word(cfg.word, cfg.optimized_letter, cfg.font, cfg.experiment_dir)

    if cfg.save.image:
        filename = os.path.join(
            cfg.experiment_dir, "output-png", "output.png")
        check_and_create_dir(filename)
        imshow = img.detach().cpu()
        pydiffvg.imwrite(imshow, filename, gamma=gamma)
        if cfg.use_wandb:
            plt.imshow(img.detach().cpu())
            wandb.log({"img": wandb.Image(plt)}, step=step)
            plt.close()

    if cfg.save.video:
        print("saving video")
        create_video(cfg.num_iter, cfg.experiment_dir, cfg.save.video_frame_freq)

    if cfg.color:
        print("painting")
        condition_img_pil = to_pil_image((img * 255).byte().permute(2, 0, 1).cpu())
        if cfg.color_prompt!= None:
            print('test1')
            paint(
                save_pth=os.path.join(cfg.experiment_dir, "output-png", "output.png"),
                prompt1=cfg.color_prompt,
                condition_img=condition_img_pil
            )
        else:
            print('test2')
            paint(
                save_pth=os.path.join(cfg.experiment_dir, "output-png", "output.png"),
                prompt1="draw the "+cfg.semantic_concept,
                condition_img=condition_img_pil
            )
        if cfg.mode == "word":
            replace_letter_with_png(
                cfg.word, cfg.optimized_letter, cfg.font, cfg.experiment_dir,
                os.path.join(cfg.experiment_dir, "output-png", "output_white.png"))
    if cfg.use_wandb:
        wandb.finish()
