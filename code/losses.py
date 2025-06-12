import torch.nn as nn
import torchvision
from scipy.spatial import Delaunay
import torch
import numpy as np
from torch.nn import functional as nnf
from easydict import EasyDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import model_info

class SDSLoss(nn.Module):
    def __init__(self, cfg, device):
        super(SDSLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(cfg.diffusion.model,
                                                       torch_dtype=torch.float16, use_auth_token=cfg.token)
        ## lora
        # model_path = "sayakpaul/sd-model-finetuned-lora-t4"
        # info = model_info(model_path)
        # model_base = info.cardData["base_model"]
        # self.pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        # lora_path = "/root/autodl-tmp/git_resp/DLproject/lora/Pecha_Swords_LORA_V1.2.safetensors"
        # self.pipe.load_lora_weights(lora_path)

        # lora_path = "/root/autodl-tmp/git_resp/DLproject/lora/pytorch_lora_weights1.safetensors"
        # self.pipe.load_lora_weights(lora_path)

        self.pipe = self.pipe.to(self.device)
        # default scheduler: PNDMScheduler(beta_start=0.00085, beta_end=0.012,
        # beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        self.text_embeddings = None
        self.embed_text()

    def embed_text(self):
        # tokenizer and embed text
        text_input = self.pipe.tokenizer(self.cfg.caption, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                         max_length=text_input.input_ids.shape[-1],
                                         return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        self.text_embeddings = self.text_embeddings.repeat_interleave(self.cfg.batch_size, 0)
        del self.pipe.tokenizer
        del self.pipe.text_encoder


    def forward(self, x_aug):
        sds_loss = 0

        # encode rendered image
        x = x_aug * 2. - 1.
        with torch.cuda.amp.autocast():
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample())
        latent_z = 0.18215 * init_latent_z  # scaling_factor * init_latents

        with torch.inference_mode():
            # sample timesteps
            timestep = torch.randint(
                low=50,
                high=min(950, self.cfg.diffusion.timesteps) - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            eps = torch.randn_like(latent_z)
            # zt = alpha_t * latent_z + sigma_t * eps
            noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = self.pipe.unet(z_in, timestep, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)

            eps_t = eps_t_uncond + self.cfg.diffusion.guidance_scale * (eps_t - eps_t_uncond)

            # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
            grad_z = self.alphas[timestep]**0.5 * self.sigmas[timestep] * (eps_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

        sds_loss = grad_z.clone() * latent_z
        del grad_z

        sds_loss = sds_loss.sum(1).mean()
        return sds_loss


class ToneLoss(nn.Module):
    def __init__(self, cfg):
        super(ToneLoss, self).__init__()
        self.dist_loss_weight = cfg.loss.tone.dist_loss_weight
        self.im_init = None
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.blurrer = torchvision.transforms.GaussianBlur(kernel_size=(cfg.loss.tone.pixel_dist_kernel_blur,
                                                                        cfg.loss.tone.pixel_dist_kernel_blur), sigma=(cfg.loss.tone.pixel_dist_sigma))

    def set_image_init(self, im_init):
        self.im_init = im_init.permute(2, 0, 1).unsqueeze(0)
        self.init_blurred = self.blurrer(self.im_init)


    def get_scheduler(self, step=None):
        if step is not None:
            return self.dist_loss_weight * np.exp(-(1/5)*((step-300)/(20)) ** 2)
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        blurred_cur = self.blurrer(cur_raster)
        return self.mse_loss(self.init_blurred.detach(), blurred_cur) * self.get_scheduler(step)
            

class ConformalLoss_word:
    def __init__(self, parameters: EasyDict, device: torch.device, target_letter: str, shape_groups):
        self.parameters = parameters
        self.target_letter = target_letter
        self.shape_groups = shape_groups
        self.faces = self.init_faces(device)
        self.device = device
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))]

        with torch.no_grad():
            self.angles = []
            self.reset()


    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_
    
    def get_letter_inds(self, letter_to_insert):
        for group, l in zip(self.shape_groups, self.target_letter):
            if l == letter_to_insert:
                letter_inds = group.shape_ids
                return letter_inds[0], letter_inds[-1], len(letter_inds)

    def reset(self):
        points = torch.cat([point.clone().detach() for point in self.parameters.point]).to(self.device)
        self.angles = self.get_angles(points)

    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [self.parameters.point[i].clone().detach().cpu().numpy() for i in range(len(self.parameters.point))]
            start_ind, end_ind, shapes_per_letter = self.get_letter_inds(c)
            print(c, start_ind, end_ind)
            holes = []
            if shapes_per_letter > 1:
                holes = points_np[start_ind+1:end_ind]
            poly = Polygon(points_np[start_ind], holes=holes)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=bool)
            faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_

    def __call__(self) -> torch.Tensor:
        loss_angles = 0
        points = torch.cat(self.parameters.point).to(self.device)
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            loss_angles += (nnf.mse_loss(angles[i], self.angles[i]))
        return loss_angles

class ConformalLoss_svg:
    def __init__(self, parameters: EasyDict, device: torch.device, shape_groups):
        self.parameters = parameters
        self.shape_groups = shape_groups
        self.device = device
        self.faces = self.init_faces()
        self.faces_roll_a = [torch.roll(f, 1, 1) for f in self.faces]

        with torch.no_grad():
            self.angles = []
            self.reset()

    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_

    def reset(self):
        points = torch.cat([p.clone().detach() for p in self.parameters.point]).to(self.device)
        self.angles = self.get_angles(points)

    def init_faces(self) -> torch.Tensor:
        faces_all = []
        for i, group in enumerate(self.shape_groups):
            try:
                shape_ids = group.shape_ids
                start_ind, end_ind = shape_ids[0], shape_ids[-1]
                shapes_per_group = len(shape_ids)

                # 获取每个子 path 的点
                points_np_all = [self.parameters.point[i].clone().detach().cpu().numpy()
                                for i in range(len(self.parameters.point))]

                outer = points_np_all[start_ind]
                holes = points_np_all[start_ind + 1:end_ind + 1] if shapes_per_group > 1 else []

                poly = Polygon(outer, holes=holes).buffer(0)

                # 将当前 shape group 内所有点提取出来
                group_points = np.concatenate([points_np_all[idx] for idx in shape_ids])

                tri = Delaunay(group_points)
                is_inside = np.array([
                    poly.contains(Point(group_points[face].mean(0)))
                    for face in tri.simplices
                ], dtype=bool)

                faces = torch.from_numpy(tri.simplices[is_inside]).to(self.device, dtype=torch.int64)
                faces_all.append(faces)

            except Exception as e:
                print(f"Warning: failed to build polygon for shape_group[{i}]: {e}")
        return faces_all

    def __call__(self) -> torch.Tensor:
        if not self.faces:
            return torch.tensor(0.0, device=self.device)

        points = torch.cat(self.parameters.point).to(self.device)
        angles = self.get_angles(points)
        loss_angles = sum(nnf.mse_loss(angles[i], self.angles[i]) for i in range(len(self.faces)))
        return loss_angles



