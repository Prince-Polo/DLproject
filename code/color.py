import cv2
import torch
from PIL import Image
import os
from rembg import remove
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
# from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, AutoencoderKL
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from controlnet_aux.canny import CannyDetector
# from config import Config
from PIL import ImageOps

generator= torch.Generator("cuda").manual_seed(0)

def get_canny_thin(img):
    # image = cv2.imread(image_path)
    img = np.array(img.convert("RGB"))[:, :, ::-1]
    image = cv2.resize(img, (1024, 1024))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image_gray, 100, 200)

    try:
        edges_thin = cv2.ximgproc.thinning(edges)
    except AttributeError:
        raise RuntimeError("请安装 opencv-contrib-python: pip install opencv-contrib-python")

    edges_rgb = cv2.cvtColor(edges_thin, cv2.COLOR_GRAY2RGB)
    Image.fromarray(edges_rgb).save("canny_thin.png")
    return Image.fromarray(edges_rgb)

def clean_alpha(image, threshold=200):
    """将 RGBA 图像中的半透明边缘处理掉，避免阴影"""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    r, g, b, a = image.split()

    a = a.point(lambda p: 255 if p > threshold else 0)
    
    cleaned = Image.merge("RGBA", (r, g, b, a))
    return cleaned

def get_canny(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1024, 1024))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_gray, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

# 2. 加载 T2I-Adapter（canny）



# image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     image=canny_detector(condition_image, detect_resolution=384, image_resolution=1024),
#     num_inference_steps=30,
#     guidance_scale=7.5,
#     adapter_conditioning_scale=1.0,
#     callback=save_intermediate_images,
#     callback_steps=1, 
# ).images[0]
negative_prompt1 = "monochrome, grayscale, blurry"

def paint(save_pth,prompt1,condition_img,negative_prompt=negative_prompt1):
    print(prompt1)
    print(negative_prompt)
    condition_img.save("condition_image.png") 
    image = pipe(
    prompt=prompt1,
    negative_prompt=negative_prompt,
    image=get_canny_thin(condition_img),
    num_inference_steps=200,
    guidance_scale=7.5,
    adapter_conditioning_scale=1.0,
    generator=generator,
    ).images[0]
    image = remove(image)  
    image = clean_alpha(image)
    white_bg = Image.new("RGB", image.size, (255, 255, 255))

    white_bg.paste(image, mask=image.split()[3]) 
    print("save_pth_chk", save_pth)
    white_bg.save(save_pth.replace("output.png", "output_white.png"))
    return 0

adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

# 3. 加载 SDXL + T2I-Adapter 管道
# pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     adapter=adapter,
#     torch_dtype=torch.float16,
#     variant="fp16"
# ).to("cuda")
canny_detector = CannyDetector()

model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()


# 4. 准备输入图
# condition_image = Image.fromarray(cv2.imread("/root/autodl-tmp/DDColor/test2.png") ) # ← 你的图
# condition_image.save("condition_image.png")  # 保存输入图像以供参考
# # 5. 提示词
# # prompt = "paint the"+", cartoon"
# # prompt="the background should be pure white with out any objects, isolated, no shadows, no texture, background color #ffffff"
# canny_detector(condition_image, detect_resolution=384, image_resolution=1024).save("canny_image.png")  # 保存边缘图像以供参考
# # 6. 生成图像

intermediate_images = []
# image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     image=canny_detector(condition_image, detect_resolution=384, image_resolution=1024),
#     num_inference_steps=30,
#     guidance_scale=7.5,
#     adapter_conditioning_scale=1.0,
# ).images[0]
# image=remove(image)  # 移除背景
# # 7. 保存结果
# image.save("result_t2i_adapter.png")

# white_bg = Image.new("RGB", image.size, (255, 255, 255))
# # 将透明图像粘贴到白色背景上（使用 alpha 通道作为掩码）
# white_bg.paste(image, mask=image.split()[3])  # image.split()[3] 是 alpha 通道

# # 保存结果
# white_bg.save("result_t2i_adapter_white.png")
