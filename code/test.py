from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

def get_canny_thin(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1024, 1024))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image_gray, 100, 200)

    # 细化操作（需安装 opencv-contrib-python）
    try:
        edges_thin = cv2.ximgproc.thinning(edges)
    except AttributeError:
        raise RuntimeError("请安装 opencv-contrib-python: pip install opencv-contrib-python")

    edges_rgb = cv2.cvtColor(edges_thin, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


# 1. 加载模型（确保你已登录 huggingface-cli 或设置了 token）
model_id = 'runwayml/stable-diffusion-v1-5'
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_auth_token=True,  # 确保你已登录 huggingface-cli 或设置了 token
    generator=torch.manual_seed(0)  # 设置随机种子
    image=get_canny_thin("/root/autodl-tmp/DLproject/code/output.png")
).to("cuda")


# 2. 设置 prompt
prompt = "Kremlin"

# 3. 生成图像
with torch.autocast("cuda"):
    result = pipe(prompt, guidance_scale=7.5, num_inference_steps=30)

# 4. 保存图像
image: Image.Image = result.images[0]
image.save("mouse_result.png")
print("Image saved to mouse_result.png")
