from diffusers import StableDiffusionPipeline
import torch

model_path = "./stable_diffusion/sd-v1-4-full-ema-diffusers/"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")


def generate_image(input_text):
    prompt = input_text
    image = pipe(prompt).images[0]
    # image.save("output.png")
    return image