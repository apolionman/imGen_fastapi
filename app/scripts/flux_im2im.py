import torch, uuid, os
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import random

# Load pipeline
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype="torch.float16",
    device_map="balanced"
)

def generate_im2im_task(prompt: str, image_path: str, seed: int = None) -> dict:
    print(f"🧐 Check Prompt: {prompt} and image: {image_path}")
    try:
        if seed is None:
            seed = random.randint(0, 999999)
        generator = torch.manual_seed(seed)
        input_image = load_image(image_path)
        image = pipe(
            image=input_image,
            prompt=prompt,
            guidance_scale=2.5,
            generator=generator
        ).images[0]

        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        genI_name = os.path.join(output_dir, f"{str(uuid.uuid4())[:8]}.png")
        image.save(genI_name)
        print(f"✅ Image saved to {genI_name} (Seed: {seed})")

        return {
            "status": "success",
            "image_path": genI_name,
            "seed": seed  # Return seed info optionally
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
