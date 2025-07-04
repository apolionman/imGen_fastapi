import torch, uuid, os, gc
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import random

def generate_im2im_task(prompt: str, image_path: str, seed: int = None) -> dict:
    from huggingface_hub import login
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    
    print("Start loading pipeline!")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype="torch.float16",
        device_map="balanced"
    )
    print("Pipe loaded starting generating image")
    try:
        print(f"Went inside try")
        if seed is None:
            seed = random.randint(0, 999999)
            print(f"seed#{seed}")
        generator = torch.manual_seed(seed)
        input_image = load_image(image_path)
        print(f"🔍 Starting inference with prompt: '{prompt}', seed: {seed}")
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            image=input_image,
            guidance_scale=2.5,
            generator=generator
        ).images[0]
        if not image or not image.images or image.images[0] is None:
            raise ValueError("❌ Image generation returned empty result.")
        
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
