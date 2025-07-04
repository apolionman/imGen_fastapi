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
        torch_dtype=torch.float16,
        device_map="balanced"
    )
    try:
        print(f"Went inside try")
        if seed is None:
            seed = random.randint(0, 999999)

        generator = torch.manual_seed(seed)
        input_image = load_image(image_path).convert("RGB").resize((1024, 1024))
        input_image.save("debug_input.png")

        print(f"🧐 Input image loaded: {image_path}")
        print(f"🔍 Prompt: '{prompt}', Seed: {seed}")

        result = pipe(
            prompt,
            height=1024,
            width=1024,
            image=input_image,
            guidance_scale=2.5,
            generator=generator
        )

        print(f"📦 Result keys: {result.keys()}")
        image = result.images[0]
        print(f"🖼️ Image type: {type(image)}")

        if image is None:
            print("⚠️ Image is None")
            return {"status": "error", "error": "Generated image is None"}

        output_dir = "/tmp/output"
        os.makedirs(output_dir, exist_ok=True)
        genI_name = os.path.join(output_dir, f"{str(uuid.uuid4())[:8]}.png")

        try:
            image.save(genI_name)
            print(f"✅ Image saved to {genI_name} (Seed: {seed})")
        except Exception as e:
            print(f"❌ Failed to save image: {e}")
            return {"status": "error", "error": f"Image save failed: {e}"}

        return {
            "status": "success",
            "image_path": genI_name,
            "seed": seed
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

