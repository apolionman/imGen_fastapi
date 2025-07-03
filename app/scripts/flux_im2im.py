import torch, uuid, os, gc
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import random

# Load pipeline
def load_pipeline():
    return FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.float16,
        device_map="balanced"
    )

def generate_im2im_task(prompt: str, image_path: str, seed: int = None) -> dict:
    try:
        if seed is None:
            seed = random.randint(0, 999999)

        generator = torch.manual_seed(seed)
        
        # Load image
        image = load_image(image_path)

        # Load pipeline
        pipe = load_pipeline()

        # Generate image
        result = pipe(prompt=prompt, image=image, generator=generator).images[0]

        # Save result
        image_id = str(uuid.uuid4())
        output_path = os.path.join("./output", f"{image_id}.png")
        result.save(output_path)

        # Clean up
        del pipe
        torch.cuda.empty_cache()
        gc.collect()

        return {"image_id": image_id, "image_path": output_path}

    except Exception as e:
        print(f"❌ Error in generate_im2im_task: {e}")
        return {"error": str(e)}
