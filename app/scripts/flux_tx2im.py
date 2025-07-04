import torch, uuid, os, gc
from diffusers import FluxPipeline
import random

def generate_image_task(prompt: str, seed: int = None) -> dict:
    from huggingface_hub import login
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    # Load pipeline
    print("Start loading pipeline!")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16,
        device_map="balanced"
    )
    print("Pipe loaded starting generating image")
    try:
        if seed is None:
            seed = random.randint(0, 999999)
        generator = torch.manual_seed(seed)

        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
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
