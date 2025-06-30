import torch, uuid, os
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)

async def generate_image_task(prompt: str):
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.manual_seed(0)
    ).images[0]
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True) 
    genI_name = os.path.join(output_dir, f"{str(uuid.uuid4())[:8]}.png")
    image.save(genI_name)
    return genI_name
