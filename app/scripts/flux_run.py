import torch, uuid, os, sys, argparse
from diffusers import FluxPipeline

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, required=True)
args = parser.parse_args()

# Load pipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)

# Generate image
image = pipe(
    args.prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.manual_seed(0)
).images[0]

# Save image
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
genI_name = os.path.join(output_dir, f"{str(uuid.uuid4())[:8]}.png")
image.save(genI_name)
print(f"✅ Image saved to {genI_name}")
