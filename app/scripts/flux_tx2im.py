import torch, uuid, os, gc, random
from diffusers import FluxPipeline
from supabase import create_client, Client

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def generate_image_task(prompt: str,
                        user_uuid: str,
                        task_id: str,
                        seed: int = None) -> dict:
    from huggingface_hub import login
    login(token=os.environ["HUGGINGFACE_TOKEN"])

    print("Start loading pipeline!")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16,
        device_map="balanced"
    )
    print("Pipe loaded, starting image generation")

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
        filename = f"{uuid.uuid4().hex[:8]}.png"
        image_path = os.path.join(output_dir, filename)
        image.save(image_path)

        backend_url = os.getenv("BACKEND_URL")
        image_url = f"{backend_url}/generated/images/{filename}"

        # Save to Supabase
        try:
            supabase.table("thumbnail_tasks").upsert({
                "task_id": task_id,
                "user_id": user_uuid,
                "image_url": image_url
            }).on_conflict(["task_id", "user_id"]).execute()
        except Exception as e:
            print(f"⚠️ Supabase insert error: {e}")

        print(f"✅ Image saved to {image_path} (Seed: {seed})")

        return {
            "status": "success",
            "image_path": image_path,
            "filename": filename,
            "seed": seed,
            "user_id": user_uuid,
            "task_id": task_id
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
