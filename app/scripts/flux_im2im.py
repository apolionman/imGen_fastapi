import torch, uuid, os, gc
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import random
from PIL import Image
from supabase import create_client, Client

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def generate_im2im_task(prompt: str,
                        user_uuid: str,
                        task_id: str,
                        image_path: str, 
                        seed: int = None
                        ) -> dict:
    from huggingface_hub import login
    login(token=os.environ["HUGGINGFACE_TOKEN"])



    print("Start loading pipeline!")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.float16,
        device_map="balanced"
    )

    print(f"image here: {image_path}")
    if seed is None:
        seed = random.randint(0, 999999)

    generator = torch.manual_seed(seed)
    input_image = Image.open(image_path).convert("RGB").resize((1024, 1024))
    # input_image = load_image(image_path).convert("RGB").resize((1024, 1024))

    print(f"🧐 Input image loaded: {image_path}")
    print(f"🔍 Prompt: '{prompt}', Seed: {seed}")

    try:
        result = pipe(
            prompt=prompt,
            image=input_image,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=generator
        )

        print(f"📦 Result keys: {result.keys()}")
        image = result.images[0]
        print(f"🖼️ Image type: {type(image)}")

        if image is None:
            print("⚠️ Image is None")
            return {"status": "error", "error": "Generated image is None"}

        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex[:8]}.png"
        genI_name = os.path.join(output_dir, filename)
        
        backend_url = os.getenv("BACKEND_URL")
        image_url = f"{backend_url}/generated/images/{filename}"

        try:
            image.save(genI_name)
            print(f"✅ Image saved to {genI_name} (Seed: {seed})")
            try:
                supabase.table("thumbnail_tasks").upsert({
                    "task_id": task_id,
                    "user_id": user_uuid,
                    "image_url": image_url
                }).on_conflict(["task_id", "user_uuid"]).execute()
            except Exception as e:
                print(f"⚠️ Supabase insert error: {e}")
        except Exception as e:
            print(f"❌ Failed to save image: {e}")
            return {"status": "error", "error": f"Image save failed: {e}"}

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

