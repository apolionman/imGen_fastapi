import torch, uuid, os, gc, random
from diffusers import FluxPipeline
from supabase import create_client, Client
from io import BytesIO
import mimetypes

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
        torch_dtype=torch.float32,
        device_map="balanced"
    )
    print("Pipe loaded, starting image generation")

    # try:
    # if seed is None:
    #     seed = random.randint(0, 999999)
    # generator = torch.manual_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(42)
    result = pipe(
        prompt,
        height=768,
        width=1360,
        guidance_scale=2.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=generator
    ).images[0]
    print("Image generation completed")
    image = result.images[0]

    if image is None:
        print("⚠️ Image is None")
        return {"status": "error", "error": "Generated image is None"}

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:8]}.png"
    genI_name = os.path.join(output_dir, filename)
    image.save(genI_name)
    
    print('File is here ===> ', genI_name)

    file_path = f"thumbnails/{filename}"
    print('Output path is here ==>', file_path)
    
    content_type, _ = mimetypes.guess_type(filename)

    # Fallback if not recognized
    if content_type is None:
        content_type = "application/octet-stream"

    with open(genI_name, "rb") as f:
        upload_response = supabase.storage.from_("thumbnails").upload(
            path=file_path,
            file=f,
            file_options={"content-type": content_type}
        )
    print('Upload response ===> ', upload_response)
    if not upload_response or "error" in upload_response:
        print(f"⚠️ Upload error: {upload_response.get('error')}")
        return {"status": "error", "error": "Upload failed"}

    # Use returned relative path as image_url
    image_url = upload_response.get("path")

    # Save path to Supabase DB
    try:
        supabase.table("thumbnail_tasks").upsert({
            "task_id": task_id,
            "user_id": user_uuid,
            "image_url": image_url
        }).on_conflict(["task_id", "user_id"]).execute()
    except Exception as e:
        print(f"⚠️ Supabase insert error: {e}")
        return {"status": "error", "error": "DB insert failed"}

    print(f"✅ Image uploaded to Supabase at {image_url} (Seed: {seed})")

    return {
        "status": "success",
        "image_url": image_url,
        "filename": filename,
        "seed": seed,
        "user_id": user_uuid,
        "task_id": task_id
    }

    # except Exception as e:
    #     return {"status": "error", "error": str(e)}
