import torch, uuid, os, gc, shutil
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import random
from PIL import Image
from supabase import create_client, Client
import mimetypes

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def generate_im2im_task(prompt: str,
                        image_path: str,
                        user_uuid: str,
                        task_id: str,
                        ) -> dict:
    from huggingface_hub import login
    login(token=os.environ["HUGGINGFACE_TOKEN"])

    try:
        print("Start loading pipeline!")
        pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.float16,
            device_map="balanced"
        )

        input_image = Image.open(image_path).convert("RGB").resize((1024, 1024))

        result = pipe(
            prompt=prompt,
            image=input_image,
            height=768,
            width=1360,
            guidance_scale=2.5,
            num_inference_steps=50,
            max_sequence_length=512,
        )

        image = result.images[0]
        if image is None:
            print("⚠️ Image is None")
            return {"status": "error", "error": "Generated image is None"}

        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex[:8]}.png"
        genI_name = os.path.join(output_dir, filename)
        image.save(filename)

        file_path = f"thumbnails/{filename}"
        content_type, _ = mimetypes.guess_type(filename)
        if content_type is None:
            content_type = "application/octet-stream"

        with open(genI_name, "rb") as f:
            upload_response = supabase.storage.from_("thumbnails").upload(
                path=file_path,
                file=f,
                file_options={"content-type": content_type}
            )
        if upload_response is None or getattr(upload_response, 'error', None):
            print(f"⚠️ Upload error: {getattr(upload_response, 'error', 'Unknown error')}")
            return {"status": "error", "error": "Upload failed"}

        image_url = f'https://garfxtaapwmphxeqfrnd.supabase.co/storage/v1/object/public/thumbnails/thumbnails/{filename}'

        try:
            supabase.table("thumbnail_tasks").update({
                "im2im_image_url": image_url
            }).match({
                "task_id": task_id,
                "user_id": user_uuid
            }).execute()
        except Exception as e:
            print(f"⚠️ Supabase insert error: {e}")
            return {"status": "error", "error": "DB insert failed"}

        return {
            "status": "success",
            "image_url": image_url,
            "filename": filename,
            "user_id": user_uuid,
            "task_id": task_id
        }

    finally:
        # ✅ Clean up the input image
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"🧹 Deleted temp input file: {image_path}")
            except Exception as cleanup_err:
                print(f"⚠️ Failed to delete temp file {image_path}: {cleanup_err}")
