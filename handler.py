import runpod
import torch
import base64
import io
from PIL import Image
from rembg import remove, new_session
from tsr.system import TSR

print("Initializing Rembg session...")
rembg_session = new_session("u2net")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading SF3D model into VRAM...")
model = TSR.from_pretrained(
    "stabilityai/stable-fast-3d",
    config_name="config.yaml",
    weight_name="model.ckpt"
)
model.renderer.to(device)
model.to(device)

def handler(job):
    try:
        job_input = job["input"]
        image_base64 = job_input.get("image")
        
        if not image_base64:
            return {"error": "Missing image input"}

        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_bytes = base64.b64decode(image_base64)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        clean_image = remove(input_image, session=rembg_session)

        with torch.inference_mode():
            scene_codes = model(clean_image, device=device)
            mesh = model.extract_mesh(scene_codes, resolution=256)[0]

        glb_io = io.BytesIO()
        mesh.export(glb_io, file_type="glb")
        glb_bytes = glb_io.getvalue()

        glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")

        return {
            "status": "success",
            "model_mesh": f"data:model/gltf-binary;base64,{glb_base64}"
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
