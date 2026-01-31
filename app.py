from flask import Flask, request, send_file, jsonify
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import io

app = Flask(__name__)

# --- Load model once ---
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
)
model.eval()

@app.route("/segment", methods=["POST"])
def segment():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = Image.open(request.files["image"].stream).convert("RGB")
    image_np = np.array(image)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    segmentation = outputs.logits.argmax(dim=1)[0]
    sidewalk_mask = (segmentation == 1).cpu().numpy().astype(np.uint8)

    # Resize mask to original image size
    mask_pil = Image.fromarray(sidewalk_mask * 255)
    mask_resized = mask_pil.resize(image.size, Image.NEAREST)
    sidewalk_mask_resized = np.array(mask_resized) > 0

    # Overlay
    overlay_color = np.array([0, 255, 0], dtype=np.uint8)
    alpha = 0.5

    overlay = np.zeros_like(image_np, dtype=np.uint8)
    overlay[sidewalk_mask_resized] = overlay_color

    blended = image_np.copy()
    blended[sidewalk_mask_resized] = (
        (1 - alpha) * blended[sidewalk_mask_resized]
        + alpha * overlay[sidewalk_mask_resized]
    ).astype(np.uint8)

    result = Image.fromarray(blended)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

# 🔥 THIS IS THE IMPORTANT PART 🔥
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
