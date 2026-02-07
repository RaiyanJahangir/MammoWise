from transformers import pipeline
from PIL import Image
import torch
import os
import json
import re

image_directory="..."
report_directory="..."
output_dir     = "..."
model_dir = "..."  # your finetuned model
model_id="google/medgemma-4b-it"

pipe = pipeline(
    "image-text-to-text",
    model=model_dir,
    torch_dtype=torch.bfloat16,
    device="cuda",
    use_auth_token=True
)

num_images = -1
os.makedirs(output_dir, exist_ok=True)

all_files = sorted([
    fn for fn in os.listdir(image_directory)
    if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
])
files_to_process = all_files if num_images == -1 else all_files[:num_images]

# ==== PROMPT: demand ONLY Healthy/Benign/Malignant ===============================
prompt = """
                You are a certified breast radiologist with lots of experience
                in interpreting screening mammograms. You are meticulous,
                and always provide clear, concise, and clinically actionable reports.

                You are provided with a mammogram image. The image has all 4 breast views shown together.
                The upper two views are the Cranio-Caudal (CC) views of each breast, right and left, shown as a symmetric layout,
                and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

                Your task is to analyze the mammogram images and find out any abnormalities. 
                If any abnormalities are found, determine whether they are malignant (cancerous) or benign (non-cancerous).
                Malignant abnormalities often have irregular shapes, spiculated margins, and high density, 
                while benign abnormalities tend to have smooth, well-defined borders and lower density.
                Consider factors such as the shape, margins, density, and any associated features like calcifications or 
                architectural distortion. Otherwise, if no abnormalities are found, classify it as healthy.
                
                Classify the suspicion in the following way:
                - Healthy: No abnormalities detected
                - Benign: Abnormalities detected but are non-cancerous
                - Suspicious: Abnormalities detected that are cancerous or highly suspicious


                Here is the JSON format you should follow for your response:
                {
                    "suspicion": "<Healthy|Benign|Suspicious>"
                }
""".strip()

# ==== Helpers ===================================================================
def get_generated_text(output_obj) -> str:
    """Extract raw text from potential pipeline return shapes."""
    try:
        gen = output_obj[0]["generated_text"]
        if isinstance(gen, list):
            last = gen[-1]
            if isinstance(last, dict) and "content" in last:
                return str(last["content"])
            return str(gen[-1])
        return str(gen)
    except Exception:
        pass
    if isinstance(output_obj, list) and "generated_text" in output_obj[0]:
        return str(output_obj[0]["generated_text"])
    return str(output_obj)

def extract_malignancy(text: str) -> str:
    """
    Return 'healthy' | 'benign' | 'malignant' from model text.
    Looks for standalone words (case-insensitive). If none found, returns 'unknown'.
    """
    if not isinstance(text, str):
        return "unknown"
    m = re.search(r'\b(healthy|benign|malignant)\b', text, flags=re.IGNORECASE)
    return m.group(1).lower() if m else "unknown"

# ==== Inference loop ============================================================
processed_count = 0
for fname in files_to_process:
    img_path = os.path.join(image_directory, fname)
    image    = Image.open(img_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ]
        }
    ]

    # Tiny generation budget—we only want one word
    output = pipe(text=messages, max_new_tokens=3)
    raw    = get_generated_text(output)

    malignancy = extract_malignancy(raw)

    # Save ONLY {"malignancy": "<healthy|benign|malignant|unknown>"}
    result = {"malignancy": malignancy}

    base     = os.path.splitext(fname)[0]
    out_path = os.path.join(output_dir, f"{base}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    processed_count += 1
    if processed_count % 10 == 0:
        print(f"✅ Processed {processed_count} images")

print(f"✅ Total reports processed: {processed_count}")
