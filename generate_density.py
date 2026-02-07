from transformers import pipeline
from PIL import Image
import torch
import os
import json
import re

image_directory="..."
report_directory="..."
output_dir     = "..."
model_dir = "..."  # use your finetuned/desired model
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

# ==== PROMPT: demand ONLY A/B/C/D ===============================================
prompt = """
                You are a certified breast radiologist with lots of experience
                in interpreting screening mammograms. You are meticulous,
                and always provide clear, concise, and clinically actionable reports.

                You are provided with a mammogram image. The image has all 4 breast views shown together.
                The upper two views are the Cranio-Caudal (CC) views of each breast, right and left, shown as a symmetric layout,
                and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

                Your task is to analyze the mammogram images and determine the density of the breast tissue. 
                Darker color indicates fatty tissue while lighter color indicates denser tissue.
                It is easier to see the abnormalities in less dense breasts and gets more difficult as the density increases.
                
                Classify the breast density using the ACR classification, which includes:
                - ACR A: Almost entirely fatty
                - ACR B: Scattered fibroglandular densities
                - ACR C: Heterogeneously dense
                - ACR D: Extremely dense


                 Here is the JSON format you should follow for your response:
                {
                    "breast_density": "<ACR A|B|C|D> followed by a brief description of the density"
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

def extract_density_letter(text: str) -> str:
    """
    Return first standalone A/B/C/D (case-insensitive).
    Ensures it's not part of a longer token.
    """
    m = re.search(r'(?<![A-Za-z])([ABCDabcd])(?![A-Za-z])', text)
    if m:
        return m.group(1).upper()
    # fallback policy if nothing found: choose "A" (or raise)
    return "A"

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

    # tiny generation budget—we only want 1 character
    output = pipe(text=messages, max_new_tokens=2)
    raw    = get_generated_text(output)

    density = extract_density_letter(raw)

    # Save ONLY {"density": "<A|B|C|D>"}
    result = {"breast_density": density}

    base     = os.path.splitext(fname)[0]
    out_path = os.path.join(output_dir, f"{base}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    processed_count += 1
    if processed_count % 10 == 0:
        print(f"✅ Processed {processed_count} images")

print(f"✅ Total reports processed: {processed_count}")
