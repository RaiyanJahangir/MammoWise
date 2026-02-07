from transformers import pipeline
from PIL import Image
import torch
import os
import json
import re

image_directory = "..."
report_directory = "..."
output_dir = "..."

model_dir = "..."
model_id = "google/medgemma-4b-it"

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

# ==== PROMPT (ask for a single digit 0–1) =====================================
prompt = """
You are a certified breast radiologist with lots of experience
in interpreting screening mammograms. You are meticulous,
and always provide clear, concise, and clinically actionable reports.

You are provided with a mammogram image. The image has all 4 breast views shown together.
The upper two views are the Cranio-Caudal (CC) views of each breast, right and left, shown as a symmetric layout,
and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

Calcifications are small, bright white specks or spots, ranging from large, round macrocalcifications (usually benign) to smaller,
finer microcalcifications. The pattern and shape are key indicators: benign calcifications often have smooth,
round, or coarse "popcorn-like" shapes, while those with irregular, linear, or clustered shapes are more suspicious
and may warrant further evaluation, as they can be a sign of breast cancer.

Print a single integer (0/1) that indicates the absence or presence of calcification.
Here is the JSON format you should follow for your response:
{
    "calcification": "<0|1>"
}
""".strip()


# ==== Helpers ==================================================================
def get_generated_text(output_obj) -> str:
    """
    Works across possible pipeline return shapes and extracts the raw text.
    """
    # Common Med-Gemma chat pipeline shape:
    try:
        # often: [{"generated_text": [{"role": "...", "content": "..."} , ...]}]
        gen = output_obj[0]["generated_text"]
        if isinstance(gen, list):
            # take last content chunk
            last = gen[-1]
            if isinstance(last, dict) and "content" in last:
                return str(last["content"])
            # if list of strings
            return str(gen[-1])
        # sometimes it's a plain string
        return str(gen)
    except Exception:
        pass

    # Fallbacks some pipelines may return "generated_text" directly as string
    if isinstance(output_obj, list) and "generated_text" in output_obj[0]:
        return str(output_obj[0]["generated_text"])

    # Last resort: convert whole object
    return str(output_obj)


def extract_single_digit_0_1(text: str) -> int:
    """
    Extract the first standalone digit 0–1 from text.
    Ensures it's not part of a multi-digit number.
    """
    m = re.search(r"\b([01])\b", text)
    if m:
        return int(m.group(1))
    # If nothing found, default to 1 (or raise). You can choose a different policy.
    return 1


# ==== Inference loop ============================================================
processed_count = 0

for fname in files_to_process:
    img_path = os.path.join(image_directory, fname)
    image = Image.open(img_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ]
        }
    ]

    output = pipe(text=messages, max_new_tokens=8)  # small budget; we only want 1 digit
    raw = get_generated_text(output)

    # Strictly extract the single digit 0–1
    cal_int = extract_single_digit_0_1(raw)

    # Save ONLY {"calcification": <int>}
    result = {"calcification": cal_int}

    base = os.path.splitext(fname)[0]
    out_path = os.path.join(output_dir, f"{base}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    processed_count += 1
    if processed_count % 10 == 0:
        print(f"✅ Processed {processed_count} reports")

print(f"✅ Total reports processed: {processed_count}")
