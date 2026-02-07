import os, json, re
import torch
import torch.multiprocessing as mp
from PIL import Image

from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

image_directory = "..."
output_dir      = "..."
model_dir       = "..."   # your LoRA adapter repo/dir
base_model_id   = "google/medgemma-4b-it"

num_images = -1  # -1 = all

prompt = """
                You are a board certified breast radiologist with lots of experience
                in interpreting screening mammograms. You are meticulous,
                and always provide clear, concise, and clinically actionable reports.

                You are provided with a mammogram image. The image has all 4 breast views shown together.
                The upper two views are the Cranio-Caudal (CC) views of each breast, right and left,
                and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

                At first, you should identify the breast density using the ACR classification,
                which includes:
                - ACR A: Almost entirely fatty
                - ACR B: Scattered fibroglandular densities
                - ACR C: Heterogeneously dense
                - ACR D: Extremely dense

                Then, you should summarize any findings from the images. Mention in which view the findings
                are present. Findings include
                Mass, Suspicious Calcification, Architectural Distortion, Asymmetry, Focal Asymmetry, Global Asymmetry,
                Suspicious Lymph Nodes, Nipple Retraction, Skin Retraction, Skin Thickening.
                There may be multiple findings in a single image. For normal and healthy breasts, "Healthy Breast. No Findings"

                Finally, assign a BIRADS category based on the findings:
                - BIRADS 1: Negative (no abnormalities)
                - BIRADS 2: Benign (no suspicion of cancer)
                - BIRADS 3: Probably benign (short-term follow-up recommended)
                - BIRADS 4: Suspicious abnormality (biopsy needed)
                - BIRADS 5: Highly suggestive of malignancy (high probability of cancer)

                Also assign the healthy status of the breast.

                Here is the JSON format you should generate strictly for your response:
                {
                    "breast_density": "<ACR A|B|C|D> followed by a brief description of the density",
                    "findings": "<Summary of any abnormalities as described above>",
                    "birads": "<0|1|2|3|4|5|6> followed by a brief description of the BIRADS category>",
                    "suspicion":"healthy|benign|suspicious"
                }


            """

def clean_json(text: str) -> str:
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return text
    s = text[start:end+1]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

def load_quantized_peft_model_on_gpu(gpu_id: int):
    torch.cuda.set_device(gpu_id)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # keep fp16 compute
    )

    # Load base model quantized, pinned to this GPU only
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map={"": gpu_id},     # force everything on that GPU (works because it's 4-bit)
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True)
    processor.tokenizer.padding_side = "right"

    return model, processor

@torch.inference_mode()
def generate_one(model, processor, image: Image.Image, prompt: str, max_new_tokens=512):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # IMPORTANT: do NOT truncate here
    inputs = processor(
        text=[text],
        images=[[image]],
        return_tensors="pt",
        padding=True,
        truncation=False,     # <-- KEY FIX
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.cuda.amp.autocast(dtype=torch.float16):
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    decoded = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded


def worker(gpu_id: int, files: list[str]):
    model, processor = load_quantized_peft_model_on_gpu(gpu_id)

    processed = 0
    for fname in files:
        img_path = os.path.join(image_directory, fname)
        base     = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{base}.json")

        if os.path.exists(out_path):
            continue

        image = Image.open(img_path).convert("RGB")

        raw = generate_one(model, processor, image, prompt, max_new_tokens=512)

        cleaned = clean_json(raw)
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            result = {"raw": raw.strip()}

        annotated = {"image_id": img_path, **result}
        with open(out_path, "w") as f:
            json.dump(annotated, f, indent=2)

        processed += 1
        if processed % 10 == 0:
            print(f"[GPU {gpu_id}] ✅ Processed {processed} reports")

    print(f"[GPU {gpu_id}] ✅ Done. Total processed: {processed}")

def main():
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted([
        fn for fn in os.listdir(image_directory)
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])
    if num_images != -1:
        all_files = all_files[:num_images]

    files_gpu0 = [f for i, f in enumerate(all_files) if i % 2 == 0]
    files_gpu1 = [f for i, f in enumerate(all_files) if i % 2 == 1]

    mp.set_start_method("spawn", force=True)
    p0 = mp.Process(target=worker, args=(0, files_gpu0))
    p1 = mp.Process(target=worker, args=(1, files_gpu1))
    p0.start(); p1.start()
    p0.join();  p1.join()

    print("✅ All done (both GPUs).")

if __name__ == "__main__":
    main()

