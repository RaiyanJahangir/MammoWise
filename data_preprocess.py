from datasets import Dataset
from pathlib import Path
from PIL import Image
import json

# === Configuration ===
image_dir = Path("...")   # ← replace with your image folder
report_dir = Path("...") # ← replace with your JSON folder
load_images = False  # Keep False to save paths, not PIL images
progress_every = 10 # print a progress line every N images (set to 0 to disable)

# === Collect data ===
png_files = sorted(image_dir.glob("*.png"))
total_images = len(png_files)
matched_pairs = 0
missing_reports = 0
json_errors = 0

samples = []
for idx, image_path in enumerate(png_files, start=1):
    base_name = image_path.stem
    json_path = report_dir / f"{base_name}.json"
    if not json_path.exists():
        missing_reports += 1
        # optional: print(f"[missing] {json_path}")
        continue

    try:
        with open(json_path, "r") as f:
            report_dict = json.load(f)
    except json.JSONDecodeError:
        json_errors += 1
        # optional: print(f"[json error] {json_path}")
        continue

    report_str = json.dumps(report_dict, indent=2)

    sample = {
        "image": str(image_path) if not load_images else Image.open(image_path).convert("RGB"),
        "label": report_str,
    }
    samples.append(sample)
    matched_pairs += 1

    if progress_every and idx % progress_every == 0:
        print(f"[progress] scanned {idx}/{total_images} | matched {matched_pairs} | missing {missing_reports} | json_errors {json_errors}")

# === Summary before saving ===
print("\n=== Summary ===")
print(f"Total PNG images scanned : {total_images}")
print(f"Matched image+JSON pairs : {matched_pairs}")
print(f"Missing JSON reports     : {missing_reports}")
print(f"JSON parse errors        : {json_errors}")

# === Convert and split ===
hf_dataset = Dataset.from_list(samples)
dataset_split = hf_dataset.train_test_split(test_size=0.1, seed=42)

# === Save to disk ===
save_path = Path("...")  # ← replace with a folder where dataset dict will be stored
dataset_split.save_to_disk(save_path)
# hf_dataset.save_to_disk(save_path)
print(f"\n✓ Dataset saved to: {save_path}")
print(f"  - train size: {len(dataset_split['train'])}")
print(f"  - test  size: {len(dataset_split['test'])}")