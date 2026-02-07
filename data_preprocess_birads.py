from datasets import Dataset, Features, Value
from pathlib import Path
from PIL import Image
import json, re

image_dir = Path("...")
report_dir = Path("...")
load_images = False
progress_every = 10

def _extract_birads(v):
    if v is None: return None
    if isinstance(v, (int, float)):
        v = int(v)
        return v if 0 <= v <= 6 else None
    if isinstance(v, str):
        m = re.search(r"\b([0-6])\b", v)
        return int(m.group(1)) if m else None
    return None

png_files = sorted(image_dir.glob("*.png"))
samples = []
bad = 0

for idx, p in enumerate(png_files, 1):
    jp = report_dir / f"{p.stem}.json"
    if not jp.exists():
        continue
    try:
        d = json.load(open(jp))
    except json.JSONDecodeError:
        continue

    b = _extract_birads(d.get("birads"))
    if b is None:
        bad += 1
        continue

    samples.append({
        "image": str(p) if not load_images else Image.open(p).convert("RGB"),
        "birads": int(b),  # single, consistent int column
    })
    if progress_every and idx % progress_every == 0:
        print(f"[progress] {idx}/{len(png_files)}  kept={len(samples)}  bad_birads={bad}")

features = Features({
    "image": Value("string"),   # or use datasets.Image() if you want to decode later
    "birads": Value("int64"),
})

ds = Dataset.from_list(samples, features=features)
split = ds.train_test_split(test_size=0.1, seed=42)
split.save_to_disk("...")
