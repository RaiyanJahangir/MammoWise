from datasets import Dataset, Features, Value
from pathlib import Path
from PIL import Image
import json, re

image_dir = Path("...")
report_dir = Path("...")
load_images = False
progress_every = 10

def _extract_asymmetry_from_findings(findings):
    """
    Return 1 if the 'findings' text mentions asymmetry/asymmetries/asymmetric/etc.,
    else 0. Uses regex on the findings string.
    """
    if findings is None:
        return 0

    s = str(findings).lower()
    # Match 'asymmetry', 'asymmetries', 'asymmetric', 'asymmetrically', etc.
    if re.search(r"\basymmet\w*\b", s):
        return 1
    return 0

png_files = sorted(image_dir.glob("*.png"))
samples = []
bad = 0

for idx, p in enumerate(png_files, 1):
    jp = report_dir / f"{p.stem}.json"
    if not jp.exists():
        continue
    try:
        with open(jp, "r") as fh:
            d = json.load(fh)
    except json.JSONDecodeError:
        continue

    findings = d.get("findings", "")
    aval = _extract_asymmetry_from_findings(findings)

    # aval is guaranteed 0 or 1 here, but keep pattern consistent
    if aval is None:
        bad += 1
        continue

    samples.append({
        "image": str(p) if not load_images else Image.open(p).convert("RGB"),
        "asymmetry": int(aval),  # guaranteed 0 or 1
    })

    if progress_every and idx % progress_every == 0:
        print(f"[progress] {idx}/{len(png_files)}  kept={len(samples)}  bad_asymmetry={bad}")

features = Features({
    "image": Value("string"),  # use datasets.Image() if you want decoding at load time
    "asymmetry": Value("int64"),
})

ds = Dataset.from_list(samples, features=features)
split = ds.train_test_split(test_size=0.1, seed=42)
split.save_to_disk("...")
