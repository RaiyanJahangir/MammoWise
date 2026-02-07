from datasets import Dataset, Features, Value
from pathlib import Path
from PIL import Image
import json, re

image_dir = Path("...")
report_dir = Path("...")
load_images = False
progress_every = 10

png_files = sorted(image_dir.glob("*.png"))
samples = []
missing_reports = 0
missing_findings = 0

for idx, p in enumerate(png_files, 1):
    jp = report_dir / f"{p.stem}.json"
    if not jp.exists():
        missing_reports += 1
        continue

    try:
        d = json.load(open(jp))
    except json.JSONDecodeError:
        continue

    findings = d.get("findings", "")
    if findings is None:
        findings = ""
        missing_findings += 1

    # Make sure it's a string
    if not isinstance(findings, str):
        findings = str(findings)

    # Look for the keyword "calcification" / "calcifications" (case-insensitive)
    # You can also loosen this to r"calcif" if you want to catch more variants.
    # has_calc = bool(re.search(r"\bcalcification(s)?\b", findings, flags=re.IGNORECASE))
    has_calc = bool(re.search(r"calcif", findings, flags=re.IGNORECASE))


    label = 1 if has_calc else 0

    samples.append({
        "image": str(p) if not load_images else Image.open(p).convert("RGB"),
        "calcification": int(label),
    })

    if progress_every and idx % progress_every == 0:
        print(
            f"[progress] {idx}/{len(png_files)}  "
            f"kept={len(samples)}  "
            f"missing_reports={missing_reports}  "
            f"missing_findings={missing_findings}"
        )

features = Features({
    "image": Value("string"),   # or use datasets.Image() if you want to decode later
    "calcification": Value("int64"),
})

ds = Dataset.from_list(samples, features=features)
split = ds.train_test_split(test_size=0.1, seed=42)
split.save_to_disk("...")
