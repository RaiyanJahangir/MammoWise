#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

import config


# -------------------------
# Small helpers
# -------------------------
def remove_quotes(s: str) -> str:
    return s.replace('"', "")


def extract_number(s: str) -> Optional[int]:
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None


def make_img_id_from_filename(stem: str, prefix: str = "IMG", width: int = 3) -> str:
    """
    DMID filenames can vary. By default, we replicate your pattern:
      - extract first number from filename stem
      - format as IMG### (zero-padded)
    If DMID uses a different naming rule, change this function only.
    """
    n = extract_number(stem)
    if n is None:
        # Fallback: use raw stem uppercased
        return stem.upper()
    return f"{prefix}{str(n).zfill(width)}"


# -------------------------
# Parsing logic
# -------------------------
def parse_dmid_report(report: str, img_id: str) -> Dict[str, Any]:
    """
    Parses a DMID report text into a JSON dict.

    Expected fields (robust to minor label variations):
      - Breast Composition / Breast-Composition
      - BIRADS / BI-RADS / BI-RADS Category
      - Findings (until end)
    """
    report = remove_quotes(report)

    # Breast composition (try multiple label variants)
    bc_match = re.search(r"Breast[-\s]?Composition:\s*(.*)", report, re.IGNORECASE)
    breast_composition = bc_match.group(1).strip() if bc_match else ""

    # BIRADS (allow BI-RADS variants)
    birads_match = re.search(r"\bBI[-\s]?RADS\b\s*[:\-]?\s*(\d+)", report, re.IGNORECASE)
    birads = int(birads_match.group(1)) if birads_match else 1

    # Findings: capture from "Findings:" to end (or to common footer markers if present)
    findings_match = re.search(
        r"Findings:\s*(.*?)(?=\n\*\*\*REPORT ENDS\*\*\*|$)",
        report,
        re.DOTALL | re.IGNORECASE,
    )
    findings = findings_match.group(1).strip() if findings_match else ""

    return {
        "IMG_ID": img_id,
        "Breast_Density": breast_composition,
        "BIRADS": birads,
        "Findings": findings,
        "dataset": "dmid",
    }


# -------------------------
# Main conversion
# -------------------------
def process_reports(txt_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted([p for p in txt_dir.glob("*.txt") if p.is_file()])
    if not txt_files:
        print(f"No .txt files found in: {txt_dir}")
        return

    written = 0
    for p in txt_files:
        report_content = p.read_text(encoding="utf-8", errors="ignore")

        img_id = make_img_id_from_filename(p.stem)  # default IMG### logic
        payload = parse_dmid_report(report_content, img_id)

        out_path = out_dir / f"{img_id}.json"
        out_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
        written += 1

    print(f"Converted {written} reports to JSON in: {out_dir}")


def main():
    # From config.py
    txt_dir = config.DMID_TXT_REPORTS_DIR
    out_dir = config.DMID_JSON_OUTPUT_DIR

    if not txt_dir.exists():
        raise FileNotFoundError(f"DMID txt report directory not found: {txt_dir}")

    # (Optional) ensure images dir exists; not required for conversion,
    # but helpful to catch path mistakes early.
    if hasattr(config, "DMID_IMAGES_DIR") and not config.DMID_IMAGES_DIR.exists():
        print(f"Warning: DMID images directory not found (continuing anyway): {config.DMID_IMAGES_DIR}")

    process_reports(txt_dir, out_dir)
    print("Done! DMID txt files are converted to JSON.")


if __name__ == "__main__":
    main()
