#!/usr/bin/env python3
import base64
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
import config


# ---------------------------
# Interactive helpers
# ---------------------------
def pick_option(title: str, options: List[str], default: Optional[str] = None) -> str:
    options = list(dict.fromkeys(options))
    if default and default not in options:
        options = [default] + options

    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        mark = " (default)" if default and opt == default else ""
        print(f"  {i}. {opt}{mark}")

    while True:
        raw = input("Select number (or type value): ").strip()
        if not raw and default:
            return default
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        if raw in options:
            return raw
        print("Invalid selection. Try again.")


# ---------------------------
# Ollama utilities
# ---------------------------
def ollama_has_model(model: str) -> bool:
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        # "NAME   ID   SIZE ..." header; first token of each line is model name
        return any(line.split()[0] == model for line in out.splitlines()[1:] if line.strip())
    except Exception:
        return False


def ollama_pull_model(model: str) -> None:
    print(f"Pulling model via Ollama: {model}")
    subprocess.check_call(["ollama", "pull", model])


def ensure_model_available(model: str) -> None:
    if not ollama_has_model(model):
        ollama_pull_model(model)


def image_to_base64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("utf-8")


def ollama_chat(model: str, prompt: str, image_paths: List[Path], temperature: float = 0.0) -> str:
    host = getattr(config, "OLLAMA_HOST", "http://localhost:11434")
    url = host.rstrip("/") + "/api/chat"

    images_b64 = [image_to_base64(p) for p in image_paths]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "options": {"temperature": temperature},
        "stream": False,
    }

    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "")


# ---------------------------
# Dataset utilities
# ---------------------------
def list_images(images_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def get_dataset_dirs(dataset: str) -> Tuple[Path, Optional[Path]]:
    d = dataset.lower().strip()
    if d == "vindr":
        return Path(config.VINDR_INFER_IMAGES_DIR), Path(config.VINDR_INFER_REPORTS_DIR)
    if d == "dmid":
        return Path(config.DMID_INFER_IMAGES_DIR), Path(config.DMID_INFER_REPORTS_DIR)
    raise ValueError("dataset must be one of: dmid, vindr")


# ---------------------------
# Few-shot example sampling (from dataset GT JSONs)
# ---------------------------
def birads_digit_to_suspicion(b: int) -> str:
    if b == 1:
        return "healthy"
    if b in (2, 3):
        return "benign"
    return "suspicious"


def normalize_gt_json_to_schema(gt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts different GT schemas into the unified output schema we want the model to learn.
    - VinDr merged JSON likely already has: breast_density, birads, findings, mass, calcification, asymmetry, suspicion
    - DMID JSON from your converter likely has: Breast_Composition, BIRADS, Findings
    """
    # density
    density = gt.get("breast_density", gt.get("Breast_Composition", ""))
    # birads
    b = gt.get("birads", gt.get("BIRADS", 1))
    try:
        b_int = int(b) if b is not None else 1
    except Exception:
        b_int = 1
    # findings
    findings = gt.get("findings", gt.get("Findings", ""))

    # flags (may be missing in DMID)
    mass = gt.get("mass", 0)
    calcification = gt.get("calcification", 0)
    asymmetry = gt.get("asymmetry", 0)

    # suspicion (may be missing)
    suspicion = gt.get("suspicion")
    if not suspicion:
        suspicion = birads_digit_to_suspicion(b_int)

    # Ensure ints for flags
    def to01(x):
        try:
            return 1 if int(x) == 1 else 0
        except Exception:
            return 0

    return {
        "breast_density": str(density),
        "birads": int(b_int),
        "findings": str(findings),
        "mass": to01(mass),
        "calcification": to01(calcification),
        "asymmetry": to01(asymmetry),
        "suspicion": str(suspicion),
    }


def sample_fewshot_examples(reports_dir: Path, k: int, seed: int) -> List[Dict[str, Any]]:
    """
    Randomly samples k JSON files from reports_dir and returns normalized examples.
    """
    if not reports_dir or not reports_dir.exists():
        return []

    files = sorted([p for p in reports_dir.rglob("*.json") if p.is_file()])
    if not files:
        return []

    rnd = random.Random(seed)
    if len(files) <= k:
        chosen = files
    else:
        chosen = rnd.sample(files, k)

    examples = []
    for p in chosen:
        try:
            gt = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        ex = normalize_gt_json_to_schema(gt)
        ex["_source_json"] = p.name
        examples.append(ex)

    return examples


def format_fewshot_block(examples: List[Dict[str, Any]]) -> str:
    """
    Creates a few-shot block the model can learn from.
    Since we can’t include the 5 example images (would be heavy), we provide
    their labeled outputs as formatting/phrase guidance.
    """
    if not examples:
        return ""

    lines = ["### 5 Reference Labeled Examples (same dataset)", ""]
    for i, ex in enumerate(examples, start=1):
        src = ex.get("_source_json", "")
        ex2 = {k: v for k, v in ex.items() if not k.startswith("_")}
        lines.append(f"Example {i} (source={src}):")
        lines.append(json.dumps(ex2, ensure_ascii=False))
        lines.append("")
    return "\n".join(lines)


# ---------------------------
# RAG for rag-fewshot (unchanged idea)
# ---------------------------
def try_build_rag_block(dataset: str, query_image: Path) -> str:
    """
    Same as before: uses OpenCLIP to embed the query image and retrieve top-k items from Chroma,
    then appends their metadata as example JSONs.
    """
    try:
        import chromadb
        import torch
        import open_clip
        from PIL import Image
    except Exception:
        return ""

    chroma_root = getattr(config, "CHROMA_ROOT", None)
    if chroma_root is None:
        return ""
    chroma_root = Path(chroma_root)
    if not chroma_root.exists():
        return ""

    col_name = config.CHROMA_VINDR_COLLECTION if dataset == "vindr" else config.CHROMA_DMID_COLLECTION
    top_k = int(getattr(config, "RAG_TOP_K", 3))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    model = model.to(device).eval()

    img = Image.open(query_image).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(x).float().cpu().numpy().flatten().tolist()

    client = chromadb.PersistentClient(path=str(chroma_root))
    col = client.get_or_create_collection(name=col_name)

    res = col.query(query_embeddings=[emb], n_results=top_k, include=["metadatas", "uris", "distances", "ids"])
    metadatas = (res.get("metadatas") or [[]])[0]
    uris = (res.get("uris") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    if not metadatas:
        return ""

    lines = ["### Retrieved Similar Examples (RAG)", ""]
    for i, (md, uri, rid, dist) in enumerate(zip(metadatas, uris, ids, dists), start=1):
        md = md or {}
        # Normalize to same schema-ish
        example = normalize_gt_json_to_schema(md)
        lines.append(f"Example {i} (id={rid}, distance={dist:.4f}): uri={uri}")
        lines.append(json.dumps(example, ensure_ascii=False))
        lines.append("")
    return "\n".join(lines)


# ---------------------------
# Prompt building with separate prompt variables
# ---------------------------
def build_prompt(prompt_type: str, dataset: str, image_path: Path, reports_dir: Optional[Path]) -> str:
    prompt_type = prompt_type.lower().strip()

    task = (
        "### Task\n"
        "Analyze the NEW mammogram image and output ONLY a valid JSON object with keys:\n"
        'breast_density (string), birads (int), findings (string), mass (0/1), calcification (0/1), asymmetry (0/1), suspicion ("healthy"/"benign"/"suspicious")\n'
    )

    if prompt_type == "zeroshot":
        return config.ZERO_SHOT_PROMPT.strip() + "\n\n" + task

    if prompt_type == "cot":
        return config.COT_PROMPT.strip() + "\n\n" + task

    if prompt_type == "fewshot":
        k = int(getattr(config, "FEWSHOT_K", 5))
        seed = int(getattr(config, "FEWSHOT_SEED", 42))
        examples = sample_fewshot_examples(reports_dir, k=k, seed=seed)
        fewshot_block = format_fewshot_block(examples)
        return config.FEW_SHOT_PROMPT_HEADER.strip() + "\n\n" + fewshot_block + "\n" + task

    if prompt_type == "rag-fewshot":
        rag_block = try_build_rag_block(dataset, image_path)
        return config.RAG_FEWSHOT_PROMPT.strip() + "\n\n" + rag_block + "\n" + task

    raise ValueError("prompt_type must be one of: zeroshot, fewshot, cot, rag-fewshot")


# ---------------------------
# Output path rule
# ---------------------------
def out_json_path(dataset: str, model: str, prompt_type: str, image_stem: str) -> Path:
    root = Path(__file__).resolve().parent
    out_dir = root / "results" / dataset / model.replace("/", "_") / prompt_type
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{image_stem}.json"


# ---------------------------
# Main
# ---------------------------
def main():
    model_options = getattr(config, "OLLAMA_MODEL_OPTIONS", [])
    default_model = getattr(config, "OLLAMA_DEFAULT_MODEL", model_options[0] if model_options else "llava")

    model_name = pick_option("Choose Ollama model:", model_options or [default_model], default=default_model)
    prompt_type = pick_option("Choose prompt type:", ["zeroshot", "fewshot", "cot", "rag-fewshot"], default="zeroshot")
    dataset = pick_option("Choose dataset:", ["dmid", "vindr"], default="vindr")

    ensure_model_available(model_name)

    images_dir, reports_dir = get_dataset_dirs(dataset)
    imgs = list_images(images_dir)
    if not imgs:
        raise FileNotFoundError(f"No images found under {images_dir}")

    print(f"\nDataset: {dataset} | Images: {len(imgs)}")
    print(f"Model: {model_name} | Prompt type: {prompt_type}")
    print(f"Ollama host: {getattr(config, 'OLLAMA_HOST', 'http://localhost:11434')}\n")

    processed, failed = 0, 0

    for img_path in imgs:
        try:
            prompt = build_prompt(prompt_type, dataset, img_path, reports_dir)

            response = ollama_chat(
                model=model_name,
                prompt=prompt,
                image_paths=[img_path],
                temperature=0.0,
            )

            parsed = None
            parse_error = None
            try:
                parsed = json.loads(response)
            except Exception as e:
                parse_error = str(e)

            out_path = out_json_path(dataset, model_name, prompt_type, img_path.stem)

            payload: Dict[str, Any] = {
                "dataset": dataset,
                "model": model_name,
                "prompt_type": prompt_type,
                "image_path": str(img_path),
                "raw_output": response,
                "json_parsed": parsed is not None,
            }
            if parsed is not None:
                payload["prediction"] = parsed
            else:
                payload["parse_error"] = parse_error

            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            processed += 1
            if processed % 20 == 0:
                print(f"Processed {processed}/{len(imgs)} ...")

        except Exception as e:
            failed += 1
            print(f"[ERROR] {img_path.name}: {e}")

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Failed:    {failed}")


if __name__ == "__main__":
    main()
