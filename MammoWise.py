import base64
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config


# -----------------------
# Common helpers
# -----------------------
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


def list_images(images_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def out_json_path(dataset: str, model: str, prompt_type: str, image_stem: str) -> Path:
    root = Path(__file__).resolve().parent
    out_dir = root / "results" / dataset / model.replace("/", "_") / prompt_type
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{image_stem}.json"


def safe_json_load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_gt_json_to_schema(gt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert VinDr-like or DMID-like GT json into a unified schema.
    """
    density = gt.get("breast_density", gt.get("Breast_Composition", ""))
    findings = gt.get("findings", gt.get("Findings", ""))

    b = gt.get("birads", gt.get("BIRADS", 1))
    try:
        birads = int(b) if b is not None else 1
    except Exception:
        birads = 1

    def to01(x):
        try:
            return 1 if int(x) == 1 else 0
        except Exception:
            return 0

    suspicion = gt.get("suspicion", gt.get("malignancy", None))
    if not suspicion:
        # map birads -> suspicion
        if birads == 1:
            suspicion = "healthy"
        elif birads in (2, 3):
            suspicion = "benign"
        else:
            suspicion = "suspicious"

    return {
        "breast_density": str(density),
        "birads": int(birads),
        "findings": str(findings),
        "mass": to01(gt.get("mass", 0)),
        "calcification": to01(gt.get("calcification", 0)),
        "asymmetry": to01(gt.get("asymmetry", 0)),
        "suspicion": str(suspicion).lower(),
    }


def sample_fewshot_examples(reports_dir: Optional[Path], k: int, seed: int) -> List[Dict[str, Any]]:
    if not reports_dir or not reports_dir.exists():
        return []
    files = sorted([p for p in reports_dir.rglob("*.json") if p.is_file()])
    if not files:
        return []
    rnd = random.Random(seed)
    chosen = files if len(files) <= k else rnd.sample(files, k)

    exs = []
    for p in chosen:
        gt = safe_json_load(p)
        ex = normalize_gt_json_to_schema(gt)
        ex["_source_json"] = p.name
        exs.append(ex)
    return exs


def format_fewshot_block(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return ""
    lines = ["### Few-shot labeled examples (same dataset)\n"]
    for i, ex in enumerate(examples, start=1):
        src = ex.get("_source_json", "")
        ex2 = {k: v for k, v in ex.items() if not k.startswith("_")}
        lines.append(f"Example {i} (source={src}):")
        lines.append(json.dumps(ex2, ensure_ascii=False))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# -----------------------
# Dataset config
# -----------------------
def get_dataset_dirs(dataset: str) -> Tuple[Path, Optional[Path]]:
    d = dataset.lower().strip()
    if d == "vindr":
        return Path(config.VINDR_INFER_IMAGES_DIR), Path(getattr(config, "VINDR_INFER_REPORTS_DIR", "")) or None
    if d == "dmid":
        return Path(config.DMID_INFER_IMAGES_DIR), Path(getattr(config, "DMID_INFER_REPORTS_DIR", "")) or None
    raise ValueError("dataset must be one of: vindr, dmid")


# -----------------------
# Prompt building (non-finetune)
# -----------------------
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
        exs = sample_fewshot_examples(reports_dir, k=k, seed=seed)
        fewshot_block = format_fewshot_block(exs)
        return config.FEW_SHOT_PROMPT_HEADER.strip() + "\n\n" + fewshot_block + "\n" + task

    if prompt_type == "rag-fewshot":
        rag_block = try_build_rag_block(dataset, image_path)
        return config.RAG_FEWSHOT_PROMPT.strip() + "\n\n" + rag_block + "\n" + task

    raise ValueError("prompt_type must be one of: zeroshot, fewshot, cot, rag-fewshot, finetune")


# -----------------------
# RAG block via Chroma (optional)
# -----------------------
def try_build_rag_block(dataset: str, query_image: Path) -> str:
    """
    If chromadb/open_clip are installed and CHROMA_ROOT exists, retrieve top-k
    and append their metadata as example JSONs.
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

    lines = ["### Retrieved Similar Examples (RAG)\n"]
    for i, (md, uri, rid, dist) in enumerate(zip(metadatas, uris, ids, dists), start=1):
        md = md or {}
        ex = normalize_gt_json_to_schema(md)
        lines.append(f"Example {i} (id={rid}, distance={dist:.4f}): uri={uri}")
        lines.append(json.dumps(ex, ensure_ascii=False))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# -----------------------
# Ollama backend (llava-med / qwen2.5)
# -----------------------
def ollama_has_model(model: str) -> bool:
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        return any(line.split()[0] == model for line in out.splitlines()[1:] if line.strip())
    except Exception:
        return False


def ensure_ollama_model(model: str) -> None:
    if ollama_has_model(model):
        return
    print(f"Pulling model via Ollama: {model}")
    subprocess.check_call(["ollama", "pull", model])


def image_to_base64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("utf-8")


def ollama_chat(model: str, prompt: str, image_path: Path, temperature: float = 0.0) -> str:
    import requests
    host = getattr(config, "OLLAMA_HOST", "http://localhost:11434")
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_to_base64(image_path)]}],
        "options": {"temperature": temperature},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "")


# -----------------------
# MedGemma backend (Transformers)
# -----------------------
def clean_json(text: str) -> str:
    """
    Extract { ... } from model output and remove trailing commas before } or ].
    (Same idea as your MedGemma finetune script.)  :contentReference[oaicite:1]{index=1}
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return text
    s = text[start : end + 1]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def medgemma_infer_base(prompt: str, image_path: Path, max_new_tokens: int = 512) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Runs base MedGemma (not finetuned) using transformers pipeline.
    """
    from transformers import pipeline
    from PIL import Image
    import torch

    model_id = getattr(config, "MEDGEMMA_BASE_ID", "google/medgemma-4b-it")
    hf_token = getattr(config, "HF_TOKEN", None)

    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        token=hf_token,  # works on newer transformers; safe if None
    )

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}]
    out = pipe(text=messages, max_new_tokens=max_new_tokens)

    # extract best-effort raw text
    raw = str(out)
    try:
        gen = out[0]["generated_text"]
        if isinstance(gen, list):
            last = gen[-1]
            raw = last.get("content", str(last)) if isinstance(last, dict) else str(last)
        else:
            raw = str(gen)
    except Exception:
        pass

    cleaned = clean_json(raw)
    try:
        return raw, json.loads(cleaned)
    except Exception:
        return raw, None


def load_medgemma_finetune_model():
    """
    Quantized base + LoRA adapter (PEFT), similar to your finetune script.  :contentReference[oaicite:2]{index=2}
    """
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    base_model_id = getattr(config, "MEDGEMMA_BASE_ID", "google/medgemma-4b-it")
    lora_dir = Path(getattr(config, "MEDGEMMA_LORA_DIR", ""))

    if not lora_dir or not lora_dir.exists():
        raise FileNotFoundError(f"MEDGEMMA_LORA_DIR not found: {lora_dir}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("MedGemma finetune path expects CUDA for 4-bit load.")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(lora_dir))
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True)
    processor.tokenizer.padding_side = "right"
    return model, processor


def medgemma_finetune_generate_one(model, processor, image_path: Path, prompt: str, max_new_tokens: int = 512) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Chat-template style inference (text + image), then parse JSON.
    """
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
        ],
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[[image]], return_tensors="pt", padding=True, truncation=False)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.cuda.amp.autocast(dtype=torch.float16):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    raw = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    cleaned = clean_json(raw)
    try:
        return raw, json.loads(cleaned)
    except Exception:
        return raw, None


def get_medgemma_finetune_prompt(infer_mode: str) -> str:
    """
    Uses config.MEDGEMMA_PROMPT_* if set; else falls back to built-in prompts.
    """
    infer_mode = infer_mode.lower().strip()

    # config overrides
    key_map = {
        "all": "MEDGEMMA_PROMPT_ALL",
        "birads": "MEDGEMMA_PROMPT_BIRADS",
        "density": "MEDGEMMA_PROMPT_DENSITY",
        "mass": "MEDGEMMA_PROMPT_MASS",
        "calcification": "MEDGEMMA_PROMPT_CALCIFICATION",
        "asymmetry": "MEDGEMMA_PROMPT_ASYMMETRY",
        "suspicion": "MEDGEMMA_PROMPT_SUSPICION",
    }
    cfg_key = key_map.get(infer_mode)
    if cfg_key:
        val = getattr(config, cfg_key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # built-in defaults (short but strict JSON)
    if infer_mode == "density":
        return (
            "Return ONLY valid JSON: {\"breast_density\": \"<ACR A|B|C|D>\"}\n"
            "Task: determine ACR breast density from the 4-view mammogram."
        )
    if infer_mode == "mass":
        return (
            "Return ONLY valid JSON: {\"mass\": <0|1>}\n"
            "Task: detect presence of any mass in the 4-view mammogram."
        )
    if infer_mode == "calcification":
        return (
            "Return ONLY valid JSON: {\"calcification\": <0|1>}\n"
            "Task: detect presence of suspicious calcifications in the 4-view mammogram."
        )
    if infer_mode == "asymmetry":
        return (
            "Return ONLY valid JSON: {\"asymmetry\": <0|1>}\n"
            "Task: detect presence of asymmetry in the 4-view mammogram."
        )
    if infer_mode == "birads":
        return (
            "Return ONLY valid JSON: {\"birads\": <1|2|3|4|5>}\n"
            "Task: assign BI-RADS category from the 4-view mammogram."
        )
    if infer_mode == "suspicion":
        return (
            "Return ONLY valid JSON: {\"suspicion\": \"healthy\"|\"benign\"|\"suspicious\"}\n"
            "Task: classify suspicion (healthy/benign/suspicious) from the 4-view mammogram."
        )

    # all
    return (
        "Return ONLY valid JSON with keys:\n"
        "{\"breast_density\": \"...\", \"findings\": \"...\", \"birads\": \"...\", \"suspicion\": \"healthy|benign|suspicious\"}\n"
        "Task: produce density + findings + BI-RADS + suspicion."
    )


# -----------------------
# Main orchestrator
# -----------------------
def main():
    prompt_type = pick_option(
        "Prompt type:",
        ["zeroshot", "fewshot", "cot", "rag-fewshot", "finetune"],
        default="zeroshot",
    )

    dataset = pick_option("Dataset:", ["vindr", "dmid"], default="vindr")

    model_choice = pick_option("Model:", ["medgemma", "llava-med", "qwen2.5"], default="llava-med")

    n_str = input("\nHow many images/reports to process? (-1 = all): ").strip()
    num_to_process = int(n_str) if n_str else -1

    infer_mode = "all"
    if prompt_type == "finetune":
        if model_choice != "medgemma":
            raise ValueError("prompt_type=finetune is supported ONLY for model=medgemma.")
        infer_mode = pick_option(
            "Fine-tune infer mode (MedGemma only):",
            ["all", "birads", "density", "mass", "calcification", "asymmetry", "suspicion"],
            default="all",
        )

    images_dir, reports_dir = get_dataset_dirs(dataset)
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    if num_to_process != -1:
        images = images[:num_to_process]

    print("\n====================")
    print("MammoWise run config")
    print("====================")
    print(f"dataset     : {dataset}")
    print(f"model       : {model_choice}")
    print(f"prompt_type : {prompt_type}")
    if prompt_type == "finetune":
        print(f"infer_mode  : {infer_mode}")
    print(f"images_dir  : {images_dir}")
    print(f"reports_dir : {reports_dir}")
    print(f"count       : {len(images)}")
    print("====================\n")

    processed = 0
    skipped = 0
    failed = 0

    # Prepare backends
    medgemma_finetune_bundle = None
    if model_choice == "medgemma" and prompt_type == "finetune":
        medgemma_finetune_bundle = load_medgemma_finetune_model()

    # Ollama model mapping (for llava-med/qwen2.5)
    ollama_map = getattr(config, "OLLAMA_MODEL_MAP", {})
    ollama_model_name = None
    if model_choice in ("llava-med", "qwen2.5"):
        ollama_model_name = ollama_map.get(model_choice, model_choice)
        ensure_ollama_model(ollama_model_name)

    for img_path in images:
        out_path = out_json_path(dataset, model_choice, prompt_type, img_path.stem)
        if out_path.exists():
            skipped += 1
            continue

        try:
            if model_choice in ("llava-med", "qwen2.5"):
                prompt = build_prompt(prompt_type, dataset, img_path, reports_dir)
                raw = ollama_chat(ollama_model_name, prompt, img_path, temperature=0.0)
                cleaned = clean_json(raw)
                pred = None
                try:
                    pred = json.loads(cleaned)
                except Exception:
                    pred = None

                payload = {
                    "dataset": dataset,
                    "model": model_choice,
                    "prompt_type": prompt_type,
                    "image_path": str(img_path),
                    "raw_output": raw,
                    "json_parsed": pred is not None,
                    "prediction": pred if pred is not None else None,
                }
                if pred is None:
                    payload["parse_error"] = "Failed to parse model output as JSON."

                out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

            elif model_choice == "medgemma":
                if prompt_type == "finetune":
                    model, processor = medgemma_finetune_bundle
                    prompt = get_medgemma_finetune_prompt(infer_mode)
                    raw, pred = medgemma_finetune_generate_one(model, processor, img_path, prompt, max_new_tokens=512)

                    payload = {
                        "dataset": dataset,
                        "model": model_choice,
                        "prompt_type": prompt_type,
                        "infer_mode": infer_mode,
                        "image_path": str(img_path),
                        "raw_output": raw,
                        "json_parsed": pred is not None,
                        "prediction": pred if pred is not None else None,
                    }
                    if pred is None:
                        payload["parse_error"] = "Failed to parse model output as JSON."
                    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

                else:
                    prompt = build_prompt(prompt_type, dataset, img_path, reports_dir)
                    raw, pred = medgemma_infer_base(prompt, img_path, max_new_tokens=512)
                    payload = {
                        "dataset": dataset,
                        "model": model_choice,
                        "prompt_type": prompt_type,
                        "image_path": str(img_path),
                        "raw_output": raw,
                        "json_parsed": pred is not None,
                        "prediction": pred if pred is not None else None,
                    }
                    if pred is None:
                        payload["parse_error"] = "Failed to parse model output as JSON."
                    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

            else:
                raise ValueError(f"Unsupported model choice: {model_choice}")

            processed += 1
            if processed % 10 == 0:
                print(f"✅ Processed {processed} | skipped {skipped} | failed {failed}")

        except Exception as e:
            failed += 1
            print(f"[ERROR] {img_path.name}: {e}")

    print("\n✅ Done.")
    print(f"Processed: {processed}")
    print(f"Skipped  : {skipped}")
    print(f"Failed   : {failed}")
    print(f"Outputs  : {Path(__file__).resolve().parent / 'results' / dataset / model_choice / prompt_type}")


if __name__ == "__main__":
    main()
