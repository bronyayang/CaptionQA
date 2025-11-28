#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM-only embedding pipeline for Qwen3-Embedding, saving per-image.

Input JSON(s) format:
{
  "1.jpg": [
    {"question": "...", "choices": ["Yes","No"], "answer": "Yes", "category": "..."},
    ...
  ],
  "2.jpg": [...]
}

Outputs (under --out-dir):
  - <sanitized_image_id>/embeddings.npy      # (Ni, D) float32, L2-normalized
  - <sanitized_image_id>/meta.json            # rows for this image
  - manifest.csv                                      # summary over all images

Run:
  python /home/shijyang/code/CaptionQA/pipeline/questions/2_qwen_embed.py \
    --json-glob "/workspace/data/*.json" \
    --model-id "Qwen/Qwen3-Embedding-8B" \
    --out-dir "/mnt/m2m_nobackup/qwen3_natural" \
    --tensor-parallel-size 1 \
    --dtype bfloat16
"""
import argparse, glob, json, os, csv, re, hashlib
from dataclasses import dataclass
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import torch

# ---------- text template (no choice-prefix stripping) ----------
def canon_text(question: str, choices: List[str]) -> str:
    joined = " | ".join(choices or [])
    instr = "Instruct: Represent the meaning of a multiple-choice visual question for deduplication.\n"
    return f"{instr}Question: {question.strip()}\nChoices: {joined}"

@dataclass
class Row:
    image_id: str
    src_file: str
    item_idx: int
    question: str
    choices: List[str]
    answer: str
    category: str
    text: str  # embedding text

def sanitize_image_id(image_id: str) -> str:
    """
    Make a safe directory name for the image key.
    Keeps readable stem, appends short hash to avoid collisions.
    """
    safe = re.sub(r"[^\w.\-]+", "_", image_id)  # keep alnum, _, ., -
    return f"{safe}"

def load_grouped_rows(json_paths: List[str]) -> Dict[str, List[Row]]:
    grouped: Dict[str, List[Row]] = {}
    for p in json_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for image_id, items in data.items():
            if not isinstance(items, list):
                continue
            bucket = grouped.setdefault(image_id, [])
            for idx, it in enumerate(items):
                q = (it.get("question") or "").strip()
                ch = it.get("choices") or []
                ans = it.get("answer", "")
                cat = it.get("category", "")
                bucket.append(Row(
                    image_id=image_id,
                    src_file=os.path.basename(p),
                    item_idx=idx,
                    question=q,
                    choices=ch,
                    answer=ans,
                    category=cat,
                    text=canon_text(q, ch),
                ))
    return grouped

# ---------- vLLM helpers ----------
def embed_one_image(llm, texts: List[str]) -> np.ndarray:
    outputs = llm.embed(texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    # L2 normalize
    norms = torch.linalg.norm(embeddings, ord=2, dim=1, keepdim=True)
    embeddings.div_(norms.clamp_min(1e-12))   # in-place, numerically safe

    return embeddings.cpu().numpy().astype(np.float32)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-glob", required=True, help="Glob for input JSONs, e.g. '/data/*.json'")
    ap.add_argument("--model-id", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--out-dir", default="out_embeds")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(args.json_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.json_glob}")

    print(f"[load] {len(paths)} JSON files")
    grouped = load_grouped_rows(paths)
    print(f"[prepare] {sum(len(v) for v in grouped.values())} total Q/A items across {len(grouped)} images")

    # Build vLLM once
    from vllm import LLM
    llm = LLM(
        model=args.model_id,
        task="embed",
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    # Manifest over all images
    manifest_rows = []
    for image_id in tqdm(sorted(grouped.keys()), desc="per-image embed"):
        rows = grouped[image_id]
        texts = [r.text for r in rows]
        embs = embed_one_image(llm, texts)

        # Save per-image
        safe_id = sanitize_image_id(image_id)
        img_dir = os.path.join(args.out_dir, safe_id)
        os.makedirs(img_dir, exist_ok=True)

        np.save(os.path.join(img_dir, "embeddings.npy"), embs.astype(np.float32))
        meta_full = []
        for r in rows:
            # keep key order: question, choices, answer, category (then source info)
            meta_full.append({
                "question": r.question,
                "choices": list(r.choices),
                "answer": r.answer,
                "category": r.category,
                "src_file": r.src_file,
                "item_idx": r.item_idx,
                "text": r.text,
            })
        with open(os.path.join(img_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_full, f, ensure_ascii=False, indent=2)

        manifest_rows.append([image_id, safe_id, len(rows), img_dir])

    # Save manifest
    with open(os.path.join(args.out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "sanitized_id", "num_items", "image_out_dir"])
        w.writerows(manifest_rows)

    print(f"[done] outputs under: {args.out_dir}")
    print(f"[done] manifest: {os.path.join(args.out_dir, 'manifest.json')}")
    print("[tip] Each image folder contains embeddings.npy (Ni x D) and meta.json for that image.")

if __name__ == "__main__":
    main()
