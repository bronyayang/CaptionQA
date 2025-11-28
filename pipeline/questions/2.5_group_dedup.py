#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partition questions within each image into non-overlapping groups and write:
  - groups.json                  # [[idx,...], ...]
  - groups_full.json             # [{rep_idx:..., members:[...], members_meta:[{question,choices,answer,category,...}, ...]}, ...]
  - representatives.txt          # one index per group (medoid)
  - map_member_to_rep.json       # {"member_idx": rep_idx, ...}
  - stats.json                   # summary
  - dedup.json                   # {"<image_id>": [ {question, choices, answer, category}, ... ] }

Assumes per-image folders produced by the embed step:
  <root>/<image_id>/embeddings.npy
  <root>/<image_id>/meta.json    # full fields saved above
"""
import os, json, csv, argparse
from typing import List, Dict
import numpy as np
import re

def get_gpt_tag(src_file: str) -> str:
    name = os.path.basename(src_file)
    if name.endswith('_text_kept.json'):
        return name[:-len('_text_kept.json')].split('_')[-1]
    return ""

# ---------- similarity ----------
def is_normalized(X: np.ndarray, tol: float = 1e-3) -> bool:
    n = np.linalg.norm(X, axis=1)
    return np.all(np.abs(n - 1.0) < tol)

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)

def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    if not is_normalized(X): X = l2_normalize(X)
    return X @ X.T

# ---------- graph ----------
def mutual_knn_adjacency(S: np.ndarray, k: int, tau: float) -> np.ndarray:
    N = S.shape[0]
    if N <= 1: return np.zeros((N, N), dtype=bool)
    np.fill_diagonal(S, -np.inf)
    k_eff = min(k, N-1) if N > 1 else 0
    if k_eff <= 0:
        A = (S >= tau); np.fill_diagonal(A, False); return A
    idx_sorted = np.argsort(-S, axis=1)[:, :k_eff]
    rows = np.arange(N)[:, None]
    topk = np.zeros_like(S, dtype=bool)
    topk[rows, idx_sorted] = True
    A = (topk & topk.T) & (S >= tau)
    np.fill_diagonal(A, False)
    return A

def connected_components(A: np.ndarray) -> List[List[int]]:
    N = A.shape[0]
    seen = np.zeros(N, dtype=bool)
    comps: List[List[int]] = []
    for i in range(N):
        if seen[i]: continue
        stack = [i]; seen[i] = True; g = [i]
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(A[u]):
                if not seen[v]:
                    seen[v] = True; stack.append(v); g.append(v)
        comps.append(sorted(g))
    return comps

def merge_audit(S: np.ndarray, groups: List[List[int]], tau: float) -> List[List[int]]:
    changed = True
    while changed and len(groups) > 1:
        changed = False
        G = len(groups)
        pair = None
        for i in range(G):
            gi = groups[i]
            for j in range(i+1, G):
                gj = groups[j]
                if np.max(S[np.ix_(gi, gj)]) >= tau:
                    pair = (i, j); changed = True; break
            if changed: break
        if changed:
            i, j = pair
            merged = sorted(groups[i] + groups[j])
            groups = [g for t, g in enumerate(groups) if t not in (i, j)] + [merged]
    return sorted(groups, key=lambda g: (len(g)==1, g[0]))

# ---------- cohesion guard (optional) ----------
def min_intra_similarity(S: np.ndarray, g: List[int]) -> float:
    if len(g) <= 1: return 1.0
    sub = S[np.ix_(g, g)].copy()
    np.fill_diagonal(sub, 1.0)
    mask = ~np.eye(len(g), dtype=bool)
    return float(np.min(sub[mask]))

def farthest_pair(X: np.ndarray):
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    i = int(np.argmax(np.max(S, axis=1)))
    j = int(np.argmin(S[i]))
    return i, j

def two_means_split(X: np.ndarray, idxs: List[int], max_iter=30):
    if len(idxs) <= 2:
        return ([idxs[0]], [idxs[1]]) if len(idxs)==2 else (idxs, [])
    subX = X[idxs]
    a, b = farthest_pair(subX)
    Ca, Cb = subX[a].copy(), subX[b].copy()
    assign = None
    for _ in range(max_iter):
        sim_a, sim_b = subX @ Ca, subX @ Cb
        new = sim_a >= sim_b
        if assign is not None and np.array_equal(new, assign): break
        assign = new
        Xa, Xb = subX[assign], subX[~assign]
        if len(Xa)==0 or len(Xb)==0: return idxs, []
        Ca = Xa.mean(axis=0); Ca /= max(np.linalg.norm(Ca), 1e-12)
        Cb = Xb.mean(axis=0); Cb /= max(np.linalg.norm(Cb), 1e-12)
    A = [idxs[i] for i,t in enumerate(assign) if t]
    B = [idxs[i] for i,t in enumerate(assign) if not t]
    return A, B

def cohesion_refine(Xn: np.ndarray, S: np.ndarray, groups: List[List[int]], tau_min: float, min_size=3):
    out = []
    work = list(groups)
    while work:
        g = work.pop()
        if len(g) >= min_size and min_intra_similarity(S, g) < tau_min:
            A, B = two_means_split(Xn, g)
            if len(B)==0: out.append(g)
            else: work.append(A); work.append(B)
        else:
            out.append(g)
    return sorted([sorted(x) for x in out], key=lambda h: (len(h)==1, h[0]))

# ---------- representatives ----------
def group_medoids(S: np.ndarray, groups: List[List[int]]) -> List[int]:
    reps = []
    for g in groups:
        if len(g)==1: reps.append(g[0]); continue
        sub = S[np.ix_(g, g)].copy(); np.fill_diagonal(sub, 0.0)
        scores = sub.sum(axis=1)
        reps.append(g[int(np.argmax(scores))])
    return reps

# ---------- per-image process ----------
def process_image_dir(image_dir: str, tau: float, knn: int,
                      guard: bool, tau_min: float, min_size_to_split: int) -> Dict:
    emb_p = os.path.join(image_dir, "embeddings.npy")
    meta_p = os.path.join(image_dir, "meta.json")
    if not os.path.isfile(emb_p): return {"image_dir": image_dir, "status": "skip_no_embeddings"}
    if not os.path.isfile(meta_p): return {"image_dir": image_dir, "status": "skip_no_meta_json"}
    
    X = np.load(emb_p)  # (N,D)
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)  # list[dict] with question/choices/answer/category/...

    N = X.shape[0]
    if N != len(meta): return {"image_dir": image_dir, "status": "mismatch_embeddings_meta", "N": int(N), "M": int(len(meta))}

    if N <= 1:
        groups = [[0]] if N==1 else []
        reps = [0] if N==1 else []
        out_dir = os.path.join(image_dir, "groups"); os.makedirs(out_dir, exist_ok=True)
        json.dump(groups, open(os.path.join(out_dir,"groups.json"),"w"))
        open(os.path.join(out_dir,"representatives.txt"),"w").write(("0\n" if N==1 else ""))
        json.dump({("0" if N==1 else ""): (0 if N==1 else "")}, open(os.path.join(out_dir,"map_member_to_rep.json"),"w"))
        json.dump({"N":N,"tau":tau,"knn":knn,"guard":guard,"tau_min":tau_min}, open(os.path.join(out_dir,"stats.json"),"w"), indent=2)
        # dedup.json
        img_id = os.path.basename(image_dir)
        dedup_items = ([{"question": meta[0]["question"], "choices": meta[0]["choices"],
                         "answer": meta[0].get("answer",""), "category": meta[0].get("category",""), 
                         "source": get_gpt_tag(meta[0].get("src_file",""))}]
                    if N==1 else [])
        json.dump({img_id: dedup_items}, open(os.path.join(out_dir,"dedup.json"),"w"), ensure_ascii=False, indent=2)
        return {"image_dir": image_dir, "status": "ok_trivial", "N": N}

    # 1) S
    S = cosine_sim_matrix(X)

    # 2) mutual-kNN + tau
    A = mutual_knn_adjacency(S.copy(), k=knn, tau=tau)

    # 3) components
    groups = connected_components(A)

    # 4) merge audit @tau
    groups = merge_audit(S, groups, tau)

    # 5) cohesion guard (optional) + final merge audit
    if guard:
        Xn = X if is_normalized(X) else l2_normalize(X)
        groups = cohesion_refine(Xn, S, groups, tau_min=tau_min, min_size=min_size_to_split)
        groups = merge_audit(S, groups, tau)

    # 6) representatives (medoids)
    reps = group_medoids(S, groups)

    # write outputs
    out_dir = os.path.join(image_dir, "groups"); os.makedirs(out_dir, exist_ok=True)
    groups = [[int(x) for x in g] for g in groups]
    reps = [int(r) for r in reps]
    json.dump(groups, open(os.path.join(out_dir,"groups.json"),"w"), ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir,"representatives.txt"),"w") as f:
        for r in reps: f.write(f"{r}\n")
    m2r = {}
    for g, r in zip(groups, reps):
        for m in g: m2r[str(m)] = int(r)
    json.dump(m2r, open(os.path.join(out_dir,"map_member_to_rep.json"),"w"), indent=2)

    sizes = [len(g) for g in groups]
    json.dump({"N":int(N),"num_groups":int(len(groups)),"sizes":sizes,"tau":float(tau),
               "knn":int(knn),"guard":bool(guard),"tau_min":float(tau_min)},
              open(os.path.join(out_dir,"stats.json"),"w"), indent=2)

    # groups_full.json (members with full meta)
    groups_full = []
    rep_cats = []
    for g, r in zip(groups, reps):
        cats = set()
        for i in g:
            cats.add(meta[i].get("category",""))
        groups_full.append({
            "rep_idx": int(r),
            "members": [int(x) for x in g],
            "members_meta": [
                {
                    "question": meta[i]["question"],
                    "choices": meta[i]["choices"],
                    "answer": meta[i].get("answer",""),
                    "category": meta[i].get("category",""),
                    "src_file": meta[i].get("src_file",""),
                    "item_idx": meta[i].get("item_idx", i),
                } for i in g
            ],
        })
        rep_cats.append(list(cats))
    json.dump(groups_full, open(os.path.join(out_dir,"groups_full.json"),"w"), ensure_ascii=False, indent=2)

    # dedup.json (PER IMAGE) in your required format
    img_id = os.path.basename(image_dir)
    dedup_items = []
    for i, r in enumerate(reps):
        # preserve key order: question, choices, answer, category
        dedup_items.append({
            "question": meta[r]["question"],
            "choices": list(meta[r]["choices"]),
            "answer": meta[r].get("answer",""),
            "category": rep_cats[i],
            "source": get_gpt_tag(meta[r].get("src_file","")),
        })
    json.dump({img_id: dedup_items}, open(os.path.join(out_dir,"dedup.json"),"w"),
            ensure_ascii=False, indent=2)

    return {"image_dir": image_dir, "status": "ok", "N": int(N), "num_groups": len(groups), "sizes": sizes}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--tau-min", type=float, default=0.85)
    ap.add_argument("--min-size-to-split", type=int, default=3)
    ap.add_argument("--write-merged-json", action="store_true",
                    help="also write <root>/_group_summary/final_dedup.json merging all per-image dedup.json")
    args = ap.parse_args()

    # discover image dirs
    subdirs = []
    for name in sorted(os.listdir(args.root)):
        p = os.path.join(args.root, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "embeddings.npy")):
            subdirs.append(p)

    print(f"[scan] {len(subdirs)} image dirs under {args.root}")
    stats = []
    for d in subdirs:
        info = process_image_dir(
            d, tau=args.tau, knn=args.knn,
            guard=args.guard, tau_min=args.tau_min,
            min_size_to_split=args.min_size_to_split
        )
        stats.append(info)
        print(f"[{info['status']}] {d} -> N={info.get('N','-')} groups={info.get('num_groups','-')}")

    # summary and optional merged final JSON
    summ_dir = os.path.join(args.root, "_group_summary")
    os.makedirs(summ_dir, exist_ok=True)
    with open(os.path.join(summ_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if args.write_merged_json:
        final = {}
        for d in subdirs:
            dedup_p = os.path.join(d, "groups", "dedup.json")
            if os.path.isfile(dedup_p):
                obj = json.load(open(dedup_p, "r", encoding="utf-8"))
                final.update(obj)
        with open(os.path.join(summ_dir, "final_dedup.json"), "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        print(f"[done] merged final_dedup.json at {summ_dir}")

if __name__ == "__main__":
    main()
