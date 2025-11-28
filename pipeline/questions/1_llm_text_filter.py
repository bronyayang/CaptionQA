import os, json, re, math
import argparse
import random
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from pipeline.api import AMD_vllm_text_chat_client, AMD_vllm_text_chat_call

LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ---------- helpers ----------
def extract_letter(answer_text: str, num_options: int) -> Optional[str]:
    if not answer_text:
        return None
    m = re.search(r"\b([A-Z])\b", answer_text.upper())
    if m:
        letter = m.group(1)
        idx = LETTER_ALPH.find(letter)
        if 0 <= idx < max(1, num_options):
            return letter
    m = re.search(r"\b([1-9][0-9]?)\b", answer_text)
    if m:
        k = int(m.group(1))
        if 1 <= k <= max(1, num_options):
            return LETTER_ALPH[k - 1]
    return None

def normalize_gt_letter(q: Dict[str, Any]) -> Optional[str]:
    options = q.get("options") or q.get("choices") or []
    for key in ("answer_letter", "gt_letter"):
        v = q.get(key)
        if isinstance(v, str):
            L = v.strip().upper()[:1]
            if L in LETTER_ALPH[:len(options)]:
                return L
    ans = q.get("answer")
    if isinstance(ans, int):
        if 1 <= ans <= len(options): return LETTER_ALPH[ans - 1]
    if isinstance(ans, str):
        u = ans.strip().upper()
        if u[:1] in LETTER_ALPH[:len(options)]: return u[:1]
        for i, opt in enumerate(options):
            text = opt.get("text") if isinstance(opt, dict) else str(opt)
            if ans.strip() == str(text).strip(): return LETTER_ALPH[i]
    for key in ("answer_idx", "gt_idx"):
        if key in q:
            try:
                idx = int(q[key])
                if 1 <= idx <= len(options): return LETTER_ALPH[idx - 1]
                if 0 <= idx < len(options):  return LETTER_ALPH[idx]
            except Exception:
                pass
    return None

def build_mc_question_text(q: Dict[str, Any]) -> str:
    """Build just the user content (the call will add the system prompt)."""
    question = q["question"]
    options = q.get("options") or q.get("choices") or []
    lines = []
    for idx, opt in enumerate(options):
        letter = LETTER_ALPH[idx]
        text = opt.get("text") if isinstance(opt, dict) else str(opt)
        lines.append(f"{letter}. {text}")
    return f"Question:\n{question}\n\nOptions:\n" + "\n".join(lines) + "\n\nAnswer:"

def threshold_for(n_opts: int, n: int) -> int:
    """
    Per-options thresholds (≈0.3–0.6% FPR under random guessing):
      2 -> n/n
      3 -> ceil(0.8 n)
      4 -> ceil(0.7 n)
      5+-> ceil(0.6 n)
    """
    if n_opts <= 2: return n
    if n_opts == 3: return math.ceil(0.8 * n)
    if n_opts == 4: return math.ceil(0.7 * n)
    return math.ceil(0.6 * n)

# ---------- Stage-1: single vLLM call using AMD_vllm_text_chat_call ----------
def filter_text_answerable(args):
    """
    Input:  {image_name: [ {question, options/choices, answer...}, ... ], ...}
    Output: writes two files:
      - <output_base>_kept.json
      - <output_base>_dropped.json
    """
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = list(data.keys())

    # ---------- helpers for per-sample shuffling ----------
    rng = random.Random(getattr(args, "seed", 0))

    def get_options_list(q):
        return q.get("options") or q.get("choices") or []

    def make_shuffled_q(q):
        """
        返回 (q_shuf, perm)
          - q_shuf: 复制后的题目，options/choices 已按 perm 重排
          - perm:   list，长度 = 选项数，perm[shuf_idx] = orig_idx
        """
        options = get_options_list(q)
        n_opts = len(options)
        idxs = list(range(n_opts))
        rng.shuffle(idxs)                     # 打乱到 "打乱后索引 -> 原始索引"
        # 构造重排后的 options（兼容 dict / str）
        def at(i): return options[i]
        shuffled = [at(j) for j in idxs]

        q_shuf = dict(q)                      # 浅拷贝：题干等保持不变
        if "options" in q:
            q_shuf["options"] = shuffled
        elif "choices" in q:
            q_shuf["choices"] = shuffled
        else:
            # 没有显式 options/choices，就按空处理；上层会直接 keep
            pass
        return q_shuf, idxs  # 注意：idxs 即 perm: shuf_idx -> orig_idx

    # ---------- 构造“扩展后的” batch：每题重复 args.n 次、每次随机重排 ----------
    expanded_items: List[Dict[str, str]] = []
    meta: List[tuple[int, int, List[int], int]] = []
    # meta: (image_idx, q_idx, perm, n_opts)

    for i_idx, img in enumerate(images):
        for q_idx, q in enumerate(data[img]):
            options = get_options_list(q)
            if not options or "No" in options or "no" in options:
                # 无选项：无法选择题，直接走 kept
                continue
            for _ in range(args.n):
                q_shuf, perm = make_shuffled_q(q)
                prompt = build_mc_question_text(q_shuf)  # 用打乱后的选项构建提示
                expanded_items.append({"question": prompt})
                meta.append((i_idx, q_idx, perm, max(2, len(options))))

    # 若所有题都没有可用选项，直接收尾
    if not expanded_items:
        kept = data
        dropped = {}
        base = os.path.splitext(args.output_path)[0]
        with open(f"{base}_kept.json", "w", encoding="utf-8") as f:
            json.dump(kept, f, indent=4, ensure_ascii=False)
        with open(f"{base}_dropped.json", "w", encoding="utf-8") as f:
            json.dump(dropped, f, indent=4, ensure_ascii=False)
        total = sum(len(v) for v in data.values())
        kept_n = sum(len(v) for v in kept.values())
        drop_n = 0
        print(f"Total: {total} | Kept (VL): {kept_n} | Dropped (text-answerable): {drop_n} | n={args.n}")
        return

    # ---------- vLLM：每个扩展后的提示只取 1 个样本 ----------
    client = AMD_vllm_text_chat_client(model=args.model)
    # 这里把 n 固定为 1，因为我们已经把每题重复扩展为 n 条
    all_texts = AMD_vllm_text_chat_call(
        client,
        expanded_items,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=1,
        return_all=False,              # 返回一条字符串/或与上层 API 兼容
        use_tqdm=True,
        system="You cannot see the image. Answer with a SINGLE LETTER (A, B, C, ...), no explanation.",
    )

    # 兼容如果接口总是返回 list[list[str]]
    if all_texts and isinstance(all_texts[0], list):
        all_texts = [lst[0] if lst else "" for lst in all_texts]

    # ---------- 聚合到“原始题目级别”：映回原始索引后计命中 ----------
    from collections import defaultdict
    hits = defaultdict(int)          # key=(i_idx,q_idx) -> 命中次数
    counts = defaultdict(int)        # key=(i_idx,q_idx) -> 已采样次数（应等于 args.n）

    # 预先取 GT 字母与索引
    def gt_idx_of(q):
        gtL = normalize_gt_letter(q)
        if gtL is None:
            return None
        return LETTER_ALPH.index(gtL)

    gt_idx_cache = {}  # (i_idx,q_idx) -> gt_idx or None
    for i_idx, img in enumerate(images):
        for q_idx, q in enumerate(data[img]):
            gt_idx_cache[(i_idx, q_idx)] = gt_idx_of(q)

    assert len(all_texts) == len(meta)
    for ans_text, (i_idx, q_idx, perm, n_opts) in tqdm(
        zip(all_texts, meta), total=len(all_texts), desc="Scoring (shuffled)"
    ):
        gt_idx = gt_idx_cache[(i_idx, q_idx)]
        # 无 GT 或无选项：后面整体直接 keep
        if gt_idx is None or n_opts < 2:
            continue

        # 解析模型输出字母 -> 打乱后索引 -> 原始索引
        letter = extract_letter(ans_text, n_opts)  # 返回 'A'.. 或 None
        if letter is None:
            counts[(i_idx, q_idx)] += 1
            continue
        shuf_idx = LETTER_ALPH.find(letter)
        if not (0 <= shuf_idx < len(perm)):
            counts[(i_idx, q_idx)] += 1
            continue
        orig_idx = perm[shuf_idx]                 # 映回原始索引
        if orig_idx == gt_idx:
            hits[(i_idx, q_idx)] += 1
        counts[(i_idx, q_idx)] += 1

    # ---------- 根据阈值划分 keep/drop ----------
    kept: Dict[str, List[Dict[str, Any]]] = {img: [] for img in images}
    dropped: Dict[str, List[Dict[str, Any]]] = {img: [] for img in images}

    for i_idx, img in enumerate(images):
        for q_idx, q in enumerate(data[img]):
            options = get_options_list(q)
            n_opts = max(2, len(options))
            gtL = normalize_gt_letter(q)

            if not options or gtL is None:
                kept[img].append(q)
                continue

            # 若该题扩展条目曾被过滤（极端情况），补齐逻辑
            c = counts.get((i_idx, q_idx), 0)
            h = hits.get((i_idx, q_idx), 0)
            # 正常应等于 args.n；若不是，用实际 c 计算阈值
            t = threshold_for(n_opts, c if c > 0 else args.n)

            if h >= t:
                dropped[img].append(q)
            else:
                kept[img].append(q)

    # ---------- 收尾保存 ----------
    kept    = {k: v for k, v in kept.items() if v}
    dropped = {k: v for k, v in dropped.items() if v}

    base = os.path.splitext(args.output_path)[0]
    kept_path = f"{base}_kept.json"
    drop_path = f"{base}_dropped.json"
    os.makedirs(os.path.dirname(kept_path) or ".", exist_ok=True)
    with open(kept_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=4, ensure_ascii=False)
    with open(drop_path, "w", encoding="utf-8") as f:
        json.dump(dropped, f, indent=4, ensure_ascii=False)

    total = sum(len(v) for v in data.values())
    kept_n = sum(len(v) for v in kept.values())
    drop_n = sum(len(v) for v in dropped.values())
    print(f"Total: {total} | Kept (VL): {kept_n} | Dropped (text-answerable): {drop_n} | n={args.n}")

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path",  required=True, help="输入 JSON（{image: [q1, q2, ...]}）")
    p.add_argument("--output_path", required=True, help="输出基名（会生成 *_kept.json 和 *_dropped.json）")
    p.add_argument("--model",       required=True, help="vLLM 服务的模型标识，如 'llama-3.1-8b-instruct'")
    p.add_argument("--n",           type=int, default=10, help="每题采样次数（推荐 10；省算力可用 8）")
    p.add_argument("--temperature", type=float, default=1.0, help="采样温度（0.7–1.0 建议）")
    p.add_argument("--max_tokens",  type=int, default=4, help="生成上限（只要字母即可，4 足够）")
    p.add_argument("--seed",        type=int, default=0, help="选项随机化的随机种子")
    return p.parse_args()

if __name__ == "__main__":
    args = build_args()
    filter_text_answerable(args)