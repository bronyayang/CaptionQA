import os, json, re, argparse
from typing import Dict, Any, List, Optional, Tuple

LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ---------- 前缀清理 ----------
_PREFIX_PATTERNS = [
    # 1) 字母 + .(或全角．) + 空格，且后面不是「单字母+点」的缩写（避免 "A. C. Milan" 这类被误切）
    re.compile(r'^\s*[\(\（]?\s*[A-Za-z]\s*[.．]\s+(?![A-Za-z]\s*[.．])'),

    # 2) 字母 + 其它常见枚举符号（不含 .），如 A) / A：/ （A）/ A、 等
    re.compile(r'^\s*[\(\（]?\s*[A-Za-z]\s*[\)\）:：、]\s*'),

    # 3) 数字编号：1) / （2） / 3、 / 4： / 12) / 11. 但避免误伤小数/时间（11.99 / 1:23）
    re.compile(r'^\s*[\(\（]?\s*\d{1,2}\s*(?:[\)\）、]|[:：](?!\d)|[.．](?!\d))\s*'),

    # 4) ①②③… 这类序号
    re.compile(r'^\s*[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]\s*'),
]

# 仅字母答案（可带括号/点/冒号）或“answer: X/答案: X”
_ANS_WITH_PREFIX_PAT  = re.compile(r'^\s*(?:answer|答案)\s*[:：]?\s*[A-Za-z]\s*$', re.IGNORECASE)

def strip_option_prefix_once(text: str) -> Tuple[str, bool]:
    if not isinstance(text, str):
        return str(text), False
    for pat in _PREFIX_PATTERNS:
        m = pat.match(text)
        if m:
            return text[m.end():].lstrip(), True
    return text, False

def sanitize_prefixes(q: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """清理 q 的 options/choices 前缀；返回 (新题目, 被修改的选项个数)。"""
    key = "options" if "options" in q else ("choices" if "choices" in q else None)
    if key is None or not isinstance(q.get(key), list):
        return q, 0

    modified = 0
    new_opts: List[Any] = []
    for o in q[key]:
        new_t, changed = strip_option_prefix_once(str(o))
        if changed: 
            modified = 1
        new_opts.append(new_t)

    q2 = dict(q)

    if modified == 1:
        q2["answer"] = strip_option_prefix_once(str(q["answer"]))[0]
        print("Modified Question: ", q)

    q2[key] = new_opts
    return q2, modified

# ---------- 其余工具 ----------
def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s\W_]+", " ", s)
    return s.strip()

def get_options_list(q: Dict[str, Any]) -> List[str]:
    opts = q.get("options") or q.get("choices") or []
    out: List[str] = []
    for o in opts:
        if isinstance(o, dict):
            text = o.get("text")
            out.append("" if text is None else str(text))
        else:
            out.append(str(o))
    return out

def extract_letter(answer_text: str, num_options: int) -> Optional[str]:
    if not answer_text:
        return None
    m = re.search(r"\b([A-Z])\b", answer_text.upper())
    if m:
        L = m.group(1)
        idx = LETTER_ALPH.find(L)
        if 0 <= idx < max(1, num_options):
            return L
    m = re.search(r"\b([1-9][0-9]?)\b", answer_text)
    if m:
        k = int(m.group(1))
        if 1 <= k <= max(1, num_options):
            return LETTER_ALPH[k - 1]
    return None

def normalize_gt_letter(q: Dict[str, Any], options: List[str]) -> Optional[str]:
    n = len(options)
    for key in ("answer_letter", "gt_letter"):
        v = q.get(key)
        if isinstance(v, str) and v:
            L = v.strip().upper()[:1]
            if L in LETTER_ALPH[:n]:
                return L
    for key in ("answer_idx", "gt_idx"):
        if key in q:
            try:
                idx = int(q[key])
                if 1 <= idx <= n:  return LETTER_ALPH[idx - 1]
                if 0 <= idx < n:   return LETTER_ALPH[idx]
            except Exception:
                pass
    ans = q.get("answer")
    if isinstance(ans, int):
        if 1 <= ans <= n: return LETTER_ALPH[ans - 1]
        if 0 <= ans < n:  return LETTER_ALPH[ans]
    if isinstance(ans, list):
        norm_opts = [normalize_text(x) for x in options]
        for a in ans:
            if isinstance(a, str) and normalize_text(a) in norm_opts:
                j = norm_opts.index(normalize_text(a)); return LETTER_ALPH[j]
        for a in ans:
            if isinstance(a, str):
                L = extract_letter(a, n)
                if L is not None: return L
    if isinstance(ans, str):
        norm_ans = normalize_text(ans)
        norm_opts = [normalize_text(x) for x in options]
        if norm_ans in norm_opts:
            j = norm_opts.index(norm_ans); return LETTER_ALPH[j]
        L = extract_letter(ans, n)
        if L is not None: return L
    return None

def should_drop_by_gt_mapping(q: Dict[str, Any]) -> bool:
    """选项>=2 且无法把 GT 映射到任一选项 → drop；否则 keep。"""
    options = get_options_list(q)
    if len(options) < 2:
        return False
    L = normalize_gt_letter(q, options)
    if L is None:
        return True
    idx = LETTER_ALPH.find(L)
    return not (0 <= idx < len(options))

# ---------- implied 关键词过滤（caption 场景） ----------
def _extract_question_text(q: Dict[str, Any]) -> str:
    """尽力从常见字段提取问题文本。"""
    for k in ("question", "prompt", "text", "instruction", "query", "caption"):
        if k in q:
            v = q[k]
            if isinstance(v, str):
                return v
            try:
                return " ".join(x for x in v if isinstance(x, str))
            except Exception:
                return str(v)
    # 兜底：若题干在别处
    return json.dumps(q, ensure_ascii=False)

def _build_kw_patterns(kws: List[str]) -> List[re.Pattern]:
    pats = []
    for kw in kws:
        kw = kw.strip()
        if not kw:
            continue
        # 短词用 \b，短语用“非字母边界”
        if " " in kw:
            pats.append(re.compile(r'(?<![A-Za-z])' + re.escape(kw) + r'(?![A-Za-z])', re.IGNORECASE))
        else:
            pats.append(re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE))
    return pats

def should_drop_by_keywords(q: Dict[str, Any], kw_patterns: List[re.Pattern]) -> bool:
    text = _extract_question_text(q)
    if not isinstance(text, str) or not text:
        return False
    for pat in kw_patterns:
        if pat.search(text):
            return True
    return False

# ---------- 主流程 ----------
def run(input_path: str, output_path: str, drop_caption_on_implied: bool, drop_keywords: List[str]) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data: Dict[str, List[Dict[str, Any]]] = json.load(f)

    kw_patterns = _build_kw_patterns(drop_keywords) if drop_caption_on_implied else []

    images = list(data.keys())
    kept: Dict[str, List[Dict[str, Any]]] = {img: [] for img in images}
    dropped: Dict[str, List[Dict[str, Any]]] = {img: [] for img in images}

    total = 0
    drop_n = 0
    modified_questions = 0

    for img in images:
        for q in data[img]:
            total += 1

            # 1) 清理选项前缀
            q_clean, mod_cnt = sanitize_prefixes(q)

            # 2) 是否因 implied 类关键词而 drop（caption 过滤）
            drop_kw = drop_caption_on_implied and should_drop_by_keywords(q_clean, kw_patterns)

            # 3) 基于 GT 映射规则是否 drop
            drop_map = should_drop_by_gt_mapping(q_clean)
            is_drop = drop_kw or drop_map

            # 5) 题目级修改计数
            modified_questions += mod_cnt

            if is_drop:
                dropped[img].append(q_clean)
                drop_n += 1
            else:
                kept[img].append(q_clean)

    # 清理空键
    kept    = {k: v for k, v in kept.items() if v}
    dropped = {k: v for k, v in dropped.items() if v}

    base = os.path.splitext(output_path)[0]
    kept_path = f"{base}_kept.json"
    drop_path = f"{base}_dropped.json"
    os.makedirs(os.path.dirname(kept_path) or ".", exist_ok=True)
    with open(kept_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    with open(drop_path, "w", encoding="utf-8") as f:
        json.dump(dropped, f, indent=2, ensure_ascii=False)

    kept_n = sum(len(v) for v in kept.values())
    print(f"Total: {total} | Kept: {kept_n} | Dropped: {drop_n} | Modified questions: {modified_questions}")

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path",  required=True, help="输入 JSON：{ image: [ {question, options/choices, answer...}, ... ], ... }")
    p.add_argument("--output_path", required=True, help="输出基名（会生成 *_kept.json 和 *_dropped.json）")
    p.add_argument("--drop_caption_on_implied", action="store_true",
                   help="启用：如果 question 含“implied/implicit/infer/assume/suggests”等词则直接 drop（caption 友好）")
    p.add_argument("--drop_keywords", type=str,
                   default="implied,implies,implicit,implicitly,infer,inferred,inference,assume,assumption,suggests",
                   help="逗号分隔关键词，大小写不敏感；短语需用引号包裹整串")
    return p.parse_args()

if __name__ == "__main__":
    args = build_args()
    kws = [x.strip() for x in (args.drop_keywords or "").split(",") if x.strip()]
    run(args.input_path, args.output_path, args.drop_caption_on_implied, kws)