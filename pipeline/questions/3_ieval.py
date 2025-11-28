#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, re, time
from typing import Dict, Any, List
from tqdm import tqdm
from pipeline.api import AMD_openai_client, AMD_openai_call, AMD_gemini_client, AMD_gemini_call
from pipeline.utils import load_json, encode_image
import openai

SPECIAL_FLAGS = [
    "AMBIGUOUS_QUESTION",
    "UNANSWERABLE_FROM_IMAGE",
    "NOT_SUITABLE_FOR_CAPTION_EVAL",
    "NONE_OF_THE_ABOVE"
]

LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

PROMPT_TEMPLATE = (
    "You are validating a multiple-choice question for image-caption evaluation.\n"
    "You CAN see the image(s) (already provided above).\n\n"
    "Answer the question by choosing exactly ONE option from the list below.\n"
    "The options include the original answers PLUS three meta-options:\n"
    f"- {SPECIAL_FLAGS[0]}: question wording is unclear or multiple options could be valid.\n"
    f"- {SPECIAL_FLAGS[1]}: question cannot be answered solely from visible evidence in the image(s).\n"
    f"- {SPECIAL_FLAGS[2]}: question is not suitable for evaluating a caption of these image(s) (e.g., unrelated common knowledge, subjective taste, requires reasoning/prediction).\n"
    f"- {SPECIAL_FLAGS[3]}: correct answer is not among the options.\n\n"
    "Return ONLY a SINGLE UPPERCASE LETTER (A, B, C, ...). DO NOT add any explanation.\n\n"
    "Question:\n{question}\n\nOptions:\n{options}\n\n"
    "Your answer (ONE LETTER ONLY):"
)

def build_options_block(q: Dict[str, Any]) -> List[str]:
    """Return the final options list: original options + 3 special flags (as options)."""
    orig = q.get("options") or q.get("choices") or []
    normalized = []
    for opt in orig:
        text = opt.get("text") if isinstance(opt, dict) else str(opt)
        normalized.append(text)
    # append the three flags as additional options
    normalized.extend(SPECIAL_FLAGS)
    return normalized

def options_to_text_block(options: List[str]) -> str:
    lines = []
    for i, text in enumerate(options):
        lines.append(f"{LETTER_ALPH[i]}. {text}")
    return "\n".join(lines) if lines else "(no options)"

def load_images_for_key(image_root: str, img_key: str) -> tuple[list[str], bool]:
    """
    Load images for a given key, handling both single images and multi-image folders.
    
    Returns:
        tuple: (list_of_base64_images, is_multi_image)
    """
    img_path = os.path.join(image_root, img_key)
    
    if os.path.isfile(img_path):
        # Single image file
        try:
            image_b64 = encode_image(img_path)
            return [image_b64], False
        except Exception as e:
            raise Exception(f"Failed to read single image {img_path}: {e}")
    
    elif os.path.isdir(img_path):
        # Directory containing multiple images
        folder_images = []
        for img_file in os.listdir(img_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                folder_images.append(os.path.join(img_path, img_file))
        
        if not folder_images:
            raise Exception(f"No image files found in directory {img_path}")
        
        folder_images.sort()  # Sort images within folder
        
        # Load all images as base64
        images_b64 = []
        for img_file_path in folder_images:
            try:
                image_b64 = encode_image(img_file_path)
                images_b64.append(image_b64)
            except Exception as e:
                raise Exception(f"Failed to read image {img_file_path}: {e}")
        
        return images_b64, True
    
    else:
        raise Exception(f"Path not found: {img_path}")

def extract_letter(s: str, n_opts: int) -> str | None:
    """Extract first standalone A..Z letter that falls within range [0, n_opts)."""
    if not s:
        return None
    m = re.search(r"\b([A-Z])\b", s.strip().upper())
    if not m:
        return None
    L = m.group(1)
    idx = LETTER_ALPH.find(L)
    if 0 <= idx < n_opts:
        return L
    return None

def run(args):
    # read data
    data: Dict[str, List[Dict[str, Any]]] = load_json(args.input_path)

    images = list(data.keys())
    if not images:
        base = os.path.splitext(args.output_path)[0]
        with open(f"{base}_good.json", "w", encoding="utf-8") as f: json.dump({}, f, indent=4, ensure_ascii=False)
        with open(f"{base}_bad.json",  "w", encoding="utf-8") as f: json.dump({}, f, indent=4, ensure_ascii=False)
        print("Total: 0 | Good: 0 | Bad: 0")
        return

    if args.model == 'gemini-2.5-pro':
        client = AMD_gemini_client()
    else:
        client = AMD_openai_client(model_id=args.model)

    # Load existing good/bad files for resume functionality
    base = os.path.splitext(args.output_path)[0]
    good_path = f"{base}_good.json"
    bad_path = f"{base}_bad.json"
    
    # Initialize or load existing results
    good: Dict[str, List[Dict[str, Any]]] = {}
    bad: Dict[str, List[Dict[str, Any]]] = {}
    
    if os.path.exists(good_path):
        try:
            good = load_json(good_path)
            good_questions = sum(len(questions) for questions in good.values())
            print(f"Loaded existing good file with {len(good)} images ({good_questions} questions)")
        except Exception as e:
            print(f"Warning: Could not load existing good file: {e}")
    
    if os.path.exists(bad_path):
        try:
            bad = load_json(bad_path)
            bad_questions = sum(len(questions) for questions in bad.values())
            print(f"Loaded existing bad file with {len(bad)} images ({bad_questions} questions)")
        except Exception as e:
            print(f"Warning: Could not load existing bad file: {e}")

    def save_results():
        """Save good and bad results to files"""
        os.makedirs(os.path.dirname(good_path) or ".", exist_ok=True)
        with open(good_path, "w", encoding="utf-8") as f: 
            json.dump(good, f, indent=4, ensure_ascii=False)
        with open(bad_path, "w", encoding="utf-8") as f: 
            json.dump(bad, f, indent=4, ensure_ascii=False)

    total = good_n = bad_n = 0

    for img_key in tqdm(images, desc="Per image"):
        # Skip if already processed (resume functionality)
        if img_key in good or img_key in bad:
            tqdm.write(f"[skip] {img_key}: already processed (found in existing results)")
            continue
            
        if args.model == 'gemini-2.5-pro':
            image_path = os.path.join(args.image_root, img_key)
            is_multi = os.path.isdir(image_path)
            if is_multi:
                all_paths = []
                for img_file in os.listdir(image_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        all_paths.append(os.path.join(image_path, img_file))
                
                if not all_paths:
                    raise Exception(f"No image files found in directory {image_path}")
        
                all_paths.sort()
                image_path = all_paths
            
            if isinstance(image_path, str):
                image_path = [image_path]
        else:
            images_b64, is_multi = load_images_for_key(args.image_root, img_key)
        if is_multi:
            if args.model == 'gemini-2.5-pro':
                tqdm.write(f"[info] {img_key}: processing {len(image_path)} images (multi-image folder)")
            else:
                tqdm.write(f"[info] {img_key}: processing {len(images_b64)} images (multi-image folder)")
        else:
            tqdm.write(f"[info] {img_key}: processing single image")


        for q in tqdm(data[img_key], desc=f"Questions for {img_key}", leave=False):
            question = str(q.get("question", "")).strip()
            all_options = build_options_block(q)  # original + 3 flags
            if not all_options:
                # no options => can't choose; treat as unsuitable for caption eval
                q["flags"] = ["NOT_SUITABLE_FOR_CAPTION_EVAL"]
                continue

            options_block = options_to_text_block(all_options)
            prompt = PROMPT_TEMPLATE.format(question=question, options=options_block)

            # Retry logic with exponential backoff for rate limits
            max_retries = 5
            retry_delay = 2.0
            completion = None
            
            for attempt in range(max_retries + 1):
                try:
                    if args.model == 'gemini-2.5-pro':
                        # ===== Gemini 路径（多图 b64）=====
                        # messages 可以是字符串（最简单）；也可保留 system 指令时传列表
                        completion = AMD_gemini_call(
                            client,
                            model_id=str(args.model),  # e.g., "gemini-2.5-pro"
                            messages=prompt,           # 直接把 prompt 文本作为内容
                            image_paths=image_path,
                            temperature=args.temperature
                        )
                    else:
                        content_items = []
                        for image_b64 in images_b64:
                            content_items.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                            })

                        messages = [
                            {
                                "role": "user",
                                "content": content_items,
                            },
                            {"role": "user", "content": prompt},
                        ]
                        completion = AMD_openai_call(
                            client,
                            args.model,
                            messages=messages,
                            temperature=args.temperature,
                            stream=False,
                            reasoning_effort="low"
                        )
                    break  # Success, exit retry loop
                except openai.OpenAIError as e:
                    if "429" in str(e) and attempt < max_retries:  # Rate limit error and retries left
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        tqdm.write(f"[rate_limit] {img_key}: {e} - retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        tqdm.write(f"[api_error] {img_key}: {e}")
                        # API异常时，不武断推断题目问题；保守落地为"模糊"以便后续人工复核
                        q["flags"] = ["AMBIGUOUS_QUESTION"]
                        break
                except Exception as e:
                    # 覆盖 Gemini 的错误与其它未知异常
                    # 也做指数退避（可选）
                    if "429" in str(e) and attempt < max_retries:
                        wait_time = retry_delay * (2 ** attempt)
                        tqdm.write(f"[rate_limit?] {img_key}: {e} - retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    tqdm.write(f"[api_unknown_error] {img_key}: {e}")
                    q["flags"] = ["AMBIGUOUS_QUESTION"]
                    break

            # If we failed to get completion after retries, skip this question
            if completion is None:
                continue

            if args.model == 'gemini-2.5-pro':
                text = completion.text
            else:
                # extract text
                if isinstance(completion, dict):
                    try:
                        text = completion["choices"][0]["message"]["content"]
                    except Exception:
                        text = str(completion)
                else:
                    text = str(completion)

            # parse SINGLE LETTER and map to option
            L = extract_letter(text, len(all_options))
            
            if L is not None:
                idx = LETTER_ALPH.index(L)
                model_chosen_text = all_options[idx]
                q["flags"] = [model_chosen_text]
            else:
                # Model didn't provide valid answer
                q["flags"] = ["AMBIGUOUS_QUESTION"]

        # After processing all questions for this image, classify them into good/bad
        for q in data[img_key]:
            flags = q.get("flags", [])
            if flags and len(flags) > 0:
                flag_content = flags[0]
                if flag_content in SPECIAL_FLAGS:
                    # Special flag → bad
                    bad.setdefault(img_key, []).append(q)
                else:
                    # Model's answer → check if correct
                    expected_answer = str(q.get("answer", "")).strip()
                    if flag_content.strip() == expected_answer:
                        good.setdefault(img_key, []).append(q)
                    else:
                        bad.setdefault(img_key, []).append(q)
            else:
                # No flags → treat as bad (shouldn't happen with current logic)
                bad.setdefault(img_key, []).append(q)
        
        # Save results after each image
        save_results()
        tqdm.write(f"[saved] {img_key}: {len(good.get(img_key, []))} good, {len(bad.get(img_key, []))} bad questions")

    # Final count for summary
    good_n = sum(len(questions) for questions in good.values())
    bad_n = sum(len(questions) for questions in bad.values())
    total = good_n + bad_n

    print(f"Total: {total} | Good: {good_n} (correct answers) | Bad: {bad_n} (flagged or incorrect) | model={args.model}")

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path",  required=True, help="{image: [q1, q2, ...]}")
    p.add_argument("--output_path", required=True, help="output base; will write *_good.json and *_bad.json")
    p.add_argument("--image_root",  required=True, help="directory containing images; joined with the image key")
    p.add_argument("--model",       required=True, help="model id for AMD_openai_client/AMD_openai_call")
    p.add_argument("--temperature", type=float, default=1.0)
  
    return p.parse_args()

if __name__ == "__main__":
    args = build_args()
    run(args)
