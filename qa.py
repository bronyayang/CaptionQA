"""
QA Evaluation with Captions

Evaluates questions using captions instead of images.
Each question is answered once by an LLM using only the caption as context.

Features:
- Adds "Cannot answer from the caption" option to non-yes/no questions
- Automatic shuffling of answer choices (with order tracking)
- Support for multiple model backends (OpenAI, Gemini, vLLM)

Usage:
    python qa.py \
        --caption-path captions.json \
        --question-path document_gpt4o_level1.json \
        --output-path results.json \
        --model gpt-4o
"""

import os
import json
import re
import argparse
import random
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from pipeline.api import (
    AMD_openai_client, AMD_openai_call,
    AMD_gemini_client, AMD_gemini_call,
    AMD_vllm_chat_client, AMD_vllm_text_chat_call
)

LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CANNOT_ANSWER_TEXT = "Cannot answer from the caption"

# ---------- Helper Functions ----------

def extract_letter(answer_text: str, num_options: int) -> Optional[str]:
    """Extract answer letter from model output."""
    if not answer_text:
        return None
    
    # If response contains </think>, extract letter from text after it
    if "</think>" in answer_text:
        after_think = answer_text.split("</think>", 1)[1]
        answer_text = after_think

    if "Answer: " in answer_text:
        after_answer = answer_text.split("Answer: ", 1)[1]
        answer_text = after_answer

    if "\n" in answer_text:
        after_n = answer_text.split("\n", 1)[1]
        answer_text = after_n
    
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
    """Extract ground truth answer letter from question.
    
    Expects format: {"choices": [...], "answer": "exact option text"}
    """
    choices = q.get("choices", [])
    answer = q.get("answer")
    
    if not choices or not isinstance(answer, str):
        return None
    
    # Match answer text to one of the choices
    for i, choice in enumerate(choices):
        if answer.strip() == str(choice).strip():
            return LETTER_ALPH[i]
    
    return None


def is_yesno_question(q: Dict[str, Any]) -> bool:
    """
    Check if question is a yes/no question.
    
    A question is considered yes/no if:
    1. The choices contain "Yes" and "No" (in any order, possibly with other choices), OR
    2. The question starts with common yes/no question words (is/are, do/does/did, 
       have/has, can/could, will/would, should)
    """
    choices = q.get("choices", [])
    question_text = q.get("question", "").strip()
    
    # Check if choices contain yes and no
    choice_texts = []
    for choice in choices:
        text = choice.get("text") if isinstance(choice, dict) else str(choice)
        choice_texts.append(text.strip().lower())
    
    has_yes = any("yes" in choice for choice in choice_texts)
    has_no = any("no" in choice for choice in choice_texts)
    
    if has_yes and has_no:
        return True
    
    # Check if question starts with yes/no question words
    question_lower = question_text.lower()
    yesno_starters = [
        "is ", "are ", "was ", "were ",
        "do ", "does ", "did ",
        "have ", "has ", "had ",
        "can ", "could ",
        "will ", "would ",
        "should ", "shall ",
        "may ", "might ", "must "
    ]
    
    for starter in yesno_starters:
        if question_lower.startswith(starter):
            return True
    
    return False


def add_cannot_answer_option(q: Dict[str, Any]) -> Dict[str, Any]:
    """Add 'cannot answer from the caption' option to non-yes/no questions."""
    if is_yesno_question(q):
        return q
    
    q_copy = dict(q)
    choices = q.get("choices", [])
    q_copy["choices"] = choices + [CANNOT_ANSWER_TEXT]
    
    return q_copy


def build_caption_qa_prompt(caption: str, q: Dict[str, Any]) -> str:
    """Build prompt with caption and question."""
    question = q["question"]
    choices = q.get("choices", [])
    
    lines = []
    for idx, choice in enumerate(choices):
        letter = LETTER_ALPH[idx]
        text = choice.get("text") if isinstance(choice, dict) else str(choice)
        lines.append(f"{letter}. {text}")
    
    prompt = f"""Caption:
{caption}

Question:
{question}

Options:
{chr(10).join(lines)}

Answer:"""
    
    return prompt




def detect_model_backend(model: str) -> str:
    """Detect which API backend to use based on model name."""
    model_lower = model.lower()
    if 'gemini' in model_lower:
        return 'gemini'
    elif any(vllm_model in model_lower for vllm_model in ['qwen', 'llama', 'mistral', 'phi']):
        return 'vllm'
    else:
        return 'openai'


def call_text_model(
    client: Any,
    backend: str,
    model: str,
    prompt: str,
    max_tokens: int = 4
) -> Optional[str]:
    """Call text model with appropriate backend."""
    try:
        if backend == 'gemini':
            completion = AMD_gemini_call(
                client,
                model,
                messages=prompt,
                image_paths=[],  # No images for text-only
                temperature=1.0
            )
            return completion.text.strip()
        
        elif backend == 'vllm':
            result = AMD_vllm_text_chat_call(
                client,
                prompt,
                temperature=0.0,
                max_tokens=max_tokens
            )
            if isinstance(result, list) and len(result) > 0:
                return result[0].strip()
            return None
        
        else:  # OpenAI backend
            messages = [{"role": "user", "content": prompt}]
            completion = AMD_openai_call(
                client,
                model,
                messages=messages,
                temperature=1.0,
                stream=False,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"[model_error] {e}")
        return None


# ---------- Main Evaluation Function ----------

def evaluate_qa_with_captions(args):
    """
    Evaluate questions using captions instead of images.
    Each question is answered once with shuffled choices.
    """
    
    # Load captions
    print(f"Loading captions from {args.caption_path}...")
    with open(args.caption_path, "r", encoding="utf-8") as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} captions")
    
    # Load questions
    print(f"Loading questions from {args.question_path}...")
    with open(args.question_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    print(f"Loaded questions for {len(questions_data)} images")
    
    # Initialize model client
    backend = detect_model_backend(args.model)
    print(f"Using {backend} backend for model {args.model}")
    
    if backend == 'gemini':
        client = AMD_gemini_client()
    elif backend == 'vllm':
        client = AMD_vllm_chat_client(model=args.model)
    else:
        client = AMD_openai_client(model_id=args.model)
    
    # Setup RNG for shuffling
    rng = random.Random(args.seed)
    
    # Prepare questions
    print("Preparing questions...")
    prompts: List[str] = []
    meta: List[tuple] = []  # (img_key, q_idx, perm, n_opts, gt_idx_orig)
    
    images = list(questions_data.keys())
    skipped_no_caption = 0
    skipped_no_choices = 0
    
    for img_key in images:
        # Check if we have caption for this image
        if img_key not in captions:
            skipped_no_caption += 1
            continue
        
        caption = captions[img_key]
        
        for q_idx, q in enumerate(questions_data[img_key]):
            # Add "cannot answer" option for non-yes/no questions
            q_with_option = add_cannot_answer_option(q)
            
            choices = q_with_option.get("choices", [])
            if not choices or len(choices) < 2:
                skipped_no_choices += 1
                continue
            
            # Get original ground truth
            gt_letter_orig = normalize_gt_letter(q)
            if gt_letter_orig is None:
                continue
            gt_idx_orig = LETTER_ALPH.index(gt_letter_orig)
            
            # Shuffle choices
            n_opts = len(choices)
            perm = list(range(n_opts))
            rng.shuffle(perm)
            
            # Create shuffled question
            q_shuffled = dict(q_with_option)
            shuffled_opts = [choices[i] for i in perm]
            q_shuffled["choices"] = shuffled_opts
            
            prompt = build_caption_qa_prompt(caption, q_shuffled)
            
            prompts.append(prompt)
            meta.append((img_key, q_idx, perm, n_opts, gt_idx_orig))
    
    print(f"Prepared {len(prompts)} questions")
    print(f"Skipped: {skipped_no_caption} (no caption), {skipped_no_choices} (no choices)")
    
    if not prompts:
        print("No questions to evaluate!")
        return
    
    # Incremental saving and resume
    system_prompt = "You are given a caption describing an image, and a question about the image. Answer with a SINGLE LETTER (A, B, C, ...), no explanation."

    # Load existing results if present (auto-resume)
    results = {}
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    try:
        with open(args.output_path, "r", encoding="utf-8") as f:
            results = json.load(f) or {}
        print(f"Loaded existing results from {args.output_path} (resume mode)")
    except Exception as e:
        print(f"Warning: could not load existing results ({e}); starting fresh")
        results = {}

    # Map image -> already processed count
    processed_count = {k: len(v) for k, v in results.items() if isinstance(v, list)}

    # Print intermediate totals from loaded results
    if results:
        existing_total = sum(len(v) for v in results.values())
        existing_total_score = sum(sum(item.get("score", 0.0) for item in v) for v in results.values())
        existing_correct = sum(
            sum(1 for item in v if item.get("is_correct")) for v in results.values()
        )
        existing_cannot = sum(
            sum(1 for item in v if item.get("is_cannot_answer")) for v in results.values()
        )
        existing_avg = (existing_total_score / existing_total) if existing_total else 0.0
        existing_acc = (existing_correct / existing_total) if existing_total else 0.0
        print(
            f"[resume] loaded={existing_total} | total_score={existing_total_score:.2f} "
            f"| avg_score={existing_avg:.4f} | accuracy={existing_acc:.2%} | cannot_answer={existing_cannot}"
        )

    # Determine which (img_key, q_idx) still need processing
    indices_to_process = []
    for i, (img_key, q_idx, _perm, _n_opts, _gt_idx_orig) in enumerate(meta):
        done = processed_count.get(img_key, 0)
        if q_idx >= done:
            indices_to_process.append(i)

    total_remaining = len(indices_to_process)
    already_done = sum(processed_count.get(k, 0) for k in images)
    print(f"Already processed: {already_done}; remaining: {total_remaining}")

    if total_remaining == 0:
        # Print summary from existing results and exit
        total_questions = sum(len(v) for v in results.values())
        total_score = sum(sum(item.get("score", 0.0) for item in v) for v in results.values())
        correct_answers = sum(
            sum(1 for item in v if item.get("is_correct")) for v in results.values()
        )
        cannot_answer_count = sum(
            sum(1 for item in v if item.get("is_cannot_answer")) for v in results.values()
        )
        overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        average_score = total_score / total_questions if total_questions > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"Evaluation Results:")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Total questions: {total_questions}")
        print(f"Correct answers: {correct_answers} ({overall_accuracy:.2%})")
        print(f"'Cannot answer' selections: {cannot_answer_count}")
        print(f"Total score: {total_score:.2f} / {total_questions}")
        print(f"Average score: {average_score:.4f}")
        print(f"{'='*60}")
        print(f"\nScoring rules:")
        print(f"  - Correct answer: 1.0 point")
        print(f"  - Incorrect answer: 0.0 points")
        print(f"  - 'Cannot answer': 1/n_choices + 0.05 points")
        print(f"{'='*60}")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # Process in batches and save incrementally
    batch_size = max(1, int(args.save_every))
    for start in range(0, total_remaining, batch_size):
        idxs = indices_to_process[start:start + batch_size]
        batch_prompts = [prompts[i] for i in idxs]

        # Collect responses for this batch
        batch_responses: List[str] = []
        if backend == 'vllm':
            outs = AMD_vllm_text_chat_call(
                client,
                batch_prompts,
                temperature=0.0,
                max_tokens=args.max_tokens,
                n=1,
                return_all=False,
                use_tqdm=False,
                system=system_prompt,
            )
            if outs and isinstance(outs, list) and len(outs) > 0 and isinstance(outs[0], list):
                batch_responses = [lst[0] if lst else "" for lst in outs]
            else:
                batch_responses = [o if isinstance(o, str) else "" for o in (outs or [])]
        else:
            for p in tqdm(batch_prompts, desc=f"Evaluating {start+1}-{start+len(idxs)}/{total_remaining}"):
                full_prompt = f"{system_prompt}\n\n{p}"
                r = call_text_model(
                    client, backend, args.model, full_prompt,
                    max_tokens=args.max_tokens
                )
                batch_responses.append(r or "")

        # Score and append results for this batch
        for resp, i in zip(batch_responses, idxs):
            img_key, q_idx, perm, n_opts, gt_idx_orig = meta[i]

            letter = extract_letter(resp, n_opts)
            is_correct = False
            is_cannot_answer = False
            model_answer_text = None
            score = 0.0

            if letter is not None:
                shuf_idx = LETTER_ALPH.find(letter)
                if 0 <= shuf_idx < len(perm):
                    orig_idx = perm[shuf_idx]

                    q = questions_data[img_key][q_idx]
                    q_with_option = add_cannot_answer_option(q)
                    choices = q_with_option.get("choices", [])
                    original_choices = q.get("choices", [])
                    n_original_choices = len(original_choices)

                    if orig_idx < len(choices):
                        choice = choices[orig_idx]
                        model_answer_text = choice.get("text") if isinstance(choice, dict) else str(choice)

                        if model_answer_text == CANNOT_ANSWER_TEXT:
                            is_cannot_answer = True
                            score = (1.0 / n_original_choices) + 0.05
                        elif orig_idx == gt_idx_orig:
                            is_correct = True
                            score = 1.0
                        else:
                            score = 0.0

            if img_key not in results:
                results[img_key] = []

            q = questions_data[img_key][q_idx]
            result_entry = {
                "question": q["question"],
                "choices": q.get("choices", []),
                "ground_truth": q.get("answer"),
                "model_answer": model_answer_text,
                "model_response": resp,
                "is_correct": is_correct,
                "is_cannot_answer": is_cannot_answer,
                "score": round(score, 4),
                "category": q.get("category", "")
            }
            results[img_key].append(result_entry)

        # Save after each batch
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {sum(len(v) for v in results.values())} results -> {args.output_path}")
        # Print running totals including loaded + this batch
        running_total = sum(len(v) for v in results.values())
        running_total_score = sum(sum(item.get("score", 0.0) for item in v) for v in results.values())
        running_correct = sum(
            sum(1 for item in v if item.get("is_correct")) for v in results.values()
        )
        running_cannot = sum(
            sum(1 for item in v if item.get("is_cannot_answer")) for v in results.values()
        )
        running_avg = (running_total_score / running_total) if running_total else 0.0
        running_acc = (running_correct / running_total) if running_total else 0.0
        print(
            f"[progress] processed={running_total} | total_score={running_total_score:.2f} "
            f"| avg_score={running_avg:.4f} | accuracy={running_acc:.2%} | cannot_answer={running_cannot}"
        )
        # if running_total >= 500:
        #     break

    # Final summary computed from all saved results
    total_questions = sum(len(v) for v in results.values())
    total_score = sum(sum(item.get("score", 0.0) for item in v) for v in results.values())
    correct_answers = sum(
        sum(1 for item in v if item.get("is_correct")) for v in results.values()
    )
    cannot_answer_count = sum(
        sum(1 for item in v if item.get("is_cannot_answer")) for v in results.values()
    )
    overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    average_score = total_score / total_questions if total_questions > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {correct_answers} ({overall_accuracy:.2%})")
    print(f"'Cannot answer' selections: {cannot_answer_count}")
    print(f"Total score: {total_score:.2f} / {total_questions}")
    print(f"Average score: {average_score:.4f}")
    print(f"{'='*60}")
    print(f"\nScoring rules:")
    print(f"  - Correct answer: 1.0 point")
    print(f"  - Incorrect answer: 0.0 points")
    print(f"  - 'Cannot answer': 1/n_choices + 0.05 points")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate questions using captions instead of images"
    )
    
    # Input/Output
    parser.add_argument("--caption-path", type=str, required=True,
                       help="Path to caption JSON file ({img_key: caption})")
    parser.add_argument("--question-path", type=str, required=True,
                       help="Path to question JSON file ({img_key: [{question, choices, answer}, ...]})")
    parser.add_argument("--output-path", type=str, required=True,
                       help="Path to save evaluation results")
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True,
                       help="Model to use for evaluation (e.g., gpt-4o, llama-3.1-8b-instruct)")
    
    # Evaluation parameters
    parser.add_argument("--max-tokens", type=int, default=4,
                       help="Maximum tokens for response (default: 4)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for option shuffling (default: 0)")
    parser.add_argument("--save-every", type=int, default=50,
                       help="Save incremental results every N questions (default: 50)")
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.caption_path):
        print(f"Error: Caption file {args.caption_path} does not exist")
        return
    
    if not os.path.exists(args.question_path):
        print(f"Error: Question file {args.question_path} does not exist")
        return
    
    evaluate_qa_with_captions(args)


if __name__ == "__main__":
    main()
