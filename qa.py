"""
QA Evaluation with Captions using Borise/CaptionQA Dataset

Evaluates questions using captions instead of images.
Each question is answered once by an LLM using only the caption as context.

Features:
- Loads questions from Hugging Face dataset (Borise/CaptionQA)
- Requires custom captions (from caption.py)
- Uses Qwen2.5-72B-Instruct for evaluation
- Adds "Cannot answer from the caption" option to non-yes/no questions
- Automatic shuffling of answer choices (with order tracking)

Usage:
    # Evaluate on a specific domain
    python qa.py \
        --caption-path captions.json \
        --output-path results.json \
        --split natural

    # Evaluate on all domains
    python qa.py \
        --caption-path captions.json \
        --output-path results.json \
        --split all
"""

import os
import json
import re
import argparse
import random
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from datasets import load_dataset
from pipeline.api import AMD_vllm_chat_client, AMD_vllm_text_chat_call

LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CANNOT_ANSWER_TEXT = "Cannot answer from the caption"

# Fixed model for evaluation
EVAL_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Available domain splits in the dataset
DOMAIN_SPLITS = ["natural", "document", "ecommerce", "embodiedai"]

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


def normalize_gt_letter(choices: List[str], answer: str) -> Optional[str]:
    """Extract ground truth answer letter from question.
    
    Args:
        choices: List of choice strings
        answer: The correct answer text
    
    Returns:
        Letter corresponding to the correct choice, or None if not found
    """
    if not choices or not isinstance(answer, str):
        return None
    
    # Match answer text to one of the choices
    for i, choice in enumerate(choices):
        if answer.strip() == str(choice).strip():
            return LETTER_ALPH[i]
    
    return None


def is_yesno_question(question_text: str, choices: List[str]) -> bool:
    """
    Check if question is a yes/no question.
    
    A question is considered yes/no if:
    1. The choices contain "Yes" and "No" (in any order, possibly with other choices), OR
    2. The question starts with common yes/no question words (is/are, do/does/did, 
       have/has, can/could, will/would, should)
    """
    # Check if choices contain yes and no
    choice_texts = [str(c).strip().lower() for c in choices]
    
    has_yes = any("yes" in choice for choice in choice_texts)
    has_no = any("no" in choice for choice in choice_texts)
    
    if has_yes and has_no:
        return True
    
    # Check if question starts with yes/no question words
    question_lower = question_text.strip().lower()
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


def add_cannot_answer_option(question_text: str, choices: List[str]) -> List[str]:
    """Add 'cannot answer from the caption' option to non-yes/no questions."""
    if is_yesno_question(question_text, choices):
        return choices
    
    return choices + [CANNOT_ANSWER_TEXT]


def build_caption_qa_prompt(caption: str, question: str, choices: List[str]) -> str:
    """Build prompt with caption and question."""
    lines = [f"{LETTER_ALPH[i]}. {choice}" for i, choice in enumerate(choices)]
    
    prompt = f"""Caption:
{caption}

Question:
{question}

Options:
{chr(10).join(lines)}

Answer:"""
    
    return prompt


def load_captionqa_dataset(dataset_name: str, split: str):
    """
    Load CaptionQA dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset name (default: "Borise/CaptionQA")
        split: Domain split to use: "natural", "document", "ecommerce", "embodiedai", or "all"
    
    Returns:
        HuggingFace dataset object
    """
    print(f"Loading dataset {dataset_name} (split: {split})...")
    
    # Load the specified split directly (including "all" which combines all domains)
    dataset = load_dataset(dataset_name, split=split)
    
    print(f"Loaded {len(dataset)} entries")
    return dataset


# ---------- Main Evaluation Function ----------

def evaluate_qa_with_captions(args):
    """
    Evaluate questions using captions instead of images.
    Each question is answered once with shuffled choices.
    """
    
    # Load dataset from HuggingFace
    dataset = load_captionqa_dataset(args.dataset, args.split)
    
    # Load captions (required)
    print(f"Loading captions from {args.caption_path}...")
    with open(args.caption_path, "r", encoding="utf-8") as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} captions")
    
    # Initialize vLLM client for Qwen2.5-72B-Instruct
    print(f"Using model: {EVAL_MODEL}")
    client = AMD_vllm_chat_client(model=EVAL_MODEL)
    
    # Setup RNG for shuffling
    rng = random.Random(args.seed)
    
    # Prepare questions
    print("Preparing questions...")
    prompts: List[str] = []
    meta: List[tuple] = []  # (image_id, q_idx, perm, n_opts, gt_idx_orig, original_question_data)
    
    skipped_no_caption = 0
    skipped_no_choices = 0
    
    for entry in dataset:
        # Get image identifier (e.g., "nat_001", "doc_042")
        image_id = entry.get('id')
        if image_id is None:
            continue
        
        # Use id directly as caption lookup key
        image_key = str(image_id)
        
        # Get caption (required)
        if image_key not in captions:
            skipped_no_caption += 1
            continue
        caption = captions[image_key]
        
        # Get questions for this entry
        questions = entry.get('questions', [])
        if not questions:
            # Single question format
            if 'question' in entry:
                cat = entry.get('category', [])
                if isinstance(cat, list):
                    cat = cat[0] if cat else ''
                questions = [{
                    'question': entry['question'],
                    'choices': entry.get('choices', []),
                    'answer': entry.get('answer'),
                    'category': cat
                }]
            else:
                continue
        
        for q_idx, q in enumerate(questions):
            question_text = q.get('question', '')
            choices = q.get('choices', [])
            answer = q.get('answer')
            # category can be a list or string
            category = q.get('category', [])
            if isinstance(category, list):
                category = category[0] if category else ''
            
            if not choices or len(choices) < 2:
                skipped_no_choices += 1
                continue
            
            # Get original ground truth
            gt_letter_orig = normalize_gt_letter(choices, answer)
            if gt_letter_orig is None:
                continue
            gt_idx_orig = LETTER_ALPH.index(gt_letter_orig)
            
            # Add "cannot answer" option for non-yes/no questions
            choices_with_option = add_cannot_answer_option(question_text, choices)
            
            # Shuffle choices
            n_opts = len(choices_with_option)
            perm = list(range(n_opts))
            rng.shuffle(perm)
            
            # Create shuffled choices
            shuffled_opts = [choices_with_option[i] for i in perm]
            
            prompt = build_caption_qa_prompt(caption, question_text, shuffled_opts)
            
            prompts.append(prompt)
            meta.append((image_key, q_idx, perm, n_opts, gt_idx_orig, {
                'question': question_text,
                'choices': choices,
                'answer': answer,
                'category': category
            }))
    
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

    # Determine which (image_key, q_idx) still need processing
    indices_to_process = []
    for i, (image_key, q_idx, _perm, _n_opts, _gt_idx_orig, _q_data) in enumerate(meta):
        done = processed_count.get(image_key, 0)
        if q_idx >= done:
            indices_to_process.append(i)

    total_remaining = len(indices_to_process)
    print(f"Already processed: {len(prompts) - total_remaining}; remaining: {total_remaining}")

    if total_remaining == 0:
        # Print summary from existing results and exit
        print_final_summary(results)
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # Process in batches and save incrementally
    batch_size = max(1, int(args.save_every))
    for start in range(0, total_remaining, batch_size):
        idxs = indices_to_process[start:start + batch_size]
        batch_prompts = [prompts[i] for i in idxs]

        # Collect responses using vLLM
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

        # Score and append results for this batch
        for resp, i in zip(batch_responses, idxs):
            image_key, q_idx, perm, n_opts, gt_idx_orig, q_data = meta[i]

            letter = extract_letter(resp, n_opts)
            is_correct = False
            is_cannot_answer = False
            model_answer_text = None
            score = 0.0

            original_choices = q_data['choices']
            n_original_choices = len(original_choices)
            choices_with_option = add_cannot_answer_option(q_data['question'], original_choices)

            if letter is not None:
                shuf_idx = LETTER_ALPH.find(letter)
                if 0 <= shuf_idx < len(perm):
                    orig_idx = perm[shuf_idx]

                    if orig_idx < len(choices_with_option):
                        model_answer_text = str(choices_with_option[orig_idx])

                        if model_answer_text == CANNOT_ANSWER_TEXT:
                            is_cannot_answer = True
                            score = (1.0 / n_original_choices) + 0.05
                        elif orig_idx == gt_idx_orig:
                            is_correct = True
                            score = 1.0
                        else:
                            score = 0.0

            if image_key not in results:
                results[image_key] = []

            result_entry = {
                "question": q_data['question'],
                "choices": q_data['choices'],
                "ground_truth": q_data['answer'],
                "model_answer": model_answer_text,
                "model_response": resp,
                "is_correct": is_correct,
                "is_cannot_answer": is_cannot_answer,
                "score": round(score, 4),
                "category": q_data.get('category', '')
            }
            results[image_key].append(result_entry)

        # Save after each batch
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {sum(len(v) for v in results.values())} results -> {args.output_path}")
        
        # Print running totals
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

    # Final summary
    print_final_summary(results)


def print_final_summary(results: Dict[str, List]):
    """Print final evaluation summary."""
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
    print(f"Model: {EVAL_MODEL}")
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
        description="Evaluate questions using captions with Borise/CaptionQA dataset (using Qwen2.5-72B-Instruct)"
    )
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="Borise/CaptionQA",
                       help="HuggingFace dataset name (default: Borise/CaptionQA)")
    parser.add_argument("--split", type=str, default="all",
                       choices=["natural", "document", "ecommerce", "embodiedai", "all"],
                       help="Domain split to evaluate (default: all)")
    
    # Caption source (required)
    parser.add_argument("--caption-path", type=str, required=True,
                       help="Path to caption JSON file ({img_key: caption})")
    
    # Output
    parser.add_argument("--output-path", type=str, required=True,
                       help="Path to save evaluation results")
    
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
    
    evaluate_qa_with_captions(args)


if __name__ == "__main__":
    main()
