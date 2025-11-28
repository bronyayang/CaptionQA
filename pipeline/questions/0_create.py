import argparse
import os
import json
from tqdm import tqdm
import time
import random
import openai
import re
from typing import Tuple, Optional, Any
from pipeline.utils import load_json, encode_image
from pipeline.api import AMD_openai_client, AMD_openai_call

def generate_prompt(big_cat, sub_cat=None, q=None, examples=None):
    """
    Helper function to generate the prompt string based on provided parameters.
    - big_cat: The big category (required)
    - sub_cat: The subcategory (optional)
    - q: The specific question (optional)
    - examples: Examples to include in the prompt (optional)
    """
    # Base header for the prompt
    header = "You are given an image or images. Your task is to create multiple-choice questions (with answers) based on the "
    if q is not None:
        header += f'question "{q}" within the '
    if sub_cat is not None:
        header += f'subcategory "{sub_cat}" within the '
    header += f'big category "{big_cat}".'

    # Focus instructions: either with examples or not
    if examples:
        focus = f' Only create questions if they pertain to "{sub_cat or q}" (e.g., {examples}) with the image.'
    else:
        focus = f' Only create questions if they pertain to "{sub_cat or q or big_cat}" with the image.'

    prompt = f"""
{header}

1.{focus}
2. If no valid question can be formed based on this focus, return None.
3. Format the output as a JSON list of objects, where each object contains the following keys:
    - "question": The text of the question.
    - "choices": A list of strings representing the answer options.
    - "answer": The correct answer from the choices.
4. If multiple questions are generated, include them all in the list.
    """.strip()
    return prompt


def create_prompt(taxonomy, level=0):
    if not isinstance(taxonomy, dict):
        return []

    prompts = []
    cats = []

    if level == 0:
        for t, sub_tax in taxonomy.items():
            examples = ', '.join(sub_tax.keys()) if sub_tax else None
            prompts.append(generate_prompt(big_cat=t, examples=examples))
            cats.append(t)

    elif level == 1:
        for t, sub_tax in taxonomy.items():
            if not sub_tax:
                prompts.append(generate_prompt(big_cat=t))
                cats.append(t)
            else:
                for s, next_level in sub_tax.items():
                    examples = None
                    if next_level:
                        examples = ', '.join(next_level.keys()) if isinstance(next_level, dict) else ', '.join(next_level)
                    prompts.append(generate_prompt(big_cat=t, sub_cat=s, examples=examples))
                    cats.append(f"{t} - {s}")

    elif level == 2:
        for t, sub_tax in taxonomy.items():
            # Always add prompt for big category t if subcategories exist
            if not sub_tax:
                prompts.append(generate_prompt(big_cat=t))
            for s, next_level in sub_tax.items():
                if not next_level:
                    prompts.append(generate_prompt(big_cat=t, sub_cat=s))
                    cats.append(f"{t} - {s}")
                elif isinstance(next_level, list):
                    examples = ', '.join(next_level)
                    prompts.append(generate_prompt(big_cat=t, sub_cat=s, examples=examples))
                    cats.append(f"{t} - {s}")
                else:
                    for q, value in next_level.items():
                        examples = ', '.join(value) if value else None
                        prompts.append(generate_prompt(big_cat=t, sub_cat=s, q=q, examples=examples))
                        cats.append(f"{t} - {s} - {q}")
    return prompts, cats


def parse_with_status(response: str) -> Tuple[Optional[Any], str]:
    """
    Returns (data, status) where status in {"ok", "json_error", "no_json"}.
    Mirrors your parse() logic but gives a machine-checkable status.
    """
    m = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
    if not m:
        return None, "no_json"
    json_str = m.group(1)
    try:
        return json.loads(json_str), "ok"
    except json.JSONDecodeError:
        return None, "json_error"


def _sleep_backoff(attempt: int, base: float, factor: float, jitter: float) -> None:
    time.sleep(base * (factor ** attempt) + random.uniform(0, max(0.0, jitter)))


def create_questions(args):
    taxonomy = load_json(args.taxonomy)
    prompts, cats = create_prompt(taxonomy, args.level)
    # get the absolute path of the images
    image_root_path = os.path.abspath(args.benchmark_folder)
    
    # Discover both single images and image folders
    input_items = []  # List of tuples: (key, image_paths, is_multi)
    
    for item in os.listdir(image_root_path):
        item_path = os.path.join(image_root_path, item)
        
        if os.path.isfile(item_path) and item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Single image file
            input_items.append((item, [item_path], False))
        elif os.path.isdir(item_path):
            # Directory containing multiple images
            folder_images = []
            for img_file in os.listdir(item_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    folder_images.append(os.path.join(item_path, img_file))
            
            if folder_images:  # Only add if folder contains images
                folder_images.sort()  # Sort images within folder
                input_items.append((item, folder_images, True))
    
    input_items.sort(key=lambda x: x[0])  # Sort by key name
    
    # Print summary of discovered items
    print(f"Discovered {len(input_items)} items to process:")
    for key, image_paths, is_multi in input_items:
        if is_multi:
            print(f"  📁 {key}: {len(image_paths)} images (multi-image folder)")
        else:
            print(f"  🖼️  {key}: single image")

    client = AMD_openai_client(model_id=args.model)

    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            benchmark = json.load(f)
        print("Loaded previous")
    else:
        benchmark = {}
        
    for key, image_paths, is_multi in tqdm(input_items, desc="Processing Items"):
        # Skip items that are already processed
        if key in benchmark:
            print(f"Skipping {key} (already processed)")
            continue
            
        benchmark[key] = []
        
        # Encode all images for this item
        encoded_images = [encode_image(img_path) for img_path in image_paths]
        
        for p, c in tqdm(zip(prompts, cats), desc="Processing Prompts", leave=False):
            # Create content list with all images
            content_items = []
            for encoded_image in encoded_images:
                content_items.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                })
            
            messages = [
                {
                    "role": "user",
                    "content": content_items,
                },
                {"role": "user", "content": p},
            ]

            # First attempt: call once, no internal retries (we control retries only on JSON parse error)
            try:
                completion = AMD_openai_call(
                    client,
                    args.model,
                    messages=messages,
                    temperature=1.0,
                    stream=False,
                    reasoning_effort="low"
                )
            except openai.OpenAIError as e:
                tqdm.write(f"[api_error] {key} / cat={c}: {e}")
                continue
            except Exception as e:
                tqdm.write(f"[api_unknown_error] {key} / cat={c}: {e}")
                continue

            raw = completion.choices[0].message.content
            data, status = parse_with_status(raw)

            if status == "ok":
                questions = data
            elif status == "no_json":
                tqdm.write(f"[no_json] {key} / cat={c} -> skipping")
                continue
            else:  # "json_error" -> retry the API call only for this case
                success = False
                for attempt in range(args.json_retries):
                    _sleep_backoff(attempt, args.backoff_base, args.backoff_factor, args.jitter)
                    try:
                        completion = AMD_openai_call(
                            client,
                            args.model,
                            messages=messages,
                            temperature=1.0,
                            stream=False,
                            reasoning_effort="low"
                        )
                    except openai.OpenAIError as e:
                        tqdm.write(f"[api_error_retry] {key} / cat={c} (attempt {attempt+1}): {e}")
                        # per your rule: on API error, just stop retrying and continue to next prompt
                        success = False
                        break
                    except Exception as e:
                        tqdm.write(f"[api_unknown_error_retry] {key} / cat={c} (attempt {attempt+1}): {e}")
                        success = False
                        break

                    raw = completion.choices[0].message.content
                    data2, status2 = parse_with_status(raw)
                    if status2 == "ok":
                        questions = data2
                        success = True
                        break
                    elif status2 == "no_json":
                        # per your rule: no retry on "no JSON"; stop retrying and continue
                        tqdm.write(f"[no_json_after_retry] {key} / cat={c} -> skipping")
                        success = False
                        break
                    else:
                        # still JSON decoding error; loop to next retry attempt
                        continue

                if not success:
                    # Could not recover valid JSON — skip this prompt
                    continue

            # Normalize questions to a list
            if isinstance(questions, dict):
                questions = [questions]

            for q in questions or []:
                q["category"] = c
                benchmark[basename].append(q)
                
        with open(args.output_path, 'w') as f:
            json.dump(benchmark, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-folder", type=str, default="document")
    parser.add_argument("--taxonomy", type=str, default="document_taxonomy_v0.json")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--output-path", type=str, default="question.json")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--json-retries", type=int, default=2,
                    help="When JSON parsing fails, how many times to retry the API call")
    parser.add_argument("--backoff-base", type=float, default=0.5, help="Initial backoff seconds")
    parser.add_argument("--backoff-factor", type=float, default=2.0, help="Backoff growth factor")
    parser.add_argument("--jitter", type=float, default=0.25, help="Random jitter added to sleep")

    args = parser.parse_args()
    create_questions(args)