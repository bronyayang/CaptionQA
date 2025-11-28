#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, random
from typing import Dict, Any, List
from pipeline.utils import load_json

def sample_questions_random(data: Dict[str, List[Dict[str, Any]]], n: int, seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sample n questions randomly from all questions across all images.
    """
    random.seed(seed)
    
    # Flatten all questions with their image keys
    all_questions = []
    for img_key, questions in data.items():
        for q in questions:
            all_questions.append((img_key, q))
    
    # Sample n questions
    if n >= len(all_questions):
        print(f"Warning: Requested {n} questions, but only {len(all_questions)} available. Taking all.")
        sampled = all_questions
    else:
        sampled = random.sample(all_questions, n)
    
    # Rebuild the structure
    result = {}
    for img_key, q in sampled:
        result.setdefault(img_key, []).append(q)
    
    return result

def sample_questions_per_image(data: Dict[str, List[Dict[str, Any]]], n: int, seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sample n questions per image (or all if image has fewer than n questions).
    """
    random.seed(seed)
    
    result = {}
    total_sampled = 0
    
    for img_key, questions in data.items():
        if n >= len(questions):
            # Take all questions from this image
            result[img_key] = questions.copy()
            total_sampled += len(questions)
        else:
            # Sample n questions from this image
            sampled = random.sample(questions, n)
            result[img_key] = sampled
            total_sampled += n
    
    print(f"Sampled {total_sampled} questions from {len(result)} images")
    return result

def sample_questions_proportional(data: Dict[str, List[Dict[str, Any]]], n: int, seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sample n questions total, distributed proportionally across images based on their question counts.
    """
    random.seed(seed)
    
    # Calculate total questions
    total_questions = sum(len(questions) for questions in data.values())
    
    if n >= total_questions:
        print(f"Warning: Requested {n} questions, but only {total_questions} available. Taking all.")
        return data.copy()
    
    result = {}
    total_sampled = 0
    
    # Calculate proportion for each image
    for img_key, questions in data.items():
        # Calculate how many questions this image should contribute
        proportion = len(questions) / total_questions
        target_count = max(1, int(n * proportion))  # At least 1 question per image
        
        # Don't sample more than available
        sample_count = min(target_count, len(questions))
        
        if sample_count > 0:
            sampled = random.sample(questions, sample_count)
            result[img_key] = sampled
            total_sampled += sample_count
    
    print(f"Sampled {total_sampled} questions proportionally from {len(result)} images")
    return result

def main():
    parser = argparse.ArgumentParser(description="Sample n questions from a JSON file")
    parser.add_argument("--input_path", required=True, help="Path to input JSON file")
    parser.add_argument("--n", type=int, required=True, help="Number of questions to sample")
    parser.add_argument("--strategy", choices=["random", "per_image", "proportional"], 
                        default="random", help="Sampling strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Generate output path automatically
    base_name = os.path.splitext(args.input_path)[0]
    extension = os.path.splitext(args.input_path)[1]
    output_path = f"{base_name}_sample_{args.n}{extension}"
    
    # Load input data
    print(f"Loading data from {args.input_path}")
    print(f"Output will be saved to {output_path}")
    data = load_json(args.input_path)
    
    # Count original questions
    total_original = sum(len(questions) for questions in data.values())
    print(f"Original data: {len(data)} images, {total_original} questions")
    
    # Sample based on strategy
    if args.strategy == "random":
        print(f"Sampling {args.n} questions randomly from all questions")
        sampled_data = sample_questions_random(data, args.n, args.seed)
    elif args.strategy == "per_image":
        print(f"Sampling {args.n} questions per image")
        sampled_data = sample_questions_per_image(data, args.n, args.seed)
    elif args.strategy == "proportional":
        print(f"Sampling {args.n} questions proportionally across images")
        sampled_data = sample_questions_proportional(data, args.n, args.seed)
    
    # Count sampled questions
    total_sampled = sum(len(questions) for questions in sampled_data.values())
    print(f"Sampled data: {len(sampled_data)} images, {total_sampled} questions")
    
    # Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved sampled data to {output_path}")

if __name__ == "__main__":
    main()
