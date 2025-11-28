#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from typing import Dict, Any, List
from collections import defaultdict

def merge_questions(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple questions that are the same.
    Assumes choices and answer are identical, merges flags.
    """
    if not questions:
        return {}
    
    # Use the first question as base
    merged = dict(questions[0])
    
    # Collect all flags from all duplicate questions
    all_flags = []
    for q in questions:
        flags = q.get("flags", [])
        if isinstance(flags, str):
            all_flags.append(flags)
        elif isinstance(flags, list):
            all_flags.extend(flags)
    
    # Remove duplicates while preserving order
    unique_flags = []
    seen = set()
    for flag in all_flags:
        if flag not in seen:
            unique_flags.append(flag)
            seen.add(flag)
    
    merged["flags"] = unique_flags
    return merged

def process_files(file_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process and merge multiple JSON files.
    """
    # Dictionary to track unique questions by (image, question_text)
    question_tracker: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    
    # Load all files and collect questions
    for file_path in file_paths:
        print(f"Loading {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for image_key, qa_list in data.items():
            for question in qa_list:
                question_text = question.get("question", "")
                key = (image_key, question_text)
                question_tracker[key].append(question)
    
    # Merge duplicate questions and build output
    merged_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_questions = 0
    
    for (image_key, question_text), duplicate_questions in question_tracker.items():
        merged_question = merge_questions(duplicate_questions)
        if merged_question:  # Only add non-empty questions
            merged_data[image_key].append(merged_question)
            total_questions += 1
    
    print(f"Total merged questions: {total_questions}")
    return dict(merged_data)

def count_questions_in_file(file_path: str) -> int:
    """
    Count total questions in a single JSON file.
    """
    print(f"Counting questions in {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_questions = 0
    for image_key, qa_list in data.items():
        total_questions += len(qa_list)
    
    print(f"Total questions: {total_questions}")
    return total_questions

def main():
    ap = argparse.ArgumentParser(description="Merge JSON files with same image+question combinations")
    ap.add_argument("--input_files", nargs="+", required=True, help="List of input JSON file paths")
    ap.add_argument("--output_path", help="Output JSON path (required if multiple input files)")
    args = ap.parse_args()
    
    if len(args.input_files) == 1:
        # Single file case - just count questions
        count_questions_in_file(args.input_files[0])
    else:
        # Multiple files case - merge and save
        if not args.output_path:
            raise ValueError("--output_path is required when merging multiple files")
        
        merged_data = process_files(args.input_files)
        
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        
        print(f"Merged data saved to {args.output_path}")

if __name__ == "__main__":
    main()
