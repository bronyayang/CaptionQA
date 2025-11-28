#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import os
from typing import Dict, Any, List

SPECIAL_FLAGS = [
    "AMBIGUOUS_QUESTION",
    "UNANSWERABLE_FROM_IMAGE",
    "NOT_SUITABLE_FOR_CAPTION_EVAL",
    "NONE_OF_THE_ABOVE"
]

def normalize_flag(flag):
    """Normalize flags - if it's a special flag, return a common marker"""
    if flag in SPECIAL_FLAGS:
        return "SPECIAL_FLAG"
    return flag

def build_index(data: Dict[str, List[Dict[str, Any]]]) -> Dict[tuple, Dict[str, Any]]:
    """
    Build index for file data: (image_key, question_text, normalized_flag) -> QA dict
    """
    idx = {}
    for img, qlist in data.items():
        for qa in qlist:
            question = str(qa.get("question", ""))
            flags = qa.get("flags", [])
            
            # Only process if flags has exactly 1 item
            if isinstance(flags, list) and len(flags) == 1:
                flag = normalize_flag(flags[0])
                key = (img, question, flag)
                if key not in idx:
                    idx[key] = qa
            
    return idx

def process(file1: Dict[str, List[Dict[str, Any]]], 
           file2: Dict[str, List[Dict[str, Any]]]) -> tuple:
    """
    Process two files and return (accepted_items, review_items, discarded_items)
    accepted_items: intersection of file1 and file2 with same non-special flags
    review_items: items from file1 and file2 that don't match
    discarded_items: items where both files have special flags for the same question
    """
    # Build index from file1 for matching
    idx1 = build_index(file1)
    
    accepted: Dict[str, List[Dict[str, Any]]] = {}
    review: Dict[str, List[Dict[str, Any]]] = {}
    discarded: Dict[str, List[Dict[str, Any]]] = {}
    
    # Track which keys from idx1 have been matched
    matched_keys_from_idx1 = set()
    
    # Process file2 items
    for img, qlist2 in file2.items():
        accepted_list: List[Dict[str, Any]] = []
        review_list: List[Dict[str, Any]] = []
        discarded_list: List[Dict[str, Any]] = []
        
        for qa2 in qlist2:
            question = str(qa2.get("question", ""))
            flags2 = qa2.get("flags", [])
            
            # Check if flags has exactly 1 item
            if isinstance(flags2, list) and len(flags2) == 1:
                flag2 = normalize_flag(flags2[0])
                key = (img, question, flag2)
                
                # Check if this exact combination exists in file1
                if key in idx1:
                    # Check if both are special flags
                    original_flag2 = flags2[0]
                    qa1 = idx1[key]
                    flags1 = qa1.get("flags", [])
                    
                    if isinstance(flags1, list) and len(flags1) == 1:
                        original_flag1 = flags1[0]
                        
                        # If both are special flags, discard
                        if original_flag1 in SPECIAL_FLAGS and original_flag2 in SPECIAL_FLAGS:
                            discarded_list.append(qa2)
                        else:
                            # Both have same normalized flag and at least one is not special
                            accepted_list.append(qa2)
                    else:
                        accepted_list.append(qa2)
                    
                    matched_keys_from_idx1.add(key)
                else:
                    review_list.append(qa2)
            else:
                # If flags doesn't have exactly 1 item, put in review
                review_list.append(qa2)
        
        if accepted_list:
            if img not in accepted:
                accepted[img] = []
            accepted[img].extend(accepted_list)
        if review_list:
            if img not in review:
                review[img] = []
            review[img].extend(review_list)
        if discarded_list:
            if img not in discarded:
                discarded[img] = []
            discarded[img].extend(discarded_list)
    
    # Add unmatched items from file1 to review
    for key, qa1 in idx1.items():
        if key not in matched_keys_from_idx1:
            img = key[0]  # Extract image name from key
            if img not in review:
                review[img] = []
            review[img].append(qa1)
    
    return accepted, review, discarded

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file1", required=True, help="Path to file1.json")
    ap.add_argument("--file2", required=True, help="Path to file2.json")
    args = ap.parse_args()
    
    # Load both files
    with open(args.file1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(args.file2, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    
    # Process the files
    accepted, review, discarded = process(data1, data2)
    
    # Generate output file names based on file1 name
    file1_basename = os.path.splitext(os.path.basename(args.file1))[0]
    accept_output = f"{file1_basename}_accept.json"
    review_output = f"{file1_basename}_review.json"
    discard_output = f"{file1_basename}_discard.json"
    
    # Save results
    with open(accept_output, "w", encoding="utf-8") as f:
        json.dump(accepted, f, indent=4, ensure_ascii=False)
    
    with open(review_output, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=4, ensure_ascii=False)
    
    with open(discard_output, "w", encoding="utf-8") as f:
        json.dump(discarded, f, indent=4, ensure_ascii=False)
    
    print(f"Accepted items saved to: {accept_output}")
    print(f"Review items saved to: {review_output}")
    print(f"Discarded items saved to: {discard_output}")
    print(f"Accepted (intersection with non-special flags): {sum(len(qlist) for qlist in accepted.values())} questions")
    print(f"Review (non-matching from both files): {sum(len(qlist) for qlist in review.values())} questions")
    print(f"Discarded (both files have special flags): {sum(len(qlist) for qlist in discarded.values())} questions")

if __name__ == "__main__":
    main()
