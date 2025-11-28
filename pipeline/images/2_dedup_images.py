import os
import imagehash
from PIL import Image
from collections import defaultdict
import shutil
from pathlib import Path

def calculate_image_hash(image_path):
    try:
        with Image.open(image_path) as img:
            # Calculate average hash
            return str(imagehash.average_hash(img))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def find_duplicates(directory, threshold=1):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    
    # Count total images before deduplication
    total_images = sum(1 for f in os.listdir(directory) 
                      if f.lower().endswith(image_extensions))
    print(f"\nTotal images before deduplication: {total_images}")
    
    # Calculate hashes for all images
    image_hashes = {}
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            filepath = os.path.join(directory, filename)
            image_hash = calculate_image_hash(filepath)
            if image_hash:
                image_hashes[filename] = image_hash
    
    # Find duplicates using threshold
    duplicates = defaultdict(list)
    processed = set()
    
    for filename1, hash1 in image_hashes.items():
        if filename1 in processed:
            continue
            
        current_group = [filename1]
        processed.add(filename1)
        
        for filename2, hash2 in image_hashes.items():
            if filename2 in processed:
                continue
                
            # Calculate Hamming distance between hashes
            hash_diff = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
            if hash_diff <= threshold:
                current_group.append(filename2)
                processed.add(filename2)
        
        if len(current_group) > 1:
            group_hash = hash1  # Use first hash as group identifier
            duplicates[group_hash] = current_group
    
    return duplicates, total_images

def main():
    # Directory containing images
    image_dir = "./all_images"
    
    # Create main duplicates directory
    duplicates_dir = os.path.join(image_dir, "duplicates")
    os.makedirs(duplicates_dir, exist_ok=True)
    
    print("Finding duplicate images...")
    duplicates, total_before = find_duplicates(image_dir)
    
    if not duplicates:
        print("No duplicate images found.")
        return
    
    print(f"\nFound {len(duplicates)} groups of duplicate images:\n")
    
    # Process each group of duplicates
    for group_num, (hash_value, files) in enumerate(duplicates.items(), 1):
        # Create a separate folder for this group
        group_dir = os.path.join(duplicates_dir, f"group_{group_num}")
        os.makedirs(group_dir, exist_ok=True)
        
        print(f"Group {group_num}:")
        for file in files:
            src_path = os.path.join(image_dir, file)
            dst_path = os.path.join(group_dir, file)
            print(f"  - {file}")
            print(f"    Moved to {dst_path}")
            shutil.move(src_path, dst_path)
        print()
    
    # Count remaining images
    remaining_images = sum(1 for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')))
    
    print(f"\nSummary:")
    print(f"Total images before deduplication: {total_before}")
    print(f"Images remaining after deduplication: {remaining_images}")
    print(f"Total duplicate images moved: {total_before - remaining_images}")
    print(f"Number of duplicate groups: {len(duplicates)}")

if __name__ == "__main__":
    main() 