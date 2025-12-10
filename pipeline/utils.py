import json
import base64
from io import BytesIO
import os
import argparse
from pathlib import Path
from PIL import Image

def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def encode_image(image):
    if isinstance(image, str):
        with open(image, 'rb') as image_file:  
            byte_data = image_file.read() 
    else:
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def resize_image_for_api(image_path: str, max_size_bytes: int = 5 * 1024 * 1024) -> str:
    """
    Resize image if needed to fit within API size limits and return base64 encoded string.
    
    Args:
        image_path: Path to the image file
        max_size_bytes: Maximum size in bytes (default 5MB for Claude)
    
    Returns:
        Base64 encoded string of the (possibly resized) image
    """
    # Check original file size
    file_size = os.path.getsize(image_path)
    
    if file_size <= max_size_bytes:
        # File is small enough, just encode it
        return encode_image(image_path)
    
    # Need to resize - load and compress
    img = Image.open(image_path)
    
    # Convert to RGB if necessary (for JPEG encoding)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    # Try progressively smaller sizes until under limit
    quality = 85
    scale = 1.0
    
    while True:
        # Resize if needed
        if scale < 1.0:
            new_size = (int(img.width * scale), int(img.height * scale))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
        else:
            resized = img
        
        # Encode to JPEG
        output_buffer = BytesIO()
        resized.save(output_buffer, format="JPEG", quality=quality)
        byte_data = output_buffer.getvalue()
        
        if len(byte_data) <= max_size_bytes:
            return base64.b64encode(byte_data).decode("utf-8")
        
        # Reduce quality or scale
        if quality > 50:
            quality -= 10
        else:
            scale *= 0.8
        
        if scale < 0.1:
            # Give up and return what we have
            return base64.b64encode(byte_data).decode("utf-8")

def check_question_counts(json_file_path, threshold=40):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Check each image's question count
    low_question_images = []
    for image_name, questions in data.items():
        question_count = len(questions)
        # print(f"Image: {image_name}, Question Count: {question_count}")
        if question_count <= threshold:
            # Extract image number from filename (e.g., "1.jpg" -> "1")
            image_number = os.path.splitext(image_name)[0]
            low_question_images.append((image_number, question_count))
    
    # Sort by image number
    low_question_images.sort(key=lambda x: int(x[0]))
    
    # Print results
    if low_question_images:
        print(f"\nImages with {threshold} or fewer questions:")
        print("Image Number | Question Count")
        print("-" * 30)
        for img_num, count in low_question_images:
            print(f"{img_num:^12} | {count:^14}")
    else:
        print(f"All images have more than {threshold} questions.")

def combine_json_files(folder_path, output_file=None):
    """
    Combine multiple JSON files from a folder by their keys.
    Each JSON file should have the same structure with image names as keys.
    
    Args:
        folder_path (str): Path to the folder containing JSON files
        output_file (str, optional): Path to save the combined JSON. If None, returns the combined data.
    
    Returns:
        dict: Combined JSON data if output_file is None
    """
    combined_data = {}
    
    # Get all JSON files in the folder
    json_files = list(Path(folder_path).glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return None
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Merge the data
                for key, value in data.items():
                    if key in combined_data:
                        # If key exists, append the new questions
                        if isinstance(combined_data[key], list):
                            combined_data[key].extend(value)
                        else:
                            combined_data[key] = [combined_data[key], value]
                    else:
                        combined_data[key] = value
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Save to output file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        print(f"Combined data saved to {output_file}")
    
    return combined_data

if __name__ == "__main__":
    '''
    python pipeline/utils.py
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder_path", type=str, default="./dataset_viewer/CaptionQA_v0/document/questions")
    parser.add_argument("--output_file", type=str, default="./dataset_viewer/CaptionQA_v0/document/questions/combined_document.json")
    args = parser.parse_args()

    combine_json_files(args.folder_path, args.output_file)

    parser.add_argument("--json_file_path", type=str, default="./dataset_viewer/CaptionQA_v0/document/questions/combined_document.json")
    parser.add_argument("--threshold", type=int, default=900)
    args = parser.parse_args()
    json_file_path = args.json_file_path
    threshold = args.threshold
    check_question_counts(json_file_path, threshold)