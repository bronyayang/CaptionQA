import argparse
import os
import json
import mimetypes
from tqdm import tqdm
import time
import random
import openai
import tempfile
from typing import Tuple, Optional, Any, List, Dict
from PIL import Image
from datasets import load_dataset
from pipeline.utils import load_json, encode_image
from pipeline.api import (
    AMD_openai_client, AMD_openai_call, AMD_llama_client,
    AMD_gemini_client, AMD_gemini_call,
    AMD_claude_client, AMD_claude_call,
    AMD_vllm_chat_client, AMD_vllm_multimodal_call,
    AMD_vllm_server_client, AMD_vllm_server_multimodal_call
)
from caption_prompt import CAPTION_PROMPTS, get_prompt, create_taxonomy_prompts, list_available_prompts

# Available domain splits in the dataset
DOMAIN_SPLITS = ["natural", "document", "ecommerce", "embodiedai"]


def _sleep_backoff(attempt: int, base: float = 0.5, factor: float = 2.0, jitter: float = 0.25) -> None:
    """Sleep with exponential backoff and jitter."""
    time.sleep(base * (factor ** attempt) + random.uniform(0, max(0.0, jitter)))


def detect_model_backend(model: str) -> str:
    """Detect which API backend to use based on model name."""
    model_lower = model.lower()
    if 'gemini' in model_lower:
        return 'gemini'
    elif 'claude' in model_lower or 'anthropic' in model_lower:
        return 'claude'
    elif any(vllm_model in model_lower for vllm_model in [
        'qwen', 'llama', 'mistral', 'phi', 'ovis',  # Text and multimodal
        'llava', 'internvl', 'minicpm', 'cogvlm', 'fuyu', 'glm'  # Multimodal specific
    ]):
        return 'vllm'
    elif 'llama' in model_lower:
        return 'llama'
    else:
        return 'openai'


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
    
    # Load the specified split directly
    dataset = load_dataset(dataset_name, split=split)
    
    print(f"Loaded {len(dataset)} entries")
    return dataset


def save_pil_image_to_temp(pil_image: Image.Image, suffix: str = ".jpg") -> str:
    """Save a PIL image to a temporary file and return the path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    pil_image.save(temp_file.name)
    return temp_file.name


def generate_caption(
    client: Any,
    model: str,
    image_paths: List[str],
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
    retries: int = 2,
    backend: str = 'openai'
) -> Optional[str]:
    """Generate caption for image(s) using the appropriate LLM backend."""
    
    for attempt in range(retries + 1):
        try:
            if backend == 'gemini':
                # Gemini uses image file paths directly
                completion = AMD_gemini_call(
                    client,
                    model,
                    messages=prompt,
                    image_paths=image_paths,
                    temperature=temperature
                )
                caption = completion.text.strip()
                return caption
                
            elif backend == 'claude':
                # Claude uses base64 encoded images (with 5 MB limit after encoding)
                content = [{"type": "text", "text": prompt}]
                
                # Add images with base64 encoding (resize_image_for_api handles size checking)
                for img_path in image_paths:
                    # This function checks if resizing is needed and returns base64 encoded string
                    # If resizing occurs, it converts to JPEG
                    image_data = resize_image_for_api(img_path)
                    
                    # Detect mime type (original or JPEG if resized)
                    mime_type = mimetypes.guess_type(img_path)[0] or "image/jpeg"
                    
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_data
                        }
                    })
                
                messages = [{"role": "user", "content": content}]
                
                completion = AMD_claude_call(
                    client,
                    model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                caption = completion.content[0].text.strip()
                return caption
                
            elif backend == 'vllm':
                # Use vLLM multimodal API
                result = AMD_vllm_multimodal_call(
                    client,
                    {"text": prompt, "image_paths": image_paths},
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                if isinstance(result, list) and len(result) > 0:
                    return result[0].strip()
                return None
            elif backend == 'vllm_server':
                # Use vLLM server (OpenAI-compatible HTTP) multimodal API
                # Some vLLM server deployments (e.g., NVLM-D-72B) require top_p in (0, 1]
                # and may default to 0. Set a safe default only for this model.
                if str(model).strip().lower() == "nvidia/nvlm-d-72b":
                    result = AMD_vllm_server_multimodal_call(
                        client,
                        {"text": prompt, "image_paths": image_paths},
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=1.0 #add this
                    )
                else:
                    result = AMD_vllm_server_multimodal_call(
                        client,
                        {"text": prompt, "image_paths": image_paths},
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                if isinstance(result, list) and len(result) > 0:
                    return result[0].strip()
                return None
                
            else:  # OpenAI backend
                # Encode all images for OpenAI
                encoded_images = [encode_image(img_path) for img_path in image_paths]
                
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
                    {"role": "user", "content": prompt},
                ]
                
                completion = AMD_openai_call(
                    client,
                    model,
                    messages=messages,
                    temperature=temperature,
                    stream=False,
                    max_tokens=max_tokens
                )
                
                caption = completion.choices[0].message.content.strip()
                return caption
            
        except openai.OpenAIError as e:
            print(f"[api_error] attempt {attempt + 1}: {e}")
            if attempt < retries:
                _sleep_backoff(attempt)
            continue
        except Exception as e:
            print(f"[unknown_error] attempt {attempt + 1}: {e}")
            if attempt < retries:
                _sleep_backoff(attempt)
            continue
    
    return None


def caption_images(args):
    """Main function to caption images from HuggingFace dataset."""
    start_time = time.time()
    
    # Load dataset from HuggingFace
    dataset = load_captionqa_dataset(args.dataset, args.split)
    
    print(f"Processing {len(dataset)} images from {args.dataset} ({args.split} split)")

    # Initialize client based on model backend
    backend = detect_model_backend(args.model)
    print(f"Using {backend} backend for model {args.model}")
    
    if getattr(args, 'vllm_server_url', None):
        backend = 'vllm_server'
    
    if backend == 'gemini':
        client = AMD_gemini_client()
    elif backend == 'claude':
        client = AMD_claude_client()
    elif backend == 'vllm_server':
        client = AMD_vllm_server_client(base_url=args.vllm_server_url, model=args.model, tensor_parallel_size=args.tp_size)
    elif backend == 'vllm':
        # Use unified vLLM client for all vLLM models (text and multimodal)
        client = AMD_vllm_chat_client(model=args.model, tp_size=args.tp_size)
    else:
        client = AMD_openai_client(model_id=args.model)
    
    # Load existing results if available
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded existing results from {args.output_path} ({len(results)} captions)")
    else:
        results = {}
    
    # Determine which prompt to use
    if args.taxonomy:
        # Use external taxonomy file
        taxonomy = load_json(args.taxonomy)
        if args.prompt == "TAXONOMY_DEFAULT":
            tax_prompts = create_taxonomy_prompts(taxonomy, prompt_name="default")
        elif args.prompt == "TAXONOMY_STRUCTURED":
            tax_prompts = create_taxonomy_prompts(taxonomy, prompt_name="structured")
        else:
            print(f"Error: Unknown prompt name '{args.prompt}'")
            exit(1)
        print(f"Using taxonomy prompt: {args.prompt}")
        print(tax_prompts)
        prompt_text = tax_prompts
    else:
        # Use specific prompt
        prompt_text = get_prompt(args.prompt)
        print(f"Using prompt: {args.prompt}")
        print(prompt_text)
    
    # Track temporary files for cleanup
    temp_files = []
    
    # Process each entry in the dataset
    for entry in tqdm(dataset, desc="Captioning"):
        # Get image ID (e.g., "nat_001", "doc_042")
        image_id = entry.get('id')
        if image_id is None:
            continue
        
        image_key = str(image_id)
        
        # Skip if already processed and not overwriting
        if image_key in results and not args.overwrite:
            continue
        
        # Get images from the dataset entry
        images = entry.get('images', [])
        if not images:
            print(f"Skipping {image_key} (no images)")
            continue
        
        # Save PIL images to temporary files for API calls
        image_paths = []
        for img in images:
            if isinstance(img, Image.Image):
                temp_path = save_pil_image_to_temp(img)
                image_paths.append(temp_path)
                temp_files.append(temp_path)
            elif isinstance(img, str):
                # Already a path
                image_paths.append(img)
        
        if not image_paths:
            print(f"Skipping {image_key} (could not process images)")
            continue
        
        # Modify prompt for multi-image entries
        current_prompt = prompt_text
        if len(image_paths) > 1:
            if 'MULTIVIEW' in CAPTION_PROMPTS:
                current_prompt = get_prompt('MULTIVIEW')
            else:
                current_prompt = f"You are viewing {len(image_paths)} related images. {prompt_text}"
        
        caption = generate_caption(
            client=client,
            model=args.model,
            image_paths=image_paths,
            prompt=current_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            backend=backend
        )
        
        if caption:
            # Save with dataset ID as key (e.g., "nat_001": "caption...")
            results[image_key] = caption
            
            # Save after each successful caption
            with open(args.output_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"Failed to generate caption for {image_key}")
    
    # Cleanup temporary files
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except Exception:
            pass
    
    # Always write results at the end
    try:
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error writing results to {args.output_path}: {e}")
    
    print(f"Captioning complete! Results saved to {args.output_path}")
    print(f"Total captions: {len(results)}")
    elapsed = time.time() - start_time
    _mins, _secs = divmod(int(elapsed), 60)
    _hours, _mins = divmod(_mins, 60)
    print(f"Total caption time: {_hours:02d}:{_mins:02d}:{_secs:02d} ({elapsed:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="Generate captions for Borise/CaptionQA dataset")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="Borise/CaptionQA",
                       help="HuggingFace dataset name (default: Borise/CaptionQA)")
    parser.add_argument("--split", type=str, default="all",
                       choices=["natural", "document", "ecommerce", "embodiedai", "all"],
                       help="Domain split to caption (default: natural)")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save outputs to. Output file becomes: OUTPUT_DIR/prompt.lower()/model.json")
    
    # Prompt configuration
    parser.add_argument("--prompt", type=str, default="SIMPLE",
                       help="Prompt to use (e.g., SIMPLE, DETAILED, TECHNICAL, ARTISTIC, etc.)")
    parser.add_argument("--taxonomy", type=str, default=None,
                       help="Path to taxonomy JSON file (optional)")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="Model to use for captioning")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=500,
                       help="Maximum tokens to generate (default: 500)")
    
    # Processing options
    parser.add_argument("--retries", type=int, default=2,
                       help="Number of retries for failed API calls")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing captions")
    
    # Utility options
    parser.add_argument("--list-prompts", action="store_true",
                       help="List all available prompts and exit")
    parser.add_argument("--vllm-server-url", type=str, default=None,
                       help="Base URL for vLLM server (OpenAI-compatible), e.g. http://10.1.64.88:8006")
    parser.add_argument("--tp-size", type=int, default=1,
                       help="Tensor parallel size for vLLM inference (default: 1)")
    
    args = parser.parse_args()
    
    # List prompts if requested
    if args.list_prompts:
        list_available_prompts()
        return
    
    # Derive output path from --output-dir
    model_safe = "_".join((args.model or "").split("/"))
    out_dir = os.path.join(args.output_dir, (args.prompt or "").lower())
    os.makedirs(out_dir, exist_ok=True)
    args.output_path = os.path.join(out_dir, f"{model_safe}.json")
    print(f"Saving outputs to {args.output_path}...")
    
    # Run captioning
    caption_images(args)


if __name__ == "__main__":
    main()
