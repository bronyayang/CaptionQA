import argparse
import os
import json
import mimetypes
from tqdm import tqdm
import time
import random
import openai
from typing import Tuple, Optional, Any, List, Dict
from pipeline.utils import load_json, encode_image, resize_image_for_api
from pipeline.api import (
    AMD_openai_client, AMD_openai_call, AMD_llama_client,
    AMD_gemini_client, AMD_gemini_call,
    AMD_claude_client, AMD_claude_call,
    AMD_vllm_chat_client, AMD_vllm_multimodal_call,
    AMD_vllm_server_client, AMD_vllm_server_multimodal_call
)
from caption_prompt import CAPTION_PROMPTS, get_prompt, create_taxonomy_prompts, list_available_prompts


def discover_input_items(benchmark_folder: str) -> List[Tuple[str, List[str], bool]]:
    """Discover both single images and image folders."""
    image_root_path = os.path.abspath(benchmark_folder)
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
    return input_items

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
    """Main function to caption images."""
    start_time = time.time()
    
    # Discover input items
    input_items = discover_input_items(args.benchmark_folder)
    
    # Print summary of discovered items
    print(f"Discovered {len(input_items)} items to process:")
    for key, image_paths, is_multi in input_items:
        if is_multi:
            print(f"  📁 {key}: {len(image_paths)} images (multi-view folder)")
        else:
            print(f"  🖼️  {key}: single image")

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
        print(f"Loaded existing results from {args.output_path}")
    else:
        results = {}
    
    # Determine which prompt to use (single prompt only)
    if args.taxonomy:
        # Use external taxonomy file - take first prompt
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
    
    # Process each input item
    for key, image_paths, is_multi in tqdm(input_items, desc="Processing Items"):
        
        # Skip if already processed and not overwriting
        if key in results and not args.overwrite:
            print(f"Skipping {key} (already processed, use --overwrite to regenerate)")
            continue
        
        # Modify prompt for multi-view images
        current_prompt = prompt_text
        if is_multi and len(image_paths) > 1:
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
            # Simple format: {image: caption}
            results[key] = caption
            
            # Save after each successful caption
            with open(args.output_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"Failed to generate caption for {key}")
    # Always write results at the end, even if empty or failures
    try:
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error writing results to {args.output_path}: {e}")
    print(f"Captioning complete! Results saved to {args.output_path}")
    elapsed = time.time() - start_time
    _mins, _secs = divmod(int(elapsed), 60)
    _hours, _mins = divmod(_mins, 60)
    print(f"Total caption time: {_hours:02d}:{_mins:02d}:{_secs:02d} ({elapsed:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="Generate captions for images using LLM")
    
    # Input/Output arguments
    parser.add_argument("--benchmark-folder", type=str, default="images",
                       help="Folder containing images or image folders")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save outputs to. Output file becomes: OUTPUT_DIR/prompt.lower()/_joined_model.json")
    
    # Prompt configuration
    parser.add_argument("--prompt", type=str, default="SIMPLE",
                       help="Prompt to use (e.g., SIMPLE, DETAILED, TECHNICAL, ARTISTIC, etc.)")
    
    parser.add_argument("--taxonomy", type=str, default=None,
                       help="Path to taxonomy JSON file (optional, uses built-in taxonomy if not provided)")
    
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
    
    # Derive output path from --output-dir
    model_safe = "_".join((args.model or "").split("/"))
    out_dir = os.path.join(args.output_dir, (args.prompt or "").lower())
    os.makedirs(out_dir, exist_ok=True)
    args.output_path = os.path.join(out_dir, f"{model_safe}.json")
    print(f"Saving outputs to {args.output_path}...")
    
    # List prompts if requested
    if args.list_prompts:
        list_available_prompts()
        return
    
    if not os.path.exists(args.benchmark_folder):
        print(f"Error: Benchmark folder {args.benchmark_folder} does not exist")
        return
    
    # Run captioning
    caption_images(args)

if __name__ == "__main__":
    main()
'''
Natural SIMPLE
python caption.py \
    --benchmark-folder ./data/natural_v0  \
    --output-dir ./captions/natural \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --vllm-server-url http://10.1.111.76:8006 \
    --prompt SIMPLE \
    --max-tokens 10000
'''
'''
Natural LONG
python caption.py \
    --benchmark-folder ./data/natural_v0  \
    --output-dir ./captions/natural \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --vllm-server-url http://10.1.111.76:8006 \
    --prompt LONG \
    --max-tokens 10000
'''
'''
Natural SHORT
python caption.py \
    --benchmark-folder ./data/natural_v0  \
    --output-dir ./captions/natural \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --vllm-server-url http://10.1.111.76:8006 \
    --prompt SHORT
'''
'''
Natural TAXONOMY_DEFAULT
python caption.py \
    --benchmark-folder ./data/natural_v0  \
    --output-dir ./captions/natural \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --vllm-server-url http://10.1.111.76:8006 \
    --prompt TAXONOMY_DEFAULT \
    --taxonomy ./general_taxonomy_v0.json \
    --max-tokens 10000
'''
'''
Natural TAXONOMY_STRUCTURED
python caption.py \
    --benchmark-folder ./data/natural_v0  \
    --output-dir ./captions/natural \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --vllm-server-url http://10.1.111.76:8006 \
    --prompt TAXONOMY_STRUCTURED \
    --taxonomy ./general_taxonomy_v0.json \
    --max-tokens 10000
'''
'''
EmbodiedAI TAXONOMY_STRUCTURED
python caption.py \
    --benchmark-folder ./data/embodied-ai_v0  \
    --output-dir ./captions/embodied-ai \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --vllm-server-url http://10.1.111.76:8006 \
    --prompt TAXONOMY_STRUCTURED \
    --taxonomy ./embodiedai_taxonomy_v0.json \
    --max-tokens 10000
'''