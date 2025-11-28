from __future__ import annotations

import os
import getpass
import mimetypes
import base64
from dataclasses import dataclass
from typing import List, Union, Optional, Any, Dict, Iterable

# ---------- optional deps (lazy) ----------
try:
    import openai as _openai  # Azure OpenAI SDK
    from openai import BadRequestError as _BadRequestError
except Exception:  # not installed or import error
    _openai = None
    class _BadRequestError(Exception):  # fallback to avoid NameError at runtime
        pass

try:
    # Google GenAI SDK（新版）：pip install google-genai
    from google import genai as _genai
    from google.genai.types import HttpOptions, GenerateContentConfig, Part
except Exception:
    _genai = None
    HttpOptions = None
    GenerateContentConfig = None
    Part = None

try:
    import transformers as _transformers  # for AutoTokenizer & chat template
except Exception:
    _transformers = None

try:
    import vllm as _vllm  # for LLM, SamplingParams
except Exception:
    _vllm = None

try:
    import anthropic as _anthropic  # for Claude models
except Exception:
    _anthropic = None


class OptionalDependencyError(ImportError):
    pass


def current_user() -> str:
    for k in ("AMD_API_USER", "API_USER", "SLURM_JOB_USER",
              "SUDO_USER", "LOGNAME", "USER", "USERNAME"):
        v = os.getenv(k)
        if v:
            return v
    try:
        return getpass.getuser()
    except Exception:
        try:
            import pwd
            return pwd.getpwuid(os.getuid()).pw_name
        except Exception:
            return "unknown"


# -------------------- OpenAI (optional) --------------------
def AMD_openai_client(model_id: str, amd: bool = False) -> Any:
    """Create a standard OpenAI client using OPENAI_API_KEY.

    The model_id argument is accepted for compatibility with callers but is not
    needed to construct the client.
    """
    if _openai is None:
        raise OptionalDependencyError(
            "openai is not installed. pip install 'openai>=1.0.0'"
        )
    print(f"amd: {amd}")
    if amd:
        user = current_user()
        url = "https://llm-api.amd.com"
        client = _openai.AzureOpenAI(
            api_key="dummy",
            api_version="2024-12-01-preview",
            base_url=url,
            default_headers={
                "Ocp-Apim-Subscription-Key": "YOUR_SUBSCRIPTION_KEY",
                "user": user,
            },
        )
        # keep your existing deployment path pattern
        client.base_url = f"{url}/openai/deployments/{model_id}"
        return client
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

        # Allow overriding base URL for proxies/self-hosted compatible servers
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return _openai.OpenAI(api_key=api_key, base_url=base_url)
        return _openai.OpenAI(api_key=api_key)


def AMD_llama_client() -> Any:
    """Create OpenAI client for AMD OnPrem Llama models."""
    if _openai is None:
        raise OptionalDependencyError(
            "openai is not installed. pip install 'openai>=1.0.0'"
        )

    
    user = current_user()
    client = _openai.OpenAI(
        base_url="https://llm-api.amd.com/OnPrem",
        api_key="dummy",
        default_headers={
            "Ocp-Apim-Subscription-Key": "YOUR_SUBSCRIPTION_KEY",
            "user": user,
        },
    )
    return client


def AMD_openai_call(
    client: Any,  # do not reference openai types here
    model_id: str,
    messages: Union[str, List[dict]],
    **kwargs: Any,
) -> Any:
    """
    Make a chat completion call. Accepts either a string or a list of message dicts.
    Extra kwargs pass-through (temperature, stream, max_completion_tokens, reasoning_effort, etc.)
    """
    if _openai is None:
        raise OptionalDependencyError(
            "openai is not installed. pip install 'openai>=1.0.0'"
        )

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    try:
        return client.chat.completions.create(model=model_id, messages=messages, **kwargs)
    except _BadRequestError as e:
        msg = str(e)
        # Retry without reasoning_effort if backend doesn't accept it
        if "Unrecognized request argument" in msg and ("reasoning_effort" in msg):
            kwargs.pop("reasoning_effort", None)
            return client.chat.completions.create(model=model_id, messages=messages, **kwargs)
        raise


def AMD_openai_structured_multimodal_call(
    client: Any,
    model_id: str,
    items: Union[Dict[str, Any], List[Dict[str, Any]]],
    json_schema: Dict[str, Any],
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    system: str = "You are a helpful assistant.",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Make a structured multimodal call with OpenAI API using response_format.
    
    Each item should have:
      - 'text' or 'question' or 'prompt': the text prompt
      - 'image_paths' or 'images': list of image file paths or URLs
    
    Args:
        client: OpenAI client
        model_id: Model identifier
        items: Single item dict or list of item dicts
        json_schema: JSON schema for structured output
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        system: System message
    
    Returns:
        Parsed JSON object(s) matching the schema
    """
    if _openai is None:
        raise OptionalDependencyError(
            "openai is not installed. pip install 'openai>=1.0.0'"
        )
    
    batch: List[Dict[str, Any]] = items if isinstance(items, list) else [items]
    results = []
    
    for it in batch:
        # Extract text
        text = None
        for key in ("text", "question", "prompt"):
            if key in it and it[key] is not None:
                text = str(it[key])
                break
        if text is None:
            raise ValueError("Each item must contain 'text', 'question', or 'prompt'")
        
        # Extract image paths
        image_paths = []
        for key in ("image_paths", "images", "image"):
            if key in it and it[key] is not None:
                img_val = it[key]
                if isinstance(img_val, str):
                    image_paths = [img_val]
                elif isinstance(img_val, (list, tuple)):
                    image_paths = [str(p) for p in img_val]
                break
        if not image_paths:
            raise ValueError("Each item must contain 'image_paths', 'images', or 'image'")
        
        # Build content with images
        content_items = []
        for img_path in image_paths:
            # Check if it's a URL or local file
            if img_path.startswith(('http://', 'https://')):
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": img_path}
                })
            else:
                # Encode local file as base64
                mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
                with open(img_path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode('utf-8')
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_data}"}
                })
        
        content_items.append({"type": "text", "text": text})
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content_items}
        ]
        
        # Use response_format for structured output
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "taxonomy_response",
                "strict": True,
                "schema": json_schema
            }
        }
        
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Parse the JSON response
        import json
        content = response.choices[0].message.content
        parsed = json.loads(content)
        results.append(parsed)
    
    return results if isinstance(items, list) else results[0]

# -------------------- Gemini (optional) --------------------
def AMD_gemini_client() -> Any:
    """Create Gemini (Google GenAI) client via AMD gateway."""
    if _genai is None:
        raise OptionalDependencyError(
            "google-genai is not installed. pip install 'google-genai>=1.0.0'"
        )
    client = _genai.Client(
        vertexai=True,
        api_key="dummy",
        http_options=HttpOptions(
            base_url="https://llm-api.amd.com/VertexGen",
            api_version="v1",
            headers={
                "Ocp-Apim-Subscription-Key": "YOUR_SUBSCRIPTION_KEY"
            },
        ),
    )
    return client

def AMD_gemini_call(
    client: Any,
    model_id: str,
    messages: str,                 # 只接受字符串
    *,
    image_paths: Iterable[str],    # 只接受本地图片路径（可多张）
    default_mime: str = "image/jpeg",
    **kwargs: Any,                 # 直接传入 GenerateContentConfig（不做过滤）
) -> Any:
    if _genai is None:
        raise OptionalDependencyError(
            "google-genai is not installed. pip install 'google-genai>=1.0.0'"
        )
    # 1) 文本（已保证是字符串）
    text = messages

    # 2) contents：图片在前、文本在后
    contents: List[Any] = []
    for p in image_paths:
        mime = mimetypes.guess_type(p)[0] or default_mime
        with open(p, "rb") as f:
            contents.append(Part.from_bytes(data=f.read(), mime_type=mime))
    contents.append(text)

    # 3) 直接构造 config（kwargs 不合法时让其抛错）
    config = GenerateContentConfig(**kwargs) if kwargs else None

    # 4) 调用
    return client.models.generate_content(
        model=model_id,
        contents=contents if len(contents) > 1 else contents[0],
        config=config,
    )

# -------------------- Claude/Anthropic (optional) --------------------
def AMD_claude_client() -> Any:
    """Create Anthropic (Claude) client via AMD gateway."""
    if _anthropic is None:
        raise OptionalDependencyError(
            "anthropic is not installed. pip install 'anthropic>=0.18.0'"
        )
    
    client = _anthropic.Anthropic(
        base_url="https://llm-api.amd.com/Anthropic",
        api_key="dummy",
        default_headers={
            "Ocp-Apim-Subscription-Key": "YOUR_SUBSCRIPTION_KEY",
            "anthropic-version": "2023-10-16"
        },
        timeout=600,
    )
    return client


def AMD_claude_call(
    client: Any,
    model_id: str,
    messages: Union[str, List[dict]],
    **kwargs: Any,
) -> Any:
    """
    Make a Claude API call. Accepts either a string or a list of message dicts.
    Extra kwargs pass-through (max_tokens, temperature, top_p, etc.)
    
    Args:
        client: Anthropic client from AMD_claude_client()
        model_id: Claude model name (e.g., "claude-3-5-sonnet-20241022")
        messages: Either a string or list of message dicts (Claude format)
        **kwargs: Additional parameters (max_tokens, temperature, top_p, etc.)
    
    Returns:
        Anthropic API response object
    
    Example:
        client = AMD_claude_client()
        response = AMD_claude_call(
            client,
            model_id="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": [...]}],
            max_tokens=1024,
            temperature=0.6,
            top_p=0.95
        )
        text = response.content[0].text
    """
    if _anthropic is None:
        raise OptionalDependencyError(
            "anthropic is not installed. pip install 'anthropic>=0.18.0'"
        )
    
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    return client.messages.create(
        model=model_id,
        messages=messages,
        **kwargs
    )

# -------------------- vLLM (optional) ----------------------
@dataclass
class AMDvLLMClient:
    tokenizer: Any
    llm: Any
    processor: Optional[Any] = None  # For multimodal models


def AMD_vllm_chat_client(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    *,
    tp_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    trust_remote_code: bool = True,
    **llm_kwargs: Any,
) -> AMDvLLMClient:
    """
    Create vLLM client for both text-only and multimodal models.
    
    For multimodal models (LLaVA, Qwen-VL, LLaMA-Vision, etc.),
    use this same function - it will work with AMD_vllm_multimodal_call.
    
    Example:
        # Text-only model
        client = AMD_vllm_chat_client(model="Qwen/Qwen2.5-7B-Instruct")
        
        # Multimodal model
        client = AMD_vllm_chat_client(model="Qwen/Qwen2-VL-7B-Instruct")
    """
    if _vllm is None:
        raise OptionalDependencyError("vllm is not installed. pip install 'vllm'")
    if _transformers is None:
        raise OptionalDependencyError("transformers is not installed. pip install 'transformers'")

    AutoTokenizer = _transformers.AutoTokenizer
    LLM = _vllm.LLM

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    
    # Try to load processor for multimodal models
    processor = None
    try:
        AutoProcessor = _transformers.AutoProcessor
        processor = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
    except Exception:
        # Processor not available, that's fine for text-only models
        pass
    
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        # max_model_len=32000,
        # max_num_batched_tokens=320000,
        # max_num_seqs=64,
        **llm_kwargs,
    )
    return AMDvLLMClient(tokenizer=tok, llm=llm, processor=processor)


# Backward compatibility alias
AMD_vllm_text_chat_client = AMD_vllm_chat_client


# ---------- generic prompt builder ----------
def _build_prompt(question: str, tokenizer: Any, system: str = "You are a helpful assistant.") -> str:
    # Prefer model-provided chat template
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except TypeError:
            # Qwen-style content list fallback
            msgs = [
                {"role": "system", "content": system},
                {"role": "user",   "content": [{"type": "text", "text": question}]},
            ]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Fallback for base models with no template
    return f"{system}\n\nUser: {question}\nAssistant:"


def AMD_vllm_text_chat_call(
    client: AMDvLLMClient,
    items: Union[str, Dict[str, str], List[Union[str, Dict[str, str]]]],
    *,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = 1.0,
    stop_token_ids: Optional[List[int]] = None,
    seed: Optional[int] = None,
    use_tqdm: bool = False,
    system: str = "You are a helpful assistant.",
    n: int = 1,               # number of samples per prompt
    return_all: bool = False, # True -> return List[List[str]] of all candidates
) -> Union[List[str], List[List[str]]]:
    if _vllm is None:
        raise OptionalDependencyError("vllm is not installed. pip install 'vllm'")

    SamplingParams = _vllm.SamplingParams

    # Normalize batch
    batch: List[Union[str, Dict[str, str]]] = items if isinstance(items, list) else [items]

    # Build prompts
    prompts: List[str] = []
    for it in batch:
        if isinstance(it, str):
            prompts.append(_build_prompt(it, client.tokenizer, system))
        elif isinstance(it, dict):
            if "prompt" in it and it["prompt"] is not None:
                prompts.append(str(it["prompt"]))
            elif "question" in it and it["question"] is not None:
                prompts.append(_build_prompt(str(it["question"]), client.tokenizer, system))
            else:
                raise ValueError("Each dict item must contain 'question' or 'prompt'.")
        else:
            raise TypeError(f"Unsupported item type: {type(it)}")

    # Sampling params
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        n=max(1, int(n)),
    )

    outs = client.llm.generate(prompts, sampling, use_tqdm=use_tqdm)

    if sampling.n == 1 and not return_all:
        return [o.outputs[0].text.strip() if getattr(o, "outputs", None) else "" for o in outs]

    all_texts: List[List[str]] = []
    for o in outs:
        candidates = [cand.text.strip() for cand in (getattr(o, "outputs", None) or [])]
        while len(candidates) < sampling.n:
            candidates.append("")
        all_texts.append(candidates)
    return all_texts


def _build_multimodal_prompt_with_processor(
    text: str,
    images: List[Any],  # PIL Images
    processor: Any,
    system: str = "You are a helpful assistant."
) -> str:
    """
    Build multimodal prompt for vLLM using processor.
    """
    placeholders = [{"type": "image", "image": img} for img in images]
    
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": text},
            ],
        },
    ]
    
    # Use processor's apply_chat_template (not tokenizer's)
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback: Try tokenizer if processor doesn't have the method
        print("Fallback: Try tokenizer if processor doesn't have the method")
        # if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "apply_chat_template"):
        #     return processor.tokenizer.apply_chat_template(
        #         messages,
        #         tokenize=False,
        #         add_generation_prompt=True
        #     )
        # Last resort: Phi4 multimodal format
        image_tokens = "".join([f"<|image_{i+1}|>" for i in range(len(images))])
        return f"<|user|>{image_tokens}{text}<|end|><|assistant|>"


def AMD_vllm_multimodal_call(
    client: AMDvLLMClient,
    items: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = 1.0,
    stop_token_ids: Optional[List[int]] = None,
    seed: Optional[int] = None,
    use_tqdm: bool = False,
    system: str = "You are a helpful assistant.",
    n: int = 1,
    return_all: bool = False,
) -> Union[List[str], List[List[str]]]:
    """
    Call vLLM with multimodal inputs (text + images).
    
    Supports all vLLM multimodal models including:
      - LLaVA variants (llava-1.5-7b-hf, llava-v1.6-vicuna-7b, etc.)
      - Qwen-VL variants (Qwen-VL-Chat, Qwen2-VL, etc.)
      - LLaMA 3.2 Vision
      - InternVL, MiniCPM-V, Phi-3-Vision, and others
    
    Each item should be a dict with:
      - 'text' or 'question' or 'prompt': the text prompt
      - 'image_paths' or 'images': list of image file paths (supports multiple images)
    
    Example:
        # Single image
        result = AMD_vllm_multimodal_call(
            client,
            {"text": "Describe this image", "image_paths": ["img.jpg"]},
        )
        
        # Multiple images per prompt
        result = AMD_vllm_multimodal_call(
            client,
            {"text": "Compare these images", "image_paths": ["img1.jpg", "img2.jpg"]},
        )
        
        # Batch processing
        results = AMD_vllm_multimodal_call(
            client,
            [
                {"question": "What's in this?", "images": ["photo1.jpg"]},
                {"text": "Describe the scene", "image_paths": ["photo2.jpg", "photo3.jpg"]}
            ],
            use_tqdm=True
        )
    """
    if _vllm is None:
        raise OptionalDependencyError("vllm is not installed. pip install 'vllm'")
    
    SamplingParams = _vllm.SamplingParams
    
    # Normalize to list
    batch: List[Dict[str, Any]] = items if isinstance(items, list) else [items]
    
    # Build prompts and extract image paths
    prompts: List[str] = []
    multi_modal_data: List[Dict[str, Any]] = []
    
    for it in batch:
        if not isinstance(it, dict):
            raise TypeError(f"Each item must be a dict, got {type(it)}")
        
        # Extract text
        text = None
        for key in ("text", "question", "prompt"):
            if key in it and it[key] is not None:
                text = str(it[key])
                break
        if text is None:
            raise ValueError("Each item must contain 'text', 'question', or 'prompt'")
        
        # Extract image paths
        image_paths = []
        for key in ("image_paths", "images", "image"):
            if key in it and it[key] is not None:
                img_val = it[key]
                if isinstance(img_val, str):
                    image_paths = [img_val]
                elif isinstance(img_val, (list, tuple)):
                    image_paths = [str(p) for p in img_val]
                break
        if not image_paths:
            raise ValueError("Each item must contain 'image_paths', 'images', or 'image'")
        
        # Load images as PIL Images
        try:
            from PIL import Image
            pil_images = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    pil_images.append(img)
                except Exception as e:
                    raise RuntimeError(f"Failed to load image {img_path}: {e}") from e
            
            # Build prompt using processor if available
            # Processor inspects the PIL images to calculate correct token count automatically
            if client.processor is not None:
                prompt = _build_multimodal_prompt_with_processor(text, pil_images, client.processor, system)
            else:
                # Fallback for models without processor
                prompt = _build_multimodal_prompt_with_processor(text, pil_images, client.tokenizer, system)
            prompts.append(prompt)
            
            # Store single image or list based on count
            multi_modal_data.append({"image": pil_images if len(pil_images) > 1 else pil_images[0]})
        except ImportError:
            raise OptionalDependencyError("PIL is required for multimodal. pip install 'Pillow'")
    
    # Sampling params
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        n=max(1, int(n)),
    )
    
    # Generate with multimodal data (vLLM 0.8.x format)
    # Build inputs as list of dicts with "prompt" and "multi_modal_data" keys
    inputs = []
    for prompt, mm_data in zip(prompts, multi_modal_data):
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": mm_data,
        })
    
    outs = client.llm.generate(inputs, sampling, use_tqdm=use_tqdm)
    
    # Extract results
    if sampling.n == 1 and not return_all:
        return [o.outputs[0].text.strip() if getattr(o, "outputs", None) else "" for o in outs]
    
    all_texts: List[List[str]] = []
    for o in outs:
        candidates = [cand.text.strip() for cand in (getattr(o, "outputs", None) or [])]
        while len(candidates) < sampling.n:
            candidates.append("")
        all_texts.append(candidates)
    return all_texts


def AMD_vllm_structured_multimodal_call(
    client: AMDvLLMClient,
    items: Union[Dict[str, Any], List[Dict[str, Any]]],
    json_schema: Dict[str, Any],
    *,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    use_tqdm: bool = False,
    system: str = "You are a helpful assistant.",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Call vLLM with structured multimodal inputs using guided JSON decoding.
    
    Supports all vLLM multimodal models with structured output enforcement.
    
    Each item should be a dict with:
      - 'text' or 'question' or 'prompt': the text prompt
      - 'image_paths' or 'images': list of image file paths (supports multiple images)
    
    Args:
        client: vLLM client
        items: Single item dict or list of item dicts
        json_schema: JSON schema for structured output
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        seed: Random seed
        use_tqdm: Show progress bar
        system: System message
    
    Returns:
        Parsed JSON object(s) matching the schema
    
    Example:
        schema = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "objects": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["description", "objects"]
        }
        
        result = AMD_vllm_structured_multimodal_call(
            client,
            {"text": "Describe this image", "image_paths": ["img.jpg"]},
            json_schema=schema
        )
    """
    if _vllm is None:
        raise OptionalDependencyError("vllm is not installed. pip install 'vllm'")
    
    SamplingParams = _vllm.SamplingParams
    
    # Normalize to list
    batch: List[Dict[str, Any]] = items if isinstance(items, list) else [items]
    
    # Build prompts and extract image paths
    prompts: List[str] = []
    multi_modal_data: List[Dict[str, Any]] = []
    
    for it in batch:
        if not isinstance(it, dict):
            raise TypeError(f"Each item must be a dict, got {type(it)}")
        
        # Extract text
        text = None
        for key in ("text", "question", "prompt"):
            if key in it and it[key] is not None:
                text = str(it[key])
                break
        if text is None:
            raise ValueError("Each item must contain 'text', 'question', or 'prompt'")
        
        # Extract image paths
        image_paths = []
        for key in ("image_paths", "images", "image"):
            if key in it and it[key] is not None:
                img_val = it[key]
                if isinstance(img_val, str):
                    image_paths = [img_val]
                elif isinstance(img_val, (list, tuple)):
                    image_paths = [str(p) for p in img_val]
                break
        if not image_paths:
            raise ValueError("Each item must contain 'image_paths', 'images', or 'image'")
        
        # Load images as PIL Images
        try:
            from PIL import Image
            pil_images = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    pil_images.append(img)
                except Exception as e:
                    raise RuntimeError(f"Failed to load image {img_path}: {e}") from e
            
            # Build prompt using processor if available
            if client.processor is not None:
                prompt = _build_multimodal_prompt_with_processor(text, pil_images, client.processor, system)
            else:
                # Fallback for models without processor
                prompt = _build_multimodal_prompt_with_processor(text, pil_images, client.tokenizer, system)
            prompts.append(prompt)
            
            # Store single image or list based on count
            multi_modal_data.append({"image": pil_images if len(pil_images) > 1 else pil_images[0]})
        except ImportError:
            raise OptionalDependencyError("PIL is required for multimodal. pip install 'Pillow'")
    
    # Sampling params with guided JSON decoding
    import json
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        guided_json=json.dumps(json_schema),  # Enable structured output
    )
    
    # Generate with multimodal data
    inputs = []
    for prompt, mm_data in zip(prompts, multi_modal_data):
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": mm_data,
        })
    
    outs = client.llm.generate(inputs, sampling, use_tqdm=use_tqdm)
    
    # Parse JSON results
    results = []
    for o in outs:
        text = o.outputs[0].text.strip() if getattr(o, "outputs", None) else "{}"
        parsed = json.loads(text)
        results.append(parsed)
    
    return results if isinstance(items, list) else results[0]


# -------------------- vLLM Server (OpenAI-compatible HTTP) -----------------
@dataclass
class AMDvLLMServerClient:
    base_url: str
    model: str
    headers: Dict[str, str]


def AMD_vllm_server_client(
    base_url: str,
    model: str,
    headers: Optional[Dict[str, str]] = None,
) -> AMDvLLMServerClient:
    """Create a lightweight client for a vLLM server exposing OpenAI-compatible APIs.

    base_url should look like: http://host:port (no trailing slash is required)
    """
    sanitized = (base_url or "").rstrip("/")
    default_headers: Dict[str, str] = {"Content-Type": "application/json"}
    if headers:
        default_headers.update(headers)
    return AMDvLLMServerClient(base_url=sanitized, model=model, headers=default_headers)


def _encode_image_b64_local(path: str, default_mime: str = "image/jpeg") -> str:
    mime = mimetypes.guess_type(path)[0] or default_mime
    with open(path, "rb") as f:
        data = f.read()
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def AMD_vllm_server_multimodal_call(
    client: AMDvLLMServerClient,
    items: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    system: str = "You are a helpful assistant.",
    n: int = 1,
    return_all: bool = False,
) -> Union[List[str], List[List[str]]]:
    """Call a running vLLM server (OpenAI-compatible) with image(s) + text.

    Each item is a dict with keys similar to AMD_vllm_multimodal_call:
      - 'text' | 'question' | 'prompt'
      - 'image_paths' | 'images' | 'image' (strings or list of strings)
    """
    try:
        import requests  # lazy import to keep dependency optional
    except Exception:
        raise OptionalDependencyError("requests is not installed. pip install 'requests'")

    batch: List[Dict[str, Any]] = items if isinstance(items, list) else [items]

    all_outputs: List[List[str]] = []
    url = f"{client.base_url}/v1/chat/completions"

    for it in batch:
        if not isinstance(it, dict):
            raise TypeError(f"Each item must be a dict, got {type(it)}")

        # Extract text
        text = None
        for key in ("text", "question", "prompt"):
            if key in it and it[key] is not None:
                text = str(it[key])
                break
        if text is None:
            raise ValueError("Each item must contain 'text', 'question', or 'prompt'")

        # Extract image paths
        image_paths: List[str] = []
        for key in ("image_paths", "images", "image"):
            if key in it and it[key] is not None:
                img_val = it[key]
                if isinstance(img_val, str):
                    image_paths = [img_val]
                elif isinstance(img_val, (list, tuple)):
                    image_paths = [str(p) for p in img_val]
                break
        if not image_paths:
            raise ValueError("Each item must contain 'image_paths', 'images', or 'image'")

        # Build content list: images first, then text, in a single user message
        content_items: List[Dict[str, Any]] = []
        for p in image_paths:
            try:
                url_b64 = _encode_image_b64_local(p)
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": url_b64},
                })
            except Exception:
                # Skip unreadable image; keep going if at least one image remains
                continue
        if not content_items:
            raise ValueError("None of the provided image paths could be read/encoded")
        content_items.append({"type": "text", "text": text})

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content_items},
        ]

        payload: Dict[str, Any] = {
            "model": client.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "n": int(max(1, n)),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if presence_penalty is not None:
            payload["presence_penalty"] = float(presence_penalty)

        resp = requests.post(url, headers=client.headers, json=payload)
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"vLLM server returned non-JSON response: HTTP {resp.status_code}")

        if resp.status_code >= 400:
            err_msg = data.get("error", {}).get("message") if isinstance(data, dict) else None
            raise RuntimeError(f"vLLM server error {resp.status_code}: {err_msg or data}")

        choices = data.get("choices", []) if isinstance(data, dict) else []
        if not choices:
            all_outputs.append([""] * int(max(1, n)))
            continue

        texts = []
        for i in range(min(len(choices), int(max(1, n)))):
            msg = choices[i].get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            texts.append(str(content).strip())

        while len(texts) < int(max(1, n)):
            texts.append("")
        all_outputs.append(texts)

    if n == 1 and not return_all:
        return [outs[0] if outs else "" for outs in all_outputs]
    return all_outputs


def AMD_vllm_server_text_chat_call(
    client: AMDvLLMServerClient,
    items: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    system: str = "You are a helpful assistant.",
    n: int = 1,
    return_all: bool = False,
) -> Union[List[str], List[List[str]]]:
    """Call a running vLLM server (OpenAI-compatible) for text chat.

    Each item can be a string (user prompt) or a dict with keys:
      - 'prompt' | 'question' | 'text'
    """
    try:
        import requests  # lazy import to keep dependency optional
    except Exception:
        raise OptionalDependencyError("requests is not installed. pip install 'requests'")

    batch_list: List[Union[str, Dict[str, Any]]] = items if isinstance(items, list) else [items]
    url = f"{client.base_url}/v1/chat/completions"
    all_outputs: List[List[str]] = []

    for it in batch_list:
        if isinstance(it, str):
            text = it
        elif isinstance(it, dict):
            text = None
            for key in ("prompt", "question", "text"):
                if key in it and it[key] is not None:
                    text = str(it[key])
                    break
            if text is None:
                raise ValueError("Each dict item must contain 'prompt', 'question', or 'text'")
        else:
            raise TypeError(f"Unsupported item type: {type(it)}")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]

        payload: Dict[str, Any] = {
            "model": client.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "n": int(max(1, n)),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if presence_penalty is not None:
            payload["presence_penalty"] = float(presence_penalty)

        resp = requests.post(url, headers=client.headers, json=payload)
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"vLLM server returned non-JSON response: HTTP {resp.status_code}")

        if resp.status_code >= 400:
            err_msg = data.get("error", {}).get("message") if isinstance(data, dict) else None
            raise RuntimeError(f"vLLM server error {resp.status_code}: {err_msg or data}")

        choices = data.get("choices", []) if isinstance(data, dict) else []
        texts: List[str] = []
        for i in range(min(len(choices), int(max(1, n)))):
            msg = choices[i].get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            texts.append(str(content).strip())

        while len(texts) < int(max(1, n)):
            texts.append("")
        all_outputs.append(texts)

    if n == 1 and not return_all:
        return [outs[0] if outs else "" for outs in all_outputs]
    return all_outputs

# -------------------- Qwen-VL via Transformers (optional) -----------------
@dataclass
class AMDQwenVLClient:
    model: Any
    processor: Any
    device: str


def AMD_qwenvl_client(
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    *,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
) -> AMDQwenVLClient:
    """Create a local Transformers client for Qwen-VL models.

    Works on CPU, CUDA, or Apple MPS depending on availability.
    """
    try:
        import torch  # type: ignore
        from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
    except Exception:
        raise OptionalDependencyError(
            "transformers/torch are not installed. pip install 'transformers torch'"
        )

    # pick device
    chosen_device = device
    if chosen_device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            chosen_device = "mps"
        elif torch.cuda.is_available():  # type: ignore
            chosen_device = "cuda"
        else:
            chosen_device = "cpu"

    processor = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
    # Use low memory dtype by default on CPU/MPS
    torch_dtype = "auto"
    try:
        import torch  # type: ignore
        if chosen_device in {"cpu", "mps"}:
            torch_dtype = torch.float16 if chosen_device == "mps" else torch.float32
    except Exception:
        pass

    model_obj = AutoModelForVision2Seq.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        device_map=chosen_device if chosen_device in {"cuda"} else None,
        torch_dtype=torch_dtype,
    )
    # Ensure model on device when not using device_map
    if chosen_device in {"cpu", "mps"}:
        model_obj.to(chosen_device)
    model_obj.eval()
    return AMDQwenVLClient(model=model_obj, processor=processor, device=chosen_device)


def AMD_qwenvl_call(
    client: AMDQwenVLClient,
    image_paths: Iterable[str],
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate caption with Qwen-VL using local Transformers.

    Supports one or multiple images.
    """
    from PIL import Image  # type: ignore
    import torch  # type: ignore

    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception:
            continue
    if not images:
        return ""

    # Build chat-style prompt with image placeholders using the model's chat template.
    # This ensures the number of image tokens matches the provided images.
    messages = [
        {
            "role": "user",
            "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]),
        }
    ]

    # Prefer processor.apply_chat_template; fallback to tokenizer.apply_chat_template; finally raw prompt.
    if hasattr(client.processor, "apply_chat_template"):
        text_for_model = client.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif hasattr(client.processor, "tokenizer") and hasattr(client.processor.tokenizer, "apply_chat_template"):
        text_for_model = client.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text_for_model = prompt

    # Qwen-VL processors accept single image or list of images along with the templated text
    inputs = client.processor(
        images=images if len(images) > 1 else images[0],
        text=text_for_model,
        return_tensors="pt",
    )
    # Move to device if needed
    if client.device in {"cuda", "mps"}:
        inputs = {k: (v.to(client.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = client.model.generate(
            **inputs,
            do_sample=temperature is not None and temperature > 0,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
        )

    # Decode
    if hasattr(client.processor, "decode"):
        text = client.processor.decode(outputs[0], skip_special_tokens=True)
    elif hasattr(client.processor, "tokenizer"):
        text = client.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore
    else:
        text = ""

    # Strip chat template markers if present so only assistant content remains
    cleaned = text
    for marker in (
        "<|im_start|>assistant",
        "<|assistant|>",
        "\nassistant\n",
        "\nassistant:",
        "assistant\n",
        "assistant:",
    ):
        idx = cleaned.rfind(marker)
        if idx != -1:
            cleaned = cleaned[idx + len(marker):].lstrip()
            break

    return cleaned.strip()


__all__ = [
    "AMD_openai_client",
    "AMD_openai_call",
    "AMD_llama_client",
    "AMD_gemini_client",
    "AMD_gemini_call",
    "AMD_claude_client",
    "AMD_claude_call",
    "AMD_vllm_chat_client",
    "AMD_vllm_text_chat_client",  # Backward compatibility alias
    "AMD_vllm_text_chat_call",
    "AMD_vllm_multimodal_call",
    "AMD_vllm_structured_multimodal_call",
    "AMD_vllm_server_client",
    "AMD_vllm_server_multimodal_call",
    "AMD_vllm_server_text_chat_call",
    "AMD_qwenvl_client",
    "AMD_qwenvl_call",
]
# python /weka/yunong/projects/CaptionQA/qa.py \
#   --caption-path "/weka/yunong/projects/CaptionQA/captions/e-commerce/short/Qwen_Qwen2.5-VL-72B-Instruct.json" \
#   --question-path "/weka/yunong/projects/CaptionQA/questions/ecommerce_ieval_good_accept.json" \
#   --output-path "/weka/yunong/projects/CaptionQA/results/e-commerce/taxonomy_structured/Qwen_Qwen2.5-VL-72B-Instruct.json" \
#   --model gpt-4o \
#   --save-every 50 \
#   --no-amd
# Natural, long, Qwen2.5-VL-72B-Instruct, gpt-4o: [progress] processed=500 | total_score=315.95 | avg_score=0.6319 | accuracy=59.00% | cannot_answer=69
# Natural, Taxonomy Default, Qwen2.5-VL-72B-Instruct, gpt-4o: [progress] processed=500 | total_score=309.80 | avg_score=0.6196 | accuracy=58.60% | cannot_answer=56 
# Natural, Taxonomy Structured, Qwen2.5-VL-72B-Instruct, gpt-4o: [progress] processed=500 | total_score=275.67 | avg_score=0.5513 | accuracy=43.60% | cannot_answer=190
# Natural, Short, Qwen2.5-VL-72B-Instruct, gpt-4o: [progress] processed=500 | total_score=278.75 | avg_score=0.5575 | accuracy=44.20% | cannot_answer=190 
# Natural, Simple, Qwen2.5-VL-72B-Instruct, gpt-4o: [progress] processed=500 | total_score=308.88 | avg_score=0.6178 | accuracy=56.60% | cannot_answer=86
# E-Commerce, Taxonomy Default, Qwen2.5-VL-72B-Instruct, gpt-4o: [progress] processed=500 | total_score=315.80 | avg_score=0.6316 | accuracy=59.00% | cannot_answer=69
# E-Commerce, Short, Qwen2.5-VL-72B-Instruct, [progress] processed=500 | total_score=205.12 | avg_score=0.4102 | accuracy=23.00% | cannot_answer=294     
# E-Commerce, Taxonomy Structured, Qwen2.5-VL-72B-Instruct, gpt-4o: [progress] processed=500 | total_score=198.10 | avg_score=0.3962 | accuracy=21.40% | cannot_answer=297
