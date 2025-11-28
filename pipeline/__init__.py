"""
Dataset tools module - API clients for AMD LLM services.
"""

from .api import (
    AMD_vllm_chat_client,
    AMD_vllm_text_chat_call,
    AMD_vllm_multimodal_call,
    AMD_vllm_server_client,
    AMD_vllm_server_multimodal_call,
    AMD_vllm_server_text_chat_call,
    AMD_openai_client,
    AMD_llama_client,
    AMD_openai_call,
    AMD_gemini_client,
    AMD_gemini_call,
    AMD_claude_client,
    AMD_claude_call,
    current_user,
)

__all__ = [
    "AMD_vllm_chat_client",
    "AMD_vllm_text_chat_call",
    "AMD_vllm_multimodal_call",
    "AMD_vllm_server_client",
    "AMD_vllm_server_multimodal_call",
    "AMD_vllm_server_text_chat_call",
    "AMD_openai_client",
    "AMD_llama_client",
    "AMD_openai_call",
    "AMD_gemini_client",
    "AMD_gemini_call",
    "AMD_claude_client",
    "AMD_claude_call",
    "current_user",
]

