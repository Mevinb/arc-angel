"""
ChatML formatting and token estimation utilities for Dolphin 3.0 Llama 3.1 8B.
"""
from typing import List, Dict

def estimate_tokens(text: str) -> int:
    """
    Fast, reliable heuristic token estimation (~3.8 characters per token for English).
    Leaves safety margin to avoid context truncation.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.6))

def format_chatml_prompt(messages: List[Dict[str, str]], append_assistant_header: bool = True) -> str:
    """
    Formats a list of messages into ChatML string:
    <|im_start|>system
    ...
    <|im_end|>
    <|im_start|>user
    ...
    <|im_end|>
    <|im_start|>assistant
    """
    formatted_chunks = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        formatted_chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    
    prompt = "\n".join(formatted_chunks)
    if append_assistant_header:
        prompt += "\n<|im_start|>assistant\n"
    return prompt
