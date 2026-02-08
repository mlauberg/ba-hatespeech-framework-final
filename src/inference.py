"""
Hate Speech Classification Inference Module.

This module provides the core classification functionality for the hate speech
detection system. It uses a local language model (via Ollama) to classify German
text as either hate speech (HS) or not hate speech (NOT-HS) based on a provided
system prompt.

The classification uses zero-temperature sampling for deterministic results and
minimal token generation for efficiency.
"""

import logging
from typing import Dict, Any

from openai import OpenAI

MODEL_NAME: str = "gemma3:4b-it-qat" #gemma3:27b, gemma3:4b-it-qat

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def classify_text(text: str, prompt: str) -> Dict[str, Any]:
    """
    Classify a single text sample as hate speech or not hate speech.

    Uses a local language model via the OpenAI-compatible Ollama API to perform
    binary classification. The function requests JSON-formatted output and parses
    the response to extract the classification label.

    Args:
        text: The German text to classify
        prompt: The system prompt containing classification instructions and examples

    Returns:
        Dictionary with the following keys:
            - 'label' (int): Binary classification result (1 for HS, 0 for NOT-HS)
            - 'error' (bool): Whether an error occurred during inference

    Note:
        If an API error occurs, the function defaults to label 0 (NOT-HS) and
        sets the error flag to True. This conservative default minimizes false
        positives in the error analysis.
    """
    messages: list = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f'Classify this text:\n"{text}"\n\nAnswer (JSON only): {{"class": "HS" or "NOT-HS"}}'}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=10
        )

        output: str = response.choices[0].message.content.strip()

        label: int = 1 if ("HS" in output and "NOT-HS" not in output) else 0

        return {"label": label, "error": False}

    except Exception as e:
        logging.error(f"Inference error for text '{text[:50]}...': {e}")
        return {"label": 0, "error": True}
