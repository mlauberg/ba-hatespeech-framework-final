"""
Meta-Prompting Optimizer Module.

This module implements an error-driven prompt refinement strategy using meta-prompting
for hate speech detection. The approach analyzes classification errors and generates
improved prompts through iterative refinement.

Methodology:
    The optimizer uses a local language model (via Ollama) to perform meta-prompting:
    analyzing false positives and false negatives, then generating refined system
    prompts with strategic few-shot examples to address systematic errors.

Theoretical Foundation:
    This implementation uses generic error-driven meta-prompting inspired by principles
    from automated prompt engineering literature:

    - Error-driven refinement: Similar in spirit to OPRO (Yang et al., 2023), but
      simplified for single-prompt evolution rather than population-based optimization
    - Meta-prompting: Using an LLM to improve prompts for another LLM
    - Active learning: Strategic few-shot example selection based on error analysis

    Related Work:
    - Yang et al. (2023): "Large Language Models as Optimizers" (OPRO)
    - Shinn et al. (2023): "Reflexion: Language Agents with Verbal Reinforcement Learning"
    - Zhou et al. (2023): "Large Language Models Are Human-Level Prompt Engineers" (APE)

    Note: This is NOT a full OPRO implementation (no prompt population or meta-optimization),
    but rather a simplified error-driven meta-prompting approach suitable for Bachelor's
    thesis scope.

Author: Bachelor Thesis Project
Date: 2025
"""

import logging
from typing import Optional

from openai import OpenAI

MODEL_NAME: str = "gemma3:27b"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def generate_improved_prompt(current_prompt: str, errors: str) -> str:
    """
    Generate an improved classification prompt through meta-prompting.

    This function implements error-driven prompt refinement by using an LLM to
    analyze classification errors and generate improved prompts. The strategy:

    1. Analyzes false negatives (missed hate speech) and adds demonstrative examples
    2. Analyzes false positives (non-hate misclassified as hate) and adds contrastive examples
    3. Creates contrastive pairs to teach context-dependent classification
    4. Maintains scientific rigor by documenting the refinement process

    Methodology:
        The meta-prompt instructs a language model to examine systematic errors and
        generate an improved system prompt with strategic few-shot examples. This
        approach is inspired by meta-prompting and active learning principles.

    Args:
        current_prompt: The current system prompt being used for classification
        errors: Formatted string containing error analysis (FP/FN with examples)

    Returns:
        Improved system prompt as a string. Returns the original prompt unchanged
        if the meta-prompting API call fails.

    Note on Reproducibility:
        Uses temperature=0.7 for creative prompt generation (stochastic by design).

        WHY temperature > 0?
        - Creative exploration: Non-zero temperature enables diverse prompt strategies
          across iterations, preventing local optima in prompt space
        - Practical necessity: Zero-temperature meta-prompting tends to produce
          repetitive, conservative refinements that fail to escape poor prompts

        Reproducibility Strategy:
        - Data splits use random_state=42 (ensures same train/val/test across runs)
        - Evaluation is deterministic (inference uses temperature=0.0)
        - While prompt TRAJECTORY is stochastic, the EVALUATION PROTOCOL is reproducible
        - This follows standard practice in prompt optimization research (Yang et al., 2023)

        Limitation: Exact prompt sequence cannot be reproduced without seed control.
        However, the final prompt's performance on the fixed test set IS reproducible.

    Raises:
        Exception: Logs but does not raise exceptions; falls back to current prompt
    """
    meta_prompt: str = f"""You are an expert in Prompt Engineering for hate speech detection.
Your task: Improve the system prompt based on the classification errors below.

ERROR ANALYSIS:
{errors}

CURRENT PROMPT:
{current_prompt}

IMPROVEMENT STRATEGY (Error-Driven Refinement):
1. FALSE NEGATIVES (missed hate speech):
   - Add few-shot examples demonstrating why similar texts ARE hate speech
   - Include reasoning to explain the classification (e.g., group-targeting language)

2. FALSE POSITIVES (non-hate misclassified as hate):
   - Add negative examples with "NOT-HS" label
   - Explain distinction: "Offensive/rude but not Hate Speech (no group targeting)"

3. CONTRASTIVE PAIRS (context-dependent classification):
   - If model reacts to trigger words (e.g., "Ausländer", "Muslim"), create pairs:
     * Example A: Using word in hate speech context → Label: HS
     * Example B: Using same word in neutral context → Label: NOT-HS
   - This teaches the model that CONTEXT matters, not just word presence

4. MAINTAIN DEFINITION:
   - Keep the core definition: Hate Speech = attacks on groups based on protected characteristics
   - Insults targeting individuals without group reference = NOT hate speech

FORMAT:
Return ONLY the improved system prompt. No explanations, no markdown, no commentary."""

    try:
        # Temperature > 0 enables creative exploration in prompt space (see docstring)
        # Reproducibility is ensured via fixed data splits, not prompt trajectory
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.7,  # Stochastic by design for diverse prompt strategies
            max_tokens=2000
        )

        improved_prompt: str = response.choices[0].message.content.strip()
        logging.info(f"Prompt optimization successful (length: {len(improved_prompt)} chars)")
        return improved_prompt

    except Exception as e:
        logging.error(f"Prompt optimization failed: {e}")
        logging.warning("Returning original prompt unchanged")
        return current_prompt
