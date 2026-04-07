"""Export fine-tuned model: merge LoRA, save, push to HuggingFace Hub."""
from __future__ import annotations
from pathlib import Path
from typing import Any


def merge_and_save(
    model: Any,
    tokenizer: Any,
    output_path: str | Path,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
) -> str:
    """Merge LoRA adapters into base model and save.

    Args:
        model: Fine-tuned model with LoRA adapters.
        tokenizer: Tokenizer.
        output_path: Local path to save merged model.
        push_to_hub: Whether to push to HuggingFace Hub.
        hub_model_id: HF Hub model ID (required if push_to_hub=True).

    Returns:
        Path where model was saved.
    """
    try:
        from unsloth import FastLanguageModel
        model = FastLanguageModel.for_inference(model)
    except ImportError:
        pass  # Fall back to standard merge

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Merge LoRA weights
    if hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
    else:
        merged = model

    merged.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    if push_to_hub and hub_model_id:
        merged.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)

    return str(output_path)


def export_gguf(model: Any, tokenizer: Any, output_path: str, quantization: str = "q4_k_m") -> str:
    """Export model to GGUF format for llama.cpp serving.

    Args:
        model: Model to export.
        tokenizer: Tokenizer.
        output_path: Output GGUF file path.
        quantization: GGUF quantization level.

    Returns:
        Path to GGUF file.

    Note:
        Requires unsloth with GGUF support.
    """
    try:
        model.save_pretrained_gguf(output_path, tokenizer, quantization_method=quantization)
    except AttributeError as e:
        raise RuntimeError(f"GGUF export requires Unsloth: {e}") from e
    return output_path
