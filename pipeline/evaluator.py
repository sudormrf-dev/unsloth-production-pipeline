"""Model evaluation: perplexity, ROUGE, custom metrics."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class EvalResult:
    """Results from model evaluation."""
    perplexity: float
    loss: float
    num_samples: int
    metadata: dict[str, Any]


def compute_perplexity(model: Any, tokenizer: Any, texts: list[str], device: str = "cuda") -> float:
    """Compute perplexity on a list of texts.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        texts: List of text samples to evaluate.
        device: Device to run on.

    Returns:
        Perplexity score (lower is better).
    """
    import torch

    model.eval()
    total_loss = 0.0
    n_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            seq_len = inputs["input_ids"].shape[-1]
            total_loss += loss * seq_len
            n_tokens += seq_len

    avg_loss = total_loss / n_tokens if n_tokens > 0 else float("inf")
    return math.exp(avg_loss)


def evaluate_model(
    model: Any,
    tokenizer: Any,
    eval_dataset: Any,
    n_samples: int = 100,
) -> EvalResult:
    """Run full evaluation on eval dataset.

    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        eval_dataset: Evaluation dataset with 'text' column.
        n_samples: Number of samples to evaluate.

    Returns:
        EvalResult with perplexity and loss.
    """
    texts = [row["text"] for row in eval_dataset.select(range(min(n_samples, len(eval_dataset))))]
    perplexity = compute_perplexity(model, tokenizer, texts)

    return EvalResult(
        perplexity=perplexity,
        loss=math.log(perplexity) if perplexity > 0 else float("inf"),
        num_samples=len(texts),
        metadata={"model": "evaluated"},
    )
