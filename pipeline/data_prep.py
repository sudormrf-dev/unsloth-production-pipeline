"""Data preparation for fine-tuning: loading, cleaning, formatting."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class DataConfig:
    """Configuration for dataset preparation."""
    dataset_name: str = "tatsu-lab/alpaca"
    max_seq_length: int = 2048
    train_split: float = 0.9
    seed: int = 42
    max_samples: int | None = None


def format_alpaca_prompt(instruction: str, input_text: str = "", output: str = "") -> str:
    """Format sample as Alpaca-style instruction prompt.

    Args:
        instruction: The instruction for the model.
        input_text: Optional input context.
        output: Expected output (empty during inference).

    Returns:
        Formatted prompt string.

    Raises:
        ValueError: If instruction is empty or whitespace-only.
    """
    if not instruction or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if input_text and input_text.strip():
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def prepare_dataset(config: DataConfig) -> dict[str, Any]:
    """Load and prepare dataset for fine-tuning.

    Args:
        config: Dataset configuration.

    Returns:
        Dict with 'train' and 'eval' dataset splits.

    Note:
        Requires datasets library. Install with: pip install datasets
    """
    from datasets import load_dataset

    dataset = load_dataset(config.dataset_name, split="train", revision="main")  # nosec B615

    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    # Format prompts
    def format_sample(sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": format_alpaca_prompt(
                instruction=sample.get("instruction", ""),
                input_text=sample.get("input", ""),
                output=sample.get("output", ""),
            )
        }

    formatted = dataset.map(format_sample)

    # Split
    split = formatted.train_test_split(
        test_size=1 - config.train_split,
        seed=config.seed,
    )
    return {"train": split["train"], "eval": split["test"]}
