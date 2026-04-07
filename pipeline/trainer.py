"""Fine-tuning trainer using Unsloth + TRL SFTTrainer."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning run."""
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    output_dir: str = "outputs/finetuned"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0  # Unsloth optimizes for dropout=0; use >0 only if overfitting
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"


def load_model_with_unsloth(config: TrainingConfig) -> tuple[Any, Any]:
    """Load model and tokenizer using Unsloth for 2x faster training.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ImportError: If unsloth is not installed.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            "unsloth not installed. Run: pip install unsloth"
        ) from e

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto detect
        load_in_4bit=True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def run_training(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    config: TrainingConfig,
) -> Any:
    """Run SFT training with TRL SFTTrainer.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        config: Training configuration.

    Returns:
        Trained model.
    """
    from trl import SFTTrainer
    from transformers import TrainingArguments

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=config.save_steps,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=args,
    )
    trainer.train()
    return model
