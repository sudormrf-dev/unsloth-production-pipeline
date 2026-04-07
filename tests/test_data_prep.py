"""Tests for data preparation module."""
import pytest
from pipeline.data_prep import format_alpaca_prompt, DataConfig


def test_format_alpaca_with_input():
    result = format_alpaca_prompt(
        instruction="Translate to French",
        input_text="Hello",
        output="Bonjour",
    )
    assert "### Instruction:" in result
    assert "Translate to French" in result
    assert "### Input:" in result
    assert "Hello" in result
    assert "### Response:" in result
    assert "Bonjour" in result


def test_format_alpaca_without_input():
    result = format_alpaca_prompt(
        instruction="Write a poem",
        output="Roses are red...",
    )
    assert "### Input:" not in result
    assert "### Instruction:" in result
    assert "Write a poem" in result


def test_format_alpaca_inference_mode():
    """During inference, output is empty."""
    result = format_alpaca_prompt(instruction="What is 2+2?")
    assert "### Response:\n" in result
    assert result.endswith("### Response:\n")


def test_data_config_defaults():
    config = DataConfig()
    assert config.max_seq_length == 2048
    assert config.train_split == 0.9
    assert config.seed == 42


def test_data_config_custom():
    config = DataConfig(
        dataset_name="custom/dataset",
        max_samples=500,
        train_split=0.8,
    )
    assert config.dataset_name == "custom/dataset"
    assert config.max_samples == 500
    assert config.train_split == 0.8


def test_format_alpaca_empty_instruction_raises():
    """Empty instruction string should raise ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        format_alpaca_prompt(instruction="")


def test_format_alpaca_whitespace_instruction_raises():
    """Whitespace-only instruction should raise ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        format_alpaca_prompt(instruction="   ")


def test_format_alpaca_whitespace_input_ignored():
    """Whitespace-only input_text should be treated as empty."""
    result = format_alpaca_prompt(instruction="Do something", input_text="   ")
    assert "### Input:" not in result
