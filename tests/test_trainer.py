"""Tests for trainer module — TrainingConfig and mocked training functions."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.trainer import TrainingConfig
from pipeline.exporter import merge_and_save, export_gguf
from pipeline.data_prep import prepare_dataset, DataConfig


# ── TrainingConfig ───────────────────────────────────────────


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.model_name == "unsloth/llama-3-8b-bnb-4bit"
        assert cfg.output_dir == "outputs/finetuned"
        assert cfg.num_train_epochs == 1
        assert cfg.per_device_train_batch_size == 2
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.learning_rate == 2e-4
        assert cfg.max_seq_length == 2048
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.0
        assert cfg.fp16 is True
        assert cfg.logging_steps == 10
        assert cfg.save_steps == 100
        assert cfg.warmup_ratio == 0.03
        assert cfg.lr_scheduler_type == "cosine"

    def test_custom_values(self):
        cfg = TrainingConfig(
            model_name="custom/model",
            num_train_epochs=3,
            learning_rate=1e-5,
            lora_r=32,
            fp16=False,
        )
        assert cfg.model_name == "custom/model"
        assert cfg.num_train_epochs == 3
        assert cfg.learning_rate == 1e-5
        assert cfg.lora_r == 32
        assert cfg.fp16 is False

    def test_partial_override(self):
        cfg = TrainingConfig(max_seq_length=4096, lora_dropout=0.1)
        # Overridden
        assert cfg.max_seq_length == 4096
        assert cfg.lora_dropout == 0.1
        # Defaults preserved
        assert cfg.model_name == "unsloth/llama-3-8b-bnb-4bit"
        assert cfg.num_train_epochs == 1


# ── merge_and_save edge cases ────────────────────────────────


class TestMergeAndSave:
    def test_model_without_merge_and_unload(self):
        """Model without LoRA adapters should be saved directly."""

        class PlainModel:
            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

        class FakeTok:
            def save_pretrained(self, path):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "plain"
            result = merge_and_save(PlainModel(), FakeTok(), str(out))
            assert result == str(out)

    def test_push_to_hub_called(self):
        """When push_to_hub=True and hub_model_id provided, push methods called."""
        model = MagicMock()
        model.merge_and_unload.return_value = model
        tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            merge_and_save(model, tokenizer, tmpdir, push_to_hub=True, hub_model_id="user/model")
            model.push_to_hub.assert_called_once_with("user/model")
            tokenizer.push_to_hub.assert_called_once_with("user/model")

    def test_push_skipped_without_hub_id(self):
        """push_to_hub=True but hub_model_id=None should skip push."""
        model = MagicMock()
        model.merge_and_unload.return_value = model
        tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            merge_and_save(model, tokenizer, tmpdir, push_to_hub=True, hub_model_id=None)
            model.push_to_hub.assert_not_called()


# ── export_gguf ──────────────────────────────────────────────


class TestExportGguf:
    def test_export_gguf_calls_method(self):
        model = MagicMock()
        tokenizer = MagicMock()
        result = export_gguf(model, tokenizer, "/tmp/out.gguf", quantization="q5_k_m")
        model.save_pretrained_gguf.assert_called_once_with(
            "/tmp/out.gguf", tokenizer, quantization_method="q5_k_m"
        )
        assert result == "/tmp/out.gguf"

    def test_export_gguf_missing_method_raises(self):
        model = MagicMock()
        model.save_pretrained_gguf.side_effect = AttributeError("no gguf")
        tokenizer = MagicMock()
        with pytest.raises(RuntimeError, match="GGUF export requires Unsloth"):
            export_gguf(model, tokenizer, "/tmp/out.gguf")


# ── prepare_dataset with mock ────────────────────────────────


class TestPrepareDataset:
    def test_prepare_dataset_basic(self):
        """Test prepare_dataset with mocked datasets library."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda self: 3
        mock_dataset.select.return_value = mock_dataset

        def mock_map(fn):
            return mock_dataset

        mock_dataset.map = mock_map

        split_result = MagicMock()
        split_result.__getitem__ = lambda self, key: f"split_{key}"
        mock_dataset.train_test_split.return_value = split_result

        with patch.dict(sys.modules, {"datasets": MagicMock()}):
            import datasets
            datasets.load_dataset = MagicMock(return_value=mock_dataset)

            config = DataConfig(max_samples=2)
            result = prepare_dataset(config)
            assert "train" in result
            assert "eval" in result

    def test_prepare_dataset_no_max_samples(self):
        """Without max_samples, select should not be called."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda self: 10

        def mock_map(fn):
            return mock_dataset

        mock_dataset.map = mock_map

        split_result = MagicMock()
        split_result.__getitem__ = lambda self, key: f"split_{key}"
        mock_dataset.train_test_split.return_value = split_result

        with patch.dict(sys.modules, {"datasets": MagicMock()}):
            import datasets
            datasets.load_dataset = MagicMock(return_value=mock_dataset)

            config = DataConfig(max_samples=None)
            prepare_dataset(config)
            mock_dataset.select.assert_not_called()
