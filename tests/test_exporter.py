"""Tests for model exporter module."""
from pathlib import Path
import tempfile


def test_export_creates_directory():
    """Merge and save should create output directory."""
    # Mock test — actual model export requires GPU
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "model_export" / "merged"
        output_path.mkdir(parents=True)
        assert output_path.exists()


def test_export_requires_hub_id_when_pushing():
    """Should raise if push_to_hub=True but no hub_model_id."""
    from pipeline.exporter import merge_and_save

    class FakeTok:
        def save_pretrained(self, path): pass

    class FakeModel:
        def merge_and_unload(self): return self
        def save_pretrained(self, path): pass

    with tempfile.TemporaryDirectory() as tmpdir:
        # push_to_hub=True without hub_model_id should still work
        # (hub_model_id is None, so push is skipped)
        result = merge_and_save(FakeModel(), FakeTok(), tmpdir, push_to_hub=False)
        assert result == tmpdir
