"""Tests for evaluator module."""
import math
from pipeline.evaluator import EvalResult


def test_eval_result_dataclass():
    result = EvalResult(
        perplexity=15.3,
        loss=math.log(15.3),
        num_samples=100,
        metadata={"model": "test"},
    )
    assert result.perplexity == 15.3
    assert result.num_samples == 100
    assert "model" in result.metadata


def test_perplexity_loss_consistency():
    """Loss should equal log(perplexity)."""
    ppl = 25.0
    result = EvalResult(perplexity=ppl, loss=math.log(ppl), num_samples=50, metadata={})
    assert abs(math.exp(result.loss) - ppl) < 1e-6
