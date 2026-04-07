# unsloth-production-pipeline

Fine-tuning LLaMA 3 with unsloth is 2x faster and uses 60% less VRAM — but a structured pipeline around it keeps your workflow reproducible and testable. This repository provides a modular MLOps pipeline that takes you from raw data to a serving-ready model in five clean stages.

> **Note**: Unsloth now includes built-in export to SafeTensors, GGUF, and direct deployment via vLLM, Ollama, and SGLang. Unsloth Studio adds a web UI for training and serving. This pipeline remains useful for scripted/CI workflows where you want full control over each stage.

---

## Why Unsloth?

Unsloth rewrites the attention kernels and optimizer steps in Triton, achieving dramatic improvements over a vanilla HuggingFace + PEFT setup:

| Metric | HuggingFace + PEFT | Unsloth | Improvement |
|---|---|---|---|
| Training speed | 1x | ~2x | +100% |
| VRAM usage | baseline | ~60% less | -60% |
| Max sequence length | limited by OOM | 2x longer | +100% |
| Gradient checkpointing | standard | smart (unsloth mode) | less recompute |

For a 7B model on a single A100 (80 GB), this means you can train with batch size 8 instead of 4, and complete an epoch on 10k samples in under an hour instead of two.

---

## Pipeline Overview

The pipeline has five stages, each modular and independently testable:

```
Raw Data → [01_data_preparation] → Formatted Dataset
                              ↓
                    [02_training] → Fine-tuned LoRA Weights
                              ↓
                     [03_export] → Merged Model / GGUF File
                              ↓
                    [04_serving] → OpenAI-compatible API
                              ↓
                 [05_evaluation] → Perplexity / Loss Report
```

Each stage reads from the previous stage's output and writes to a clearly defined directory under `outputs/`.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    unsloth-production-pipeline                       │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │ 01_data_prep │───▶│ 02_training  │───▶│  03_export   │           │
│  │              │    │              │    │              │           │
│  │ • load HF    │    │ • unsloth    │    │ • merge LoRA │           │
│  │   dataset    │    │   fast load  │    │ • safetensors│           │
│  │ • format     │    │ • LoRA PEFT  │    │ • GGUF/q4    │           │
│  │   alpaca     │    │ • SFTTrainer │    │ • HF Hub     │           │
│  │ • train/eval │    │ • cosine LR  │    └──────┬───────┘           │
│  │   split      │    │              │           │                   │
│  └──────────────┘    └──────────────┘           │                   │
│                                                 ▼                   │
│                      ┌──────────────┐    ┌──────────────┐           │
│                      │ 05_evaluation│◀───│  04_serving  │           │
│                      │              │    │              │           │
│                      │ • perplexity │    │ • vLLM serve │           │
│                      │ • loss       │    │ • OpenAI API │           │
│                      │ • n_samples  │    │ • benchmarks │           │
│                      │              │    │              │           │
│                      └──────────────┘    └──────────────┘           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/your-org/unsloth-production-pipeline
cd unsloth-production-pipeline
pip install -e ".[dev]"
pip install unsloth  # GPU required
```

### 2. Configure

Edit `configs/training_config.yaml` to set your model, dataset, and hyperparameters.

### 3. Run

```python
from pipeline.data_prep import DataConfig, prepare_dataset
from pipeline.trainer import TrainingConfig, load_model_with_unsloth, run_training
from pipeline.evaluator import evaluate_model
from pipeline.exporter import merge_and_save

# Stage 1: Data
data_cfg = DataConfig(dataset_name="tatsu-lab/alpaca", max_samples=5000)
datasets = prepare_dataset(data_cfg)

# Stage 2: Train
train_cfg = TrainingConfig(model_name="unsloth/llama-3-8b-bnb-4bit")
model, tokenizer = load_model_with_unsloth(train_cfg)
model = run_training(model, tokenizer, datasets["train"], datasets["eval"], train_cfg)

# Stage 3: Evaluate
result = evaluate_model(model, tokenizer, datasets["eval"])
print(f"Perplexity: {result.perplexity:.2f}")

# Stage 4: Export
path = merge_and_save(model, tokenizer, "outputs/merged")
print(f"Saved to: {path}")
```

---

## Configuration YAML Explained

```yaml
model:
  name: "unsloth/llama-3-8b-bnb-4bit"  # Pre-quantized 4-bit checkpoint
  max_seq_length: 2048                  # Longer = more VRAM

data:
  dataset: "tatsu-lab/alpaca"           # Any HuggingFace dataset
  max_samples: 10000                    # Cap for fast iteration
  train_split: 0.9                      # 90% train, 10% eval

training:
  epochs: 1                             # Start with 1, increase if loss plateaus
  batch_size: 2                         # Per GPU; multiply by gradient_accumulation
  gradient_accumulation: 4             # Effective batch = 2 * 4 = 8
  learning_rate: 2.0e-4                 # LoRA standard; lower for larger models
  lora_r: 16                            # Rank; higher = more params but better fit
  lora_alpha: 16                        # Usually equal to r
  lora_dropout: 0.0                      # Unsloth is optimized for dropout=0
  fp16: true                            # Use bf16 on Ampere+ GPUs for stability
  warmup_ratio: 0.03                    # 3% warmup steps
  scheduler: "cosine"                   # Cosine decay works well for LLM fine-tuning

output:
  dir: "outputs/finetuned"
  push_to_hub: false
  export_gguf: false
  gguf_quantization: "q4_k_m"           # Best quality/size tradeoff for GGUF
```

---

## Data Preparation Patterns

The `format_alpaca_prompt` function converts raw instruction data into the Alpaca prompt template, which works well across LLaMA-family models:

```
### Instruction:
{instruction}

### Input:
{optional_input}

### Response:
{output}
```

During inference, leave `output` empty — the model will complete it.

You can swap `prepare_dataset` for any dataset with `instruction`, `input`, and `output` columns, or write a custom `format_sample` function for chat-format data (ShareGPT, OpenHermes, etc.).

---

## Training Config: LoRA r/alpha and Gradient Accumulation

**LoRA rank (`r`)**: Controls how many trainable parameters are added. `r=16` adds ~2M parameters to a 7B model. Higher rank (32, 64) captures more task-specific knowledge but increases VRAM and training time. Start with 16.

**LoRA alpha**: Scaling factor. Setting `alpha = r` is the standard starting point. Increasing alpha (e.g., `alpha = 2*r`) can help if the model underfits.

**Gradient accumulation**: Simulates a larger batch size without the VRAM cost. `batch_size=2, gradient_accumulation=4` gives an effective batch of 8. Larger effective batches generally produce more stable training but may slow convergence.

**Target modules**: The pipeline targets all attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and the MLP layers (`gate_proj`, `up_proj`, `down_proj`). This is the full LoRA coverage recommended by Unsloth.

---

## Evaluation Metrics

The evaluator computes **perplexity**, the standard language model metric:

```
perplexity = exp(average cross-entropy loss per token)
```

Lower is better. A well-fine-tuned 7B model on Alpaca typically achieves perplexity in the range of 5–15 on held-out samples. Perplexity above 50 suggests the model is not adapting to the target distribution.

The `EvalResult` dataclass exposes:
- `perplexity`: The main metric
- `loss`: `log(perplexity)`, equivalent to the raw loss
- `num_samples`: How many samples were evaluated
- `metadata`: Extensible dict for custom metrics (ROUGE, BLEU, task-specific scores)

---

## Export Formats

### SafeTensors (default)

The merged model is saved in HuggingFace SafeTensors format, compatible with `transformers.AutoModelForCausalLM.from_pretrained`. This is the recommended format for serving with vLLM, TGI, or any HuggingFace-compatible inference server.

```python
merge_and_save(model, tokenizer, "outputs/merged")
```

### GGUF (for llama.cpp)

GGUF is the format used by `llama.cpp`, `ollama`, and `LM Studio`. It enables CPU inference and efficient quantized serving on consumer hardware.

```python
from pipeline.exporter import export_gguf
export_gguf(model, tokenizer, "outputs/model.gguf", quantization="q4_k_m")
```

Available quantization levels:
- `q4_k_m`: Best quality/size ratio for 4-bit (recommended)
- `q5_k_m`: Slightly better quality, 25% larger
- `q8_0`: Near-lossless, 2x size of q4
- `f16`: Full precision, for benchmarking only

### HuggingFace Hub

```python
merge_and_save(model, tokenizer, "outputs/merged", push_to_hub=True, hub_model_id="your-org/model-name")
```

---

## Hardware Requirements

| Model Size | Minimum VRAM (4-bit) | Recommended VRAM | GPU Examples |
|---|---|---|---|
| 3B | 4 GB | 8 GB | RTX 3060, T4 |
| 7B / 8B | 6 GB | 12 GB | RTX 3080, A10G |
| 13B | 10 GB | 16 GB | RTX 4080, A100-40G |
| 34B | 22 GB | 40 GB | A100-40G, 2x RTX 4090 |
| 70B | 40 GB | 80 GB | A100-80G, H100 |

All VRAM figures are for 4-bit quantization with Unsloth. Standard HuggingFace + PEFT requires approximately 150% more VRAM for the same model (2.5× baseline).

Multi-GPU training is supported via HuggingFace `accelerate`. Run `accelerate config` and set `num_processes` before launching.

---

## Alignment Techniques (DPO/ORPO)

This pipeline focuses on supervised fine-tuning (SFT). For alignment after SFT, Unsloth supports DPO, ORPO, KTO, and GRPO via TRL's `DPOTrainer` / `ORPOTrainer`. See the [Unsloth preference optimization guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto) for details on running a two-stage SFT + alignment workflow.

---

## Note on Prompt Formats

This pipeline uses the Alpaca prompt format (`### Instruction / ### Input / ### Response`), which works well for base model fine-tuning. For instruct-tuned models (e.g., `unsloth/Meta-Llama-3.1-8B-Instruct`), consider using the model's native chat template (ChatML or Llama chat format) instead. Unsloth's `get_chat_template()` helper can apply the correct format automatically.

---

## Related Repos

- [unsloth/unsloth](https://github.com/unslothai/unsloth) — The core library this pipeline wraps
- [huggingface/trl](https://github.com/huggingface/trl) — SFTTrainer and RLHF utilities
- [huggingface/peft](https://github.com/huggingface/peft) — LoRA and other parameter-efficient methods
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF inference engine
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — High-throughput serving for merged models
- [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) — Original Alpaca dataset and prompt format

---

## License

MIT — see [LICENSE](LICENSE) for details.
