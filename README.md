# Gemma-4-26B PEFT Fine-Tuning (Hermes-Agent Reasoning)

This project implements a Parameter-Efficient Fine-Tuning (PEFT) workflow to improve the format adherence and reasoning capabilities of `google/gemma-4-26B-A4B-it` while preserving more of its base coding ability.

The default training recipe mixes both configs from `ansulev/hermes-agent-reasoning-traces` (`kimi` and `glm-5.1`) with high-scoring rows from `nvidia/OpenCodeInstruct` in a configurable ratio. The intent is to teach the model the reasoning-and-tool-call pattern while retaining stronger code generation behavior.

## 🛠 Hardware & Environment

- **Target Hardware**: AMD Strix Halo (120GB Unified Memory)
- **Compute Strategy**: Standard LoRA in **BF16** (No quantization). 
    - *Reasoning*: With 120GB, we can avoid the precision loss of 4-bit quantization (QLoRA), which is critical for maintaining the exact syntax of tool calls.
- **Software Environment**: 
    - Conda environment: `finetune`
    - Backend: PyTorch with **ROCm** or **CUDA** support.

## 📂 Project Structure

All scripts are located in `~/rocm_finetune/`:

- `inspect_dataset.py`: Utility to explore dataset schema plus character/token length stats and truncation rates.
- `train_lora.py`: Primary trainer with mixed reasoning/code data, optional sliding-window chunking, auto-resume, and backend-aware ROCm/CUDA support.
- `train.sh`: Recommended launcher with backend auto-detection, TensorBoard startup, resume preflight, and model-name-based output directories.
- `merge_and_export_gguf.py`: Merges LoRA into base model, exports GGUF, and can quantize via llama.cpp.
- `test_gguf_inference.py`: Runs llama.cpp `llama-cli` prompts against a GGUF and checks required tags.
- `test_inference.py`: A testing script to load the base model + adapter and verify format adherence.

## 🚀 Training Workflow

### 1. Preparation
Ensure your conda environment is active and dependencies are installed:
```bash
conda activate finetune
pip install -r requirements-rocm.txt
```

If you are on CUDA/NVIDIA instead of ROCm/AMD:
```bash
conda activate finetune
pip install -r requirements-cuda.txt
```

### 2. Dataset Inspection
Run this first to confirm the conversation structure:
```bash
python inspect_dataset.py
```

Inspect several sequence-length thresholds in one pass:
```bash
MAX_SEQ_LENGTHS=8192,16384,32768,65536,131072 TOKENIZE_NUM_PROC=16 TOKENIZE_BATCH_SIZE=64 python inspect_dataset.py
```

### 3. Start Training
Use the launcher for the default training recipe.
```bash
./train.sh
```

Direct trainer invocation still works when you want to override env vars manually:
```bash
conda run -n finetune python train_lora.py
```

Default mixed-data ROCm run details:
- both Hermes reasoning configs (`kimi` and `glm-5.1`) are loaded and concatenated by default
- 50/50 reasoning/code mix by row count
- code rows filtered by `average_test_score >= 0.8`
- both datasets normalized into the same Gemma chat template before training

Chunked ROCm run (example):
```bash
ENABLE_CHUNKING=true MAX_SEQ_LENGTH=32768 CHUNK_STRIDE=24576 OUTPUT_DIR=./output/gemma-hermes-reasoning-results-rocm-chunk32k ADAPTER_DIR=./output/gemma-hermes-adapter-rocm-chunk32k conda run -n finetune python train_lora.py
```

More conservative mixed run with stricter code-quality filtering:
```bash
CODE_MIN_TEST_SCORE=1.0 conda run -n finetune python train_lora.py
```

Force a backend on systems that expose more than one accelerator stack:
```bash
ACCELERATOR_BACKEND=cuda ./train.sh
```

**Hyperparameters used:**
| Parameter | Value |
| :--- | :--- |
| **Method** | LoRA (BF16) |
| **Rank (r)** | 64 |
| **Alpha** | 128 |
| **Target Modules** | Attention-only by default (`q_proj`, `k_proj`, `v_proj`, `o_proj`) |
| **Learning Rate** | 1e-4 |
| **Batch Size** | 2 (with 8 gradient accumulation steps) |
| **Max Seq Length** | 8192 default (`train_lora.py`) |

## ⚙️ Configuration Knobs

### `train_lora.py` env vars

Core training:

- `BATCH_SIZE` (default: `2`)
- `GRADIENT_ACCUMULATION_STEPS` (default: `8`)
- `LEARNING_RATE` (default: `1e-4`)
- `NUM_EPOCHS` (default: `3`)
- `MAX_SEQ_LENGTH` (default: `8192`)
- `OUTPUT_DIR` (default: `./output/gemma-hermes-reasoning-results-rocm`)
- `ADAPTER_DIR` (default: `./output/gemma-hermes-adapter-rocm`)
- `ACCELERATOR_BACKEND` (default: `auto`) one of `auto`, `rocm`, `cuda`, `cpu`
- `TARGET_MODULES` (default: `q_proj,k_proj,v_proj,o_proj`) comma-separated LoRA target modules

Dataset mixing:

- `MIX_DATASETS` (default: `true`) enables 50/50 mixing of reasoning and code data
- `REASONING_DATASET_ID` (default: `ansulev/hermes-agent-reasoning-traces`)
- `REASONING_DATASET_CONFIGS` (default: `kimi,glm-5.1`) comma-separated reasoning configs to load and combine
- `CODE_DATASET_ID` (default: `nvidia/OpenCodeInstruct`)
- `CODE_DATASET_SPLIT` (default: `train`)
- `CODE_MIN_TEST_SCORE` (default: `0.8`) filters code rows by `average_test_score`
- `CODE_MAX_SAMPLES` (default: `0`) optional cap before balancing
- `DATASET_SHUFFLE_SEED` (default: `42`)

Chunking:

- `ENABLE_CHUNKING` (default: `false`)
- `CHUNK_STRIDE` (default: `MAX_SEQ_LENGTH * 3 / 4`)
- `CHUNK_BATCH_SIZE` (default: `32`)
- `CHUNK_NUM_PROC` (default: `4`)
- `MIN_CHUNK_TOKENS` (default: `256`)

### `inspect_dataset.py` env vars

- `MODEL_ID` (default: `google/gemma-4-26B-A4B-it`) for tokenizer used in token-length stats
- `MAX_SEQ_LENGTH` (default: `8192`) single threshold for truncation reporting
- `MAX_SEQ_LENGTHS` (optional, comma-separated) evaluate many thresholds in one run
- `TOKENIZE_NUM_PROC` (default: `8`) parallel workers for token-length computation
- `TOKENIZE_BATCH_SIZE` (default: `32`) tokenizer batch size during stats pass

### 4. Verification
After training completes, the adapter will be saved under `./output/` by default. Run the inference test:
```bash
conda run -n finetune python test_inference.py
```

### 5. Merge + GGUF Export (llama.cpp)
You can merge the adapter into base weights and export GGUF using your local llama.cpp checkout.

Basic merge + F16 GGUF export:
```bash
conda run -n finetune python merge_and_export_gguf.py \
    --adapter-dir ./output/gemma-hermes-adapter-rocm \
    --merged-out-dir ./output/gemma-hermes-merged \
    --gguf-out ./output/gemma-hermes-merged-f16.gguf \
    --llama-cpp-dir /home/jamie/llama.cpp
```

Merge + export + quantize in one run:
```bash
conda run -n finetune python merge_and_export_gguf.py \
    --adapter-dir ./output/gemma-hermes-adapter-rocm \
    --merged-out-dir ./output/gemma-hermes-merged \
    --gguf-out ./output/gemma-hermes-merged-f16.gguf \
    --quantize Q6_K \
    --quantized-out ./output/gemma-hermes-merged-q6_k.gguf \
    --llama-cpp-dir /home/jamie/llama.cpp
```

If quantization binary is missing, build llama.cpp first:
```bash
cd /home/jamie/llama.cpp
cmake -S . -B build
cmake --build build -j
```

### 6. GGUF Sanity Check (llama.cpp)
Quickly validate that the exported GGUF still emits expected structured tags:
```bash
conda run -n finetune python test_gguf_inference.py \
    --model ./output/gemma-hermes-merged-q6_k.gguf \
    --llama-cli /home/jamie/llama.cpp/build/bin/llama-cli \
    --require-tags "<think>,<tool_call>"
```

Useful options:

- `--ctx-size` (default: `8192`)
- `--n-predict` (default: `384`)
- `--temperature` (default: `0.2`)
- `--threads` (default: about half your CPU cores)
- `--gpu-layers` (default: `999`)
- `--json-out` (optional path) writes a structured pass/fail report for regression tracking

Example with JSON report:
```bash
conda run -n finetune python test_gguf_inference.py \
    --model ./gemma-hermes-merged-q6_k.gguf \
    --llama-cli /home/jamie/llama.cpp/build/bin/llama-cli \
    --require-tags "<think>,<tool_call>" \
    --json-out ./reports/gguf_sanity.json
```

## 🎯 Success Criteria

When evaluating the output of `test_inference.py`, look for:
1. **Reasoning**: A clear `<think>` block preceding the action.
2. **Tool Syntax**: Correct, valid syntax for the tool call (e.g., JSON inside `<tool_call>` tags).
3. **Template Adherence**: Correct use of `<start_of_turn>` and `<end_of_turn>` tokens.

## ⚠️ Troubleshooting

- **OOM Errors**: Although 120GB is plenty, if you encounter OOM, increase `gradient_accumulation_steps` or slightly reduce `MAX_SEQ_LENGTH`.
- **Format Drift**: If the model fails to follow the format, consider increasing the Rank (`r`) to 128 or decreasing the learning rate to `5e-5`.
- **AMD/ROCm Issues**: Ensure `torch.cuda.is_available()` returns `True` in your environment to confirm ROCm integration.
