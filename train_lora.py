import os
import inspect
import json
import re

ACCELERATOR_BACKEND = os.getenv("ACCELERATOR_BACKEND", "auto").strip().lower()
if ACCELERATOR_BACKEND not in {"auto", "rocm", "cuda", "cpu"}:
    raise ValueError("ACCELERATOR_BACKEND must be one of: auto, rocm, cuda, cpu")

import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

try:
    from trl.trainer.sft_config import SFTConfig
except ImportError:
    SFTConfig = None

# --- ROCm runtime hints ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_float32_matmul_precision("high")

# --- Configuration ---
MODEL_ID = "google/gemma-4-26B-A4B-it"
REASONING_DATASET_ID = os.getenv("REASONING_DATASET_ID", "ansulev/hermes-agent-reasoning-traces")
REASONING_DATASET_CONFIGS = [
    config.strip()
    for config in os.getenv("REASONING_DATASET_CONFIGS", "kimi,glm-5.1").split(",")
    if config.strip()
]
CODE_DATASET_ID = os.getenv("CODE_DATASET_ID", "nvidia/OpenCodeInstruct")
CODE_DATASET_SPLIT = os.getenv("CODE_DATASET_SPLIT", "train")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output/gemma-hermes-reasoning-results-rocm")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "./output/gemma-hermes-adapter-rocm")

# LoRA Hyperparameters
LORA_R = 64
LORA_ALPHA = 128
# Keep dropout at 0 for better throughput on non-FA2 backends.
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    module.strip()
    for module in os.getenv("TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj").split(",")
    if module.strip()
]
if not TARGET_MODULES:
    raise ValueError("TARGET_MODULES must contain at least one module name.")

# Training Hyperparameters
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
NUM_EPOCHS = float(os.getenv("NUM_EPOCHS", "1"))
# 128k is possible but very slow. Start smaller for ROCm throughput and stability.
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "8192"))
ENABLE_CHUNKING = os.getenv("ENABLE_CHUNKING", "false").lower() in {"1", "true", "yes", "y"}
CHUNK_STRIDE = max(1, int(os.getenv("CHUNK_STRIDE", str((MAX_SEQ_LENGTH * 3) // 4))))
CHUNK_BATCH_SIZE = int(os.getenv("CHUNK_BATCH_SIZE", "32"))
CHUNK_NUM_PROC = int(os.getenv("CHUNK_NUM_PROC", "8"))
MIN_CHUNK_TOKENS = int(os.getenv("MIN_CHUNK_TOKENS", "256"))
MIX_DATASETS = os.getenv("MIX_DATASETS", "true").lower() in {"1", "true", "yes", "y"}
MIX_REASONING_RATIO = float(os.getenv("MIX_REASONING_RATIO", "0.5"))
DATASET_SHUFFLE_SEED = int(os.getenv("DATASET_SHUFFLE_SEED", "42"))
CODE_MIN_TEST_SCORE = float(os.getenv("CODE_MIN_TEST_SCORE", "0.9"))
CODE_MAX_SAMPLES = int(os.getenv("CODE_MAX_SAMPLES", "0"))
REPORT_TO = [entry.strip() for entry in os.getenv("REPORT_TO", "tensorboard").split(",") if entry.strip()]
if len(REPORT_TO) == 1 and REPORT_TO[0].lower() == "none":
    REPORT_TO = []
TENSORBOARD_LOG_DIR = os.getenv("TENSORBOARD_LOG_DIR", os.path.join(OUTPUT_DIR, "tensorboard"))
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "auto").strip().lower()
PREFER_FLASH_ATTENTION_3 = os.getenv("PREFER_FLASH_ATTENTION_3", "false").lower() in {"1", "true", "yes", "y"}


def convert_custom_layers(model):
    """
    Replaces Gemma4ClippableLinear wrappers with plain nn.Linear when present.
    """
    print("Checking for custom Gemma4ClippableLinear layers...")

    modules_to_replace = []
    for name, module in model.named_modules():
        if type(module).__name__ == "Gemma4ClippableLinear":
            modules_to_replace.append((name, module))

    if not modules_to_replace:
        print("No custom Gemma4ClippableLinear layers found.")
        return model

    print(f"Found {len(modules_to_replace)} custom layers. Replacing with standard Linear...")

    for name, module in modules_to_replace:
        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        attr_name = parts[-1]
        parent = model if parent_name == "" else model.get_submodule(parent_name)

        if hasattr(module, "linear"):
            setattr(parent, attr_name, module.linear)
        else:
            print(f"Warning: {name} has no .linear attribute. Skipping.")

    print("Layer conversion complete.")
    return model


def formatting_func(example):
    convs = example["conversations"]
    tools_json = (example.get("tools") or "").strip()
    system_messages = []
    serialized_turns = []

    role_map = {
        "system": "system",
        "human": "user",
        "user": "user",
        "gpt": "model",
        "assistant": "model",
        "tool": "user",
    }

    for turn in convs:
        raw_role = turn.get("from")
        role = role_map.get(raw_role)
        content = (turn.get("value") or "").strip()

        if not role or not content:
            continue

        if role == "system":
            system_messages.append(content)
            continue

        if not serialized_turns and role == "user":
            prefix_parts = []
            if system_messages:
                prefix_parts.append("\n\n".join(system_messages))
            if tools_json:
                prefix_parts.append(f"<tools>\n{tools_json}\n</tools>")
            if prefix_parts:
                content = "\n\n".join(prefix_parts + [content])

        serialized_turns.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")

    return {"text": "\n".join(serialized_turns)}


def format_reasoning_example(example):
    return formatting_func(example)


def format_code_example(example):
    prompt = (example.get("input") or "").strip()
    response = (example.get("output") or "").strip()
    system_msg = "You are a careful coding assistant. Produce correct, runnable code when the task asks for code."

    full_prompt = f"<start_of_turn>user\n{system_msg}\n\n{prompt}<end_of_turn>\n"
    full_prompt += f"<start_of_turn>model\n{response}<end_of_turn>"
    return {"text": full_prompt}


def keep_high_scoring_code(example):
    score = example.get("average_test_score")
    if score is None:
        return False
    try:
        return float(score) >= CODE_MIN_TEST_SCORE
    except (TypeError, ValueError):
        return False


def prepare_reasoning_dataset():
    if not REASONING_DATASET_CONFIGS:
        raise ValueError("REASONING_DATASET_CONFIGS must contain at least one dataset config.")

    reasoning_datasets = []
    for config_name in REASONING_DATASET_CONFIGS:
        print(f"Loading reasoning dataset {REASONING_DATASET_ID} ({config_name})...")
        dataset = load_dataset(REASONING_DATASET_ID, config_name, split="train")
        dataset = dataset.map(format_reasoning_example, remove_columns=dataset.column_names)
        print(f"Reasoning rows after formatting for {config_name}: {len(dataset)}")
        reasoning_datasets.append(dataset)

    dataset = reasoning_datasets[0]
    if len(reasoning_datasets) > 1:
        dataset = concatenate_datasets(reasoning_datasets)
    dataset = dataset.shuffle(seed=DATASET_SHUFFLE_SEED)
    print(
        f"Combined reasoning rows after formatting: {len(dataset)} from configs {', '.join(REASONING_DATASET_CONFIGS)}"
    )
    return dataset


def prepare_code_dataset():
    print(
        f"Loading code dataset {CODE_DATASET_ID} ({CODE_DATASET_SPLIT}) with "
        f"average_test_score >= {CODE_MIN_TEST_SCORE}..."
    )
    dataset = load_dataset(CODE_DATASET_ID, split=CODE_DATASET_SPLIT)
    before_filter = len(dataset)
    dataset = dataset.filter(keep_high_scoring_code, desc="Filtering high-scoring code rows")
    print(f"Code rows after score filter: {len(dataset)} / {before_filter}")
    dataset = dataset.map(format_code_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=DATASET_SHUFFLE_SEED)
    if CODE_MAX_SAMPLES > 0:
        limited_size = min(CODE_MAX_SAMPLES, len(dataset))
        dataset = dataset.select(range(limited_size))
        print(f"Code rows after CODE_MAX_SAMPLES cap: {len(dataset)}")
    return dataset


def prepare_training_dataset():
    reasoning_dataset = prepare_reasoning_dataset()
    if not MIX_DATASETS:
        print("MIX_DATASETS disabled. Training with reasoning dataset only.")
        return reasoning_dataset

    code_dataset = prepare_code_dataset()
    if len(code_dataset) == 0:
        raise ValueError("Code dataset is empty after filtering. Lower CODE_MIN_TEST_SCORE or disable MIX_DATASETS.")
    if not (0.0 < MIX_REASONING_RATIO < 1.0):
        raise ValueError("MIX_REASONING_RATIO must be between 0 and 1, exclusive.")

    # Compute max rows that satisfy requested ratio given current dataset sizes.
    max_total_by_reasoning = int(len(reasoning_dataset) / MIX_REASONING_RATIO)
    max_total_by_code = int(len(code_dataset) / (1.0 - MIX_REASONING_RATIO))
    total_mixed_rows = min(max_total_by_reasoning, max_total_by_code)
    if total_mixed_rows < 2:
        raise ValueError(
            "Not enough rows to satisfy MIX_REASONING_RATIO with current filters. "
            "Adjust MIX_REASONING_RATIO or CODE_MIN_TEST_SCORE."
        )

    reasoning_rows = int(total_mixed_rows * MIX_REASONING_RATIO)
    code_rows = total_mixed_rows - reasoning_rows

    reasoning_rows = max(1, min(reasoning_rows, len(reasoning_dataset)))
    code_rows = max(1, min(code_rows, len(code_dataset)))

    reasoning_dataset = reasoning_dataset.select(range(reasoning_rows))
    code_dataset = code_dataset.select(range(code_rows))
    dataset = concatenate_datasets([reasoning_dataset, code_dataset]).shuffle(seed=DATASET_SHUFFLE_SEED)
    print(
        f"Mixed dataset (ratio={MIX_REASONING_RATIO:.2f}): "
        f"{len(reasoning_dataset)} reasoning + {len(code_dataset)} code = {len(dataset)} rows"
    )
    return dataset


class Gemma4TextOnlyCollator:
    """Adds mm_token_type_ids required by Gemma4 text training."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.model_start_ids = tokenizer("<start_of_turn>model\n", add_special_tokens=False)["input_ids"]
        self.end_turn_ids = tokenizer("<end_of_turn>", add_special_tokens=False)["input_ids"]

    @staticmethod
    def _find_subsequence(sequence, subsequence, start_index=0):
        max_start = len(sequence) - len(subsequence) + 1
        for index in range(start_index, max_start):
            if sequence[index : index + len(subsequence)] == subsequence:
                return index
        return -1

    def __call__(self, features):
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, fill_value=-100)

        for row_index, token_row in enumerate(input_ids.tolist()):
            search_start = 0
            while True:
                model_start = self._find_subsequence(token_row, self.model_start_ids, search_start)
                if model_start < 0:
                    break

                content_start = model_start + len(self.model_start_ids)
                turn_end = self._find_subsequence(token_row, self.end_turn_ids, content_start)
                if turn_end < 0:
                    label_end = len(token_row)
                    search_start = len(token_row)
                else:
                    label_end = turn_end + len(self.end_turn_ids)
                    search_start = label_end

                if content_start < label_end:
                    labels[row_index, content_start:label_end] = input_ids[row_index, content_start:label_end]

        batch["labels"] = labels
        # Preserve attention_mask for TRL/transformers trainer compatibility.
        batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        return batch


TURN_PATTERN = re.compile(r"<start_of_turn>.*?<end_of_turn>\n?", re.DOTALL)


def split_text_into_turns(text):
    turns = TURN_PATTERN.findall(text)
    if turns:
        return turns
    return [text]


def slice_turn_to_token_windows(turn_ids):
    windows = []
    for start in range(0, len(turn_ids), CHUNK_STRIDE):
        window = turn_ids[start : start + MAX_SEQ_LENGTH]
        if not window:
            break
        if len(window) < MIN_CHUNK_TOKENS and start > 0:
            break
        windows.append(window)
        if start + MAX_SEQ_LENGTH >= len(turn_ids):
            break
    return windows


def chunk_and_tokenize_batch(batch, tokenizer):
    chunk_input_ids = []
    chunk_attention_masks = []
    overlap_token_budget = max(0, MAX_SEQ_LENGTH - CHUNK_STRIDE)

    for text in batch["text"]:
        turn_texts = split_text_into_turns(text)
        if not turn_texts:
            continue

        tokenized_turns = [
            tokenizer(turn_text, add_special_tokens=False, truncation=False)["input_ids"]
            for turn_text in turn_texts
        ]

        current_chunk = []
        current_length = 0

        def flush_chunk():
            nonlocal current_chunk, current_length
            if current_chunk and current_length >= MIN_CHUNK_TOKENS:
                merged = []
                for turn_ids in current_chunk:
                    merged.extend(turn_ids)
                chunk_input_ids.append(merged)
                chunk_attention_masks.append([1] * len(merged))

                if overlap_token_budget > 0:
                    kept_turns = []
                    kept_length = 0
                    for turn_ids in reversed(current_chunk):
                        if kept_length + len(turn_ids) > overlap_token_budget:
                            break
                        kept_turns.insert(0, turn_ids)
                        kept_length += len(turn_ids)
                    current_chunk = kept_turns
                    current_length = kept_length
                else:
                    current_chunk = []
                    current_length = 0

        for turn_ids in tokenized_turns:
            if not turn_ids:
                continue

            if len(turn_ids) > MAX_SEQ_LENGTH:
                flush_chunk()
                for window in slice_turn_to_token_windows(turn_ids):
                    if len(window) >= MIN_CHUNK_TOKENS:
                        chunk_input_ids.append(window)
                        chunk_attention_masks.append([1] * len(window))
                current_chunk = []
                current_length = 0
                continue

            if current_length + len(turn_ids) > MAX_SEQ_LENGTH:
                flush_chunk()

            current_chunk.append(turn_ids)
            current_length += len(turn_ids)

        flush_chunk()

    return {
        "input_ids": chunk_input_ids,
        "attention_mask": chunk_attention_masks,
    }


def detect_runtime_backend():
    if not torch.cuda.is_available():
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "cuda"


def flash_attention_3_supported(runtime_backend):
    if runtime_backend != "cuda" or not torch.cuda.is_available():
        return False, "CUDA runtime not available"

    major, minor = torch.cuda.get_device_capability(0)
    # Current flash-attn-3 wheels are generally Hopper-targeted (sm90+).
    if major < 9:
        return False, f"compute capability sm_{major}{minor} does not support this flash-attn-3 build"
    return True, "supported"


def resolve_attention_impl(runtime_backend):
    fa3_ok, fa3_reason = flash_attention_3_supported(runtime_backend)

    if ATTN_IMPLEMENTATION != "auto":
        if ATTN_IMPLEMENTATION == "flash_attention_3" and not fa3_ok:
            print(
                "Requested ATTN_IMPLEMENTATION=flash_attention_3 but it is incompatible with this GPU: "
                f"{fa3_reason}. Falling back to flash_attention_2 -> sdpa."
            )
            return ["flash_attention_2", "sdpa"]
        if ATTN_IMPLEMENTATION == "flash_attention_3":
            print("Requested ATTN_IMPLEMENTATION=flash_attention_3; enabling fallback chain flash_attention_2 -> sdpa")
            return ["flash_attention_3", "flash_attention_2", "sdpa"]
        return [ATTN_IMPLEMENTATION]

    # CUDA long-context runs are significantly more stable with flash-attn kernels.
    if runtime_backend == "cuda":
        if PREFER_FLASH_ATTENTION_3 and fa3_ok:
            return ["flash_attention_3", "flash_attention_2", "sdpa"]
        if PREFER_FLASH_ATTENTION_3 and not fa3_ok:
            print(f"Skipping flash_attention_3 on this CUDA device: {fa3_reason}")
        elif not PREFER_FLASH_ATTENTION_3:
            print("PREFER_FLASH_ATTENTION_3 is disabled; using flash_attention_2 -> sdpa")
        return ["flash_attention_2", "sdpa"]
    return ["sdpa"]


def configure_cuda_sdpa_kernels(allow_math=False):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(allow_math)
    print(
        "CUDA SDPA kernels: "
        f"flash=True, mem_efficient=True, math={allow_math}"
    )


def probe_attention_runtime(model, runtime_backend):
    """Run a tiny forward pass to catch runtime kernel incompatibilities early."""
    if runtime_backend != "cuda":
        return True, None

    device = next(model.parameters()).device
    test_input_ids = torch.tensor([[1, 2, 3, 4]], device=device, dtype=torch.long)
    test_attention_mask = torch.ones_like(test_input_ids)

    try:
        with torch.inference_mode():
            _ = model(input_ids=test_input_ids, attention_mask=test_attention_mask, use_cache=False)
        return True, None
    except Exception as exc:
        return False, str(exc)


def load_tokenizer(model_id):
    tokenizer_kwargs = {}

    # Some Gemma tokenizer configs publish extra_special_tokens as a list, while
    # certain transformers builds expect a dict. Normalize that here so Colab
    # environments with newer Python / mixed package versions keep working.
    tokenizer_config_path = hf_hub_download(repo_id=model_id, filename="tokenizer_config.json")
    with open(tokenizer_config_path, encoding="utf-8") as file_obj:
        tokenizer_config = json.load(file_obj)

    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if isinstance(extra_special_tokens, list):
        additional_special_tokens = list(tokenizer_config.get("additional_special_tokens") or [])
        for token in extra_special_tokens:
            if token not in additional_special_tokens:
                additional_special_tokens.append(token)
        tokenizer_kwargs["additional_special_tokens"] = additional_special_tokens
        tokenizer_kwargs["extra_special_tokens"] = {}

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **tokenizer_kwargs)


def train():
    runtime_backend = detect_runtime_backend()
    print(f"Starting fine-tuning for {MODEL_ID}")
    print(f"Accelerator backend: requested={ACCELERATOR_BACKEND}, detected={runtime_backend}")
    print(f"MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}, BATCH_SIZE={BATCH_SIZE}, GRAD_ACC={GRADIENT_ACCUMULATION_STEPS}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, target_modules={','.join(TARGET_MODULES)}")
    print(
        f"ENABLE_CHUNKING={ENABLE_CHUNKING}, CHUNK_STRIDE={CHUNK_STRIDE}, "
        f"CHUNK_BATCH_SIZE={CHUNK_BATCH_SIZE}, CHUNK_NUM_PROC={CHUNK_NUM_PROC}"
    )
    attention_candidates = resolve_attention_impl(runtime_backend)
    print(
        "Attention backend candidates: "
        f"{attention_candidates} (ATTN_IMPLEMENTATION={ATTN_IMPLEMENTATION})"
    )
    if runtime_backend == "cuda" and "sdpa" in attention_candidates:
        # Prefer flash/mem-efficient kernels first; enable math fallback only
        # when runtime probing reports no valid backend.
        configure_cuda_sdpa_kernels(allow_math=False)
    print(
        f"Reporting backends={REPORT_TO if REPORT_TO else ['none']}, "
        f"tensorboard_log_dir={TENSORBOARD_LOG_DIR}"
    )

    tokenizer = load_tokenizer(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = prepare_training_dataset()

    if ENABLE_CHUNKING:
        original_rows = len(dataset)
        print("Chunking and tokenizing dataset with sliding token windows...")
        try:
            dataset = dataset.map(
                lambda batch: chunk_and_tokenize_batch(batch, tokenizer),
                batched=True,
                batch_size=CHUNK_BATCH_SIZE,
                num_proc=CHUNK_NUM_PROC,
                remove_columns=["text"],
                desc=f"Chunking dataset (num_proc={CHUNK_NUM_PROC})",
            )
        except Exception as e:
            print(f"Parallel chunking failed ({e}). Falling back to num_proc=1.")
            dataset = dataset.map(
                lambda batch: chunk_and_tokenize_batch(batch, tokenizer),
                batched=True,
                batch_size=CHUNK_BATCH_SIZE,
                num_proc=1,
                remove_columns=["text"],
                desc="Chunking dataset (num_proc=1)",
            )
        print(f"Chunked dataset rows: {len(dataset)} (from {original_rows})")

    print("Loading model in BF16...")
    model = None
    last_exception = None
    for attention_impl in attention_candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=attention_impl,
                trust_remote_code=True,
            )
            ok, probe_error = probe_attention_runtime(model, runtime_backend)
            if not ok and attention_impl == "sdpa" and "Invalid backend" in (probe_error or ""):
                print("SDPA runtime probe reported Invalid backend. Retrying with math SDPA enabled.")
                configure_cuda_sdpa_kernels(allow_math=True)
                ok, probe_error = probe_attention_runtime(model, runtime_backend)
            if not ok:
                print(
                    f"Runtime probe failed for attn_implementation={attention_impl}: {probe_error}"
                )
                del model
                model = None
                if runtime_backend == "cuda":
                    torch.cuda.empty_cache()
                continue
            print(f"Loaded model with attn_implementation={attention_impl}")
            break
        except Exception as exc:
            last_exception = exc
            print(f"Failed to load with attn_implementation={attention_impl}: {exc}")

    if model is None:
        if attention_candidates != ["sdpa"]:
            print("All requested attention backends failed. Final fallback: attn_implementation=sdpa")
            configure_cuda_sdpa_kernels(allow_math=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            ok, probe_error = probe_attention_runtime(model, runtime_backend)
            if not ok:
                raise RuntimeError(
                    f"Final SDPA fallback failed runtime probe: {probe_error}"
                )
        else:
            raise RuntimeError("Unable to load model with requested attention backend(s)") from last_exception

    model = convert_custom_layers(model)

    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

    training_args_kwargs = {
        "output_dir": OUTPUT_DIR,
        "logging_dir": TENSORBOARD_LOG_DIR,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "cosine",
        "num_train_epochs": NUM_EPOCHS,
        "bf16": True,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 5,
        "optim": "adamw_torch",
        "report_to": REPORT_TO if REPORT_TO else "none",
        "gradient_checkpointing": True,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": False,
    }
    supported_training_args = inspect.signature(TrainingArguments.__init__).parameters
    filtered_training_args_kwargs = {
        k: v for k, v in training_args_kwargs.items() if k in supported_training_args
    }
    training_args = TrainingArguments(**filtered_training_args_kwargs)

    # TRL releases disagree on whether push_to_hub_token is present, but some
    # versions unconditionally pop it from args.to_dict(). Always include it.
    original_to_dict = training_args.to_dict

    def compat_to_dict():
        data = original_to_dict()
        data.setdefault("push_to_hub_token", None)
        return data

    training_args.to_dict = compat_to_dict

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    sft_params = inspect.signature(SFTTrainer.__init__).parameters
    if "dataset_text_field" in sft_params and not ENABLE_CHUNKING:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_params and not ENABLE_CHUNKING:
        trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    if "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.data_collator = Gemma4TextOnlyCollator(tokenizer)

    resume_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        resume_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        print("Training started...")
        trainer.train()

    print(f"Saving adapter to {ADAPTER_DIR}...")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print("Training complete.")


if __name__ == "__main__":
    train()
