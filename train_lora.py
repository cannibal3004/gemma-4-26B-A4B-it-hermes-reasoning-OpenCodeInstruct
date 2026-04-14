import os
import inspect
import json

ACCELERATOR_BACKEND = os.getenv("ACCELERATOR_BACKEND", "auto").strip().lower()
if ACCELERATOR_BACKEND not in {"auto", "rocm", "cuda", "cpu"}:
    raise ValueError("ACCELERATOR_BACKEND must be one of: auto, rocm, cuda, cpu")

if ACCELERATOR_BACKEND == "rocm":
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

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
    system_msg = ""
    user_msg = ""
    assistant_msg = ""

    for turn in convs:
        if turn["from"] == "system":
            system_msg = turn["value"]
        elif turn["from"] == "user":
            user_msg = turn["value"]
        elif turn["from"] == "assistant":
            assistant_msg = turn["value"]

    full_prompt = f"<start_of_turn>user\n{system_msg}\n\n{user_msg}<end_of_turn>\n"
    full_prompt += f"<start_of_turn>model\n{assistant_msg}<end_of_turn>"
    return {"text": full_prompt}


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
        self.base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, features):
        batch = self.base_collator(features)
        batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        return batch


def chunk_and_tokenize_batch(batch, tokenizer):
    chunk_input_ids = []
    chunk_attention_masks = []

    for text in batch["text"]:
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            continue

        if len(ids) <= MAX_SEQ_LENGTH:
            chunk_input_ids.append(ids)
            chunk_attention_masks.append([1] * len(ids))
            continue

        for start in range(0, len(ids), CHUNK_STRIDE):
            window = ids[start : start + MAX_SEQ_LENGTH]
            if not window:
                break
            if len(window) < MIN_CHUNK_TOKENS and start > 0:
                break
            chunk_input_ids.append(window)
            chunk_attention_masks.append([1] * len(window))
            if start + MAX_SEQ_LENGTH >= len(ids):
                break

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

    print("Loading model in BF16 (ROCm)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

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

    # Compatibility shim: some TRL versions expect this legacy key.
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
