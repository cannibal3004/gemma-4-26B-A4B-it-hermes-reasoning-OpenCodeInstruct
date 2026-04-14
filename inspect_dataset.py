from datasets import load_dataset
import json
import math
import statistics
import os
from transformers import AutoTokenizer


def sample_text_from_entry(sample):
    if "conversations" in sample and isinstance(sample["conversations"], list):
        parts = []
        for turn in sample["conversations"]:
            role = turn.get("from", "unknown")
            value = turn.get("value", "")
            parts.append(f"[{role}] {value}")
        return "\n".join(parts)
    if "text" in sample and isinstance(sample["text"], str):
        return sample["text"]
    return json.dumps(sample, ensure_ascii=False)


def print_length_stats(split_dataset):
    lengths = []
    longest_idx = -1
    longest_len = -1

    for idx, sample in enumerate(split_dataset):
        sample_text = sample_text_from_entry(sample)
        length = len(sample_text)
        lengths.append(length)
        if length > longest_len:
            longest_len = length
            longest_idx = idx

    if not lengths:
        print("No rows available for length statistics.")
        return

    sorted_lengths = sorted(lengths, reverse=True)
    total = len(sorted_lengths)

    def top_pct_avg(pct):
        count = max(1, math.ceil(total * pct))
        return statistics.mean(sorted_lengths[:count]), count

    top_10_avg, top_10_count = top_pct_avg(0.10)
    top_05_avg, top_05_count = top_pct_avg(0.05)
    top_01_avg, top_01_count = top_pct_avg(0.01)

    print("Length stats (character count over reconstructed sample text):")
    print(f"- rows: {total}")
    print(f"- average length: {statistics.mean(lengths):.2f}")
    print(f"- median length: {statistics.median(lengths):.2f}")
    print(f"- longest length: {max(lengths)}")
    print(f"- shortest length: {min(lengths)}")
    print(f"- top 10% average ({top_10_count} rows): {top_10_avg:.2f}")
    print(f"- top 5% average ({top_05_count} rows): {top_05_avg:.2f}")
    print(f"- top 1% average ({top_01_count} rows): {top_01_avg:.2f}")

    longest_sample = split_dataset[longest_idx]
    preview_text = sample_text_from_entry(longest_sample)
    preview_text = preview_text[:1000]
    print(f"Longest sample index: {longest_idx}")
    print("Longest sample preview (first 1000 chars):")
    print(preview_text)


def print_token_length_stats(split_dataset, tokenizer, max_seq_length):
    tokenize_num_proc = int(os.getenv("TOKENIZE_NUM_PROC", "8"))
    tokenize_batch_size = int(os.getenv("TOKENIZE_BATCH_SIZE", "32"))

    def tokenize_batch(batch):
        if "text" in batch and len(batch["text"]) > 0 and isinstance(batch["text"][0], str):
            texts = batch["text"]
        elif "conversations" in batch:
            texts = []
            for convs in batch["conversations"]:
                parts = []
                for turn in convs:
                    role = turn.get("from", "unknown")
                    value = turn.get("value", "")
                    parts.append(f"[{role}] {value}")
                texts.append("\n".join(parts))
        else:
            texts = []
            row_count = len(next(iter(batch.values())))
            for i in range(row_count):
                row = {k: v[i] for k, v in batch.items()}
                texts.append(sample_text_from_entry(row))

        enc = tokenizer(texts, add_special_tokens=False, truncation=False)
        return {"token_length": [len(ids) for ids in enc["input_ids"]]}

    try:
        tokenized = split_dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=tokenize_batch_size,
            num_proc=tokenize_num_proc,
            remove_columns=split_dataset.column_names,
            desc=f"Tokenizing for length stats (num_proc={tokenize_num_proc})",
        )
    except Exception as e:
        print(f"Parallel tokenization failed ({e}). Falling back to num_proc=1.")
        tokenized = split_dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=tokenize_batch_size,
            num_proc=1,
            remove_columns=split_dataset.column_names,
            desc="Tokenizing for length stats (num_proc=1)",
        )

    token_lengths = tokenized["token_length"]

    if not token_lengths:
        print("No rows available for token statistics.")
        return

    sorted_lengths = sorted(token_lengths, reverse=True)
    total = len(sorted_lengths)
    max_seq_lengths = [max_seq_length]
    max_seq_lengths_env = os.getenv("MAX_SEQ_LENGTHS", "").strip()
    if max_seq_lengths_env:
        max_seq_lengths = [int(v.strip()) for v in max_seq_lengths_env.split(",") if v.strip()]

    def top_pct_avg(pct):
        count = max(1, math.ceil(total * pct))
        return statistics.mean(sorted_lengths[:count]), count

    top_10_avg, top_10_count = top_pct_avg(0.10)
    top_05_avg, top_05_count = top_pct_avg(0.05)
    top_01_avg, top_01_count = top_pct_avg(0.01)

    print("Token stats (tokenizer-based):")
    print(f"- average tokens: {statistics.mean(token_lengths):.2f}")
    print(f"- median tokens: {statistics.median(token_lengths):.2f}")
    print(f"- longest tokens: {max(token_lengths)}")
    print(f"- shortest tokens: {min(token_lengths)}")
    print(f"- top 10% avg tokens ({top_10_count} rows): {top_10_avg:.2f}")
    print(f"- top 5% avg tokens ({top_05_count} rows): {top_05_avg:.2f}")
    print(f"- top 1% avg tokens ({top_01_count} rows): {top_01_avg:.2f}")
    for current_max in max_seq_lengths:
        over_limit = sum(1 for l in token_lengths if l > current_max)
        over_limit_pct = (over_limit / total) * 100.0
        print(f"- rows over MAX_SEQ_LENGTH={current_max}: {over_limit}/{total} ({over_limit_pct:.2f}%)")

def inspect_config(repo, config_name):
    print(f"\n>>> Testing Config: {config_name}")
    try:
        dataset = load_dataset(repo, config_name)
        print(f"Dataset loaded successfully: {dataset}")
        
        split_name = list(dataset.keys())[0]
        print(f"Split: {split_name}")
        print(f"Features: {dataset[split_name].features}")
        
        print("Sample Entry:")
        sample = dataset[split_name][0]
        print(json.dumps(sample, indent=2))
        print_length_stats(dataset[split_name])
        model_id = os.getenv("MODEL_ID", "google/gemma-4-26B-A4B-it")
        max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "8192"))
        print(
            "Tokenization settings: "
            f"TOKENIZE_NUM_PROC={os.getenv('TOKENIZE_NUM_PROC', '8')}, "
            f"TOKENIZE_BATCH_SIZE={os.getenv('TOKENIZE_BATCH_SIZE', '32')}"
        )
        print(f"Loading tokenizer for token stats: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print_token_length_stats(dataset[split_name], tokenizer, max_seq_length)
        return True
    except Exception as e:
        print(f"Failed to load {config_name}: {e}")
        return False

if __name__ == "__main__":
    repo = "ansulev/hermes-agent-reasoning-traces"
    configs_to_try = ["kimi", "glm-5.1"]
    
    for cfg in configs_to_try:
        if inspect_config(repo, cfg):
            break
