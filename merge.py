import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_custom_layers(model):
    """Replaces Gemma4ClippableLinear wrappers with plain nn.Linear so PEFT can attach LoRA."""
    modules_to_replace = [
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ == "Gemma4ClippableLinear"
    ]
    for name, module in modules_to_replace:
        parts = name.split(".")
        parent = model if len(parts) == 1 else model.get_submodule(".".join(parts[:-1]))
        if hasattr(module, "linear"):
            setattr(parent, parts[-1], module.linear)
    return model


def require_path(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def merge_adapter(base_model_id, adapter_dir, merged_out_dir):
    print("\n==> Loading base model")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("==> Converting Gemma4ClippableLinear layers to standard Linear")
    base_model = convert_custom_layers(base_model)

    print("==> Loading adapter")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("==> Merging adapter into base model")
    merged_model = peft_model.merge_and_unload()

    print(f"==> Saving merged model to {merged_out_dir}")
    os.makedirs(merged_out_dir, exist_ok=True)
    merged_model.save_pretrained(
        merged_out_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    print("==> Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(merged_out_dir)

    print(f"\nDone. Merged model saved to: {merged_out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter checkpoint into the base model and save as HuggingFace format."
    )
    parser.add_argument(
        "--base-model-id",
        default="google/gemma-4-26B-A4B-it",
        help="Base model ID or local path used during LoRA training.",
    )
    parser.add_argument(
        "--adapter-dir",
        default="./output/adapter",
        help="Path to LoRA adapter directory (or checkpoint subdirectory).",
    )
    parser.add_argument(
        "--merged-out-dir",
        default="./gemma-hermes-merged",
        help="Where to save the merged HuggingFace model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    require_path(args.adapter_dir, "Adapter directory")
    merge_adapter(
        base_model_id=args.base_model_id,
        adapter_dir=args.adapter_dir,
        merged_out_dir=args.merged_out_dir,
    )


if __name__ == "__main__":
    main()
