import argparse
import os
import subprocess
import sys

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


def find_converter(llama_cpp_dir):
    candidates = [
        os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py"),
        os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find GGUF converter in llama.cpp. Expected one of: "
        f"{', '.join(candidates)}"
    )


def find_quantize_binary(llama_cpp_dir):
    candidates = [
        os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize"),
        os.path.join(llama_cpp_dir, "build", "bin", "quantize"),
        os.path.join(llama_cpp_dir, "llama-quantize"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "Could not find llama.cpp quantize binary. Build llama.cpp first, for example: "
        "cmake -S . -B build && cmake --build build -j. "
        f"Looked for: {', '.join(candidates)}"
    )


def run_command(cmd, label):
    print(f"\n==> {label}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


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


def convert_to_gguf(converter_script, merged_out_dir, gguf_out_path, outtype):
    cmd = [
        sys.executable,
        converter_script,
        merged_out_dir,
        "--outfile",
        gguf_out_path,
        "--outtype",
        outtype,
    ]
    run_command(cmd, "Converting merged HF model to GGUF")


def quantize_gguf(quantize_bin, gguf_in_path, gguf_out_path, quantize_type):
    cmd = [
        quantize_bin,
        gguf_in_path,
        gguf_out_path,
        quantize_type,
    ]
    run_command(cmd, f"Quantizing GGUF to {quantize_type}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model, export GGUF, and optionally quantize with llama.cpp."
    )
    parser.add_argument(
        "--base-model-id",
        default="google/gemma-4-26B-A4B-it",
        help="Base model ID or local path used during LoRA training.",
    )
    parser.add_argument(
        "--adapter-dir",
        default="./gemma-hermes-adapter-rocm",
        help="Path to LoRA adapter directory.",
    )
    parser.add_argument(
        "--merged-out-dir",
        default="./gemma-hermes-merged",
        help="Where to save merged HF model.",
    )
    parser.add_argument(
        "--gguf-out",
        default="./gemma-hermes-merged-f16.gguf",
        help="Path for output GGUF file (typically f16/f32).",
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        choices=["f16", "f32", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"],
        help="GGUF outtype used by llama.cpp converter.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default="/home/jamie/llama.cpp",
        help="Path to local llama.cpp checkout.",
    )
    parser.add_argument(
        "--quantize",
        default="",
        help="Optional quantization type (e.g. Q8_0, Q6_K, Q5_K_M). If empty, skip quantization.",
    )
    parser.add_argument(
        "--quantized-out",
        default="",
        help="Optional quantized GGUF output path. Required if --quantize is set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    require_path(args.adapter_dir, "Adapter directory")
    require_path(args.llama_cpp_dir, "llama.cpp directory")

    converter_script = find_converter(args.llama_cpp_dir)

    merge_adapter(
        base_model_id=args.base_model_id,
        adapter_dir=args.adapter_dir,
        merged_out_dir=args.merged_out_dir,
    )

    convert_to_gguf(
        converter_script=converter_script,
        merged_out_dir=args.merged_out_dir,
        gguf_out_path=args.gguf_out,
        outtype=args.outtype,
    )

    if args.quantize:
        if not args.quantized_out:
            raise ValueError("--quantized-out is required when --quantize is set")
        quantize_bin = find_quantize_binary(args.llama_cpp_dir)
        quantize_gguf(
            quantize_bin=quantize_bin,
            gguf_in_path=args.gguf_out,
            gguf_out_path=args.quantized_out,
            quantize_type=args.quantize,
        )

    print("\nDone.")
    print(f"Merged HF model: {args.merged_out_dir}")
    print(f"GGUF file: {args.gguf_out}")
    if args.quantize:
        print(f"Quantized GGUF: {args.quantized_out}")


if __name__ == "__main__":
    main()
