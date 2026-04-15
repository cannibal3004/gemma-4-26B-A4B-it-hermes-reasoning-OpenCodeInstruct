"""
quantize.py — Convert a merged HuggingFace model to one or more output formats.

Supported formats (passed via --formats, space-separated):
  hf              Copy the merged model as-is in HuggingFace safetensors format.
  f16             Convert to GGUF with f16 weights (via llama.cpp converter).
  f32             Convert to GGUF with f32 weights.
  bf16            Convert to GGUF with bf16 weights.
  q8_0            Convert to GGUF with q8_0 weights (native converter type).
  auto            Convert to GGUF with auto-detected type.
  Q8_0            Quantize a base GGUF to Q8_0 (via llama-quantize binary).
  Q6_K            Quantize to Q6_K.
  Q5_K_M          Quantize to Q5_K_M.
  Q4_K_M          Quantize to Q4_K_M.
  ... (any type accepted by llama-quantize)

Examples:
  # Save HF copy and create an f16 GGUF:
  python quantize.py --merged-dir ./gemma-hermes-merged --formats hf f16

  # Create f16 GGUF and several quantizations:
  python quantize.py --merged-dir ./gemma-hermes-merged \\
      --output-dir ./quants --formats f16 Q8_0 Q5_K_M Q4_K_M

  # Only create quantized GGUFs (f16 base is created temporarily):
  python quantize.py --merged-dir ./gemma-hermes-merged \\
      --output-dir ./quants --formats Q8_0 Q5_K_M
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


# Formats handled by the llama.cpp HF→GGUF converter (--outtype argument).
GGUF_CONVERTER_TYPES = {"f16", "f32", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}


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
        + ", ".join(candidates)
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
        "Could not find llama.cpp quantize binary. Build llama.cpp first, e.g.: "
        "cmake -S . -B build && cmake --build build -j. "
        "Looked for: " + ", ".join(candidates)
    )


def run_command(cmd, label):
    print(f"\n==> {label}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def save_hf(merged_dir, out_dir):
    print(f"\n==> Copying HuggingFace model to {out_dir}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(merged_dir, out_dir)
    print(f"    Saved: {out_dir}")


def convert_to_gguf(converter_script, merged_dir, gguf_out_path, outtype):
    cmd = [
        sys.executable,
        converter_script,
        merged_dir,
        "--outfile",
        gguf_out_path,
        "--outtype",
        outtype,
    ]
    run_command(cmd, f"Converting HF model to GGUF ({outtype})")
    print(f"    Saved: {gguf_out_path}")


def quantize_gguf(quantize_bin, gguf_in_path, gguf_out_path, quant_type):
    cmd = [quantize_bin, gguf_in_path, gguf_out_path, quant_type]
    run_command(cmd, f"Quantizing GGUF → {quant_type}")
    print(f"    Saved: {gguf_out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a merged HuggingFace model to one or more output formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--merged-dir",
        required=True,
        help="Path to the merged HuggingFace model directory (output of merge.py).",
    )
    parser.add_argument(
        "--output-dir",
        default="./quantized",
        help="Directory where output files/folders will be written. (default: ./quantized)",
    )
    parser.add_argument(
        "--name",
        default="",
        help=(
            "Base name for output files. Defaults to the basename of --merged-dir. "
            "e.g. 'gemma-hermes' produces gemma-hermes-f16.gguf, gemma-hermes-Q8_0.gguf, etc."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        required=True,
        metavar="FORMAT",
        help=(
            "One or more output formats. HF copy: 'hf'. "
            "GGUF base types (via converter): f16, f32, bf16, q8_0, auto. "
            "GGUF quantizations (via llama-quantize): Q8_0, Q6_K, Q5_K_M, Q4_K_M, etc."
        ),
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default="/home/jamie/llama.cpp",
        help="Path to local llama.cpp checkout. Required for any GGUF output. (default: /home/jamie/llama.cpp)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    require_path(args.merged_dir, "Merged model directory")

    model_name = args.name or os.path.basename(args.merged_dir.rstrip("/").rstrip("\\"))
    os.makedirs(args.output_dir, exist_ok=True)

    formats = args.formats

    # Partition formats into categories.
    wants_hf = "hf" in formats
    gguf_base_formats = [f for f in formats if f.lower() in GGUF_CONVERTER_TYPES]
    quant_formats = [
        f for f in formats if f != "hf" and f.lower() not in GGUF_CONVERTER_TYPES
    ]

    needs_llama_cpp = bool(gguf_base_formats) or bool(quant_formats)

    converter_script = None
    quantize_bin = None

    if needs_llama_cpp:
        require_path(args.llama_cpp_dir, "llama.cpp directory")
        converter_script = find_converter(args.llama_cpp_dir)
    if quant_formats:
        quantize_bin = find_quantize_binary(args.llama_cpp_dir)

    # --- HF copy ---
    if wants_hf:
        hf_out = os.path.join(args.output_dir, f"{model_name}-hf")
        save_hf(args.merged_dir, hf_out)

    # --- GGUF base conversions ---
    # Track one base GGUF path to use as the source for downstream quantizations.
    gguf_for_quant = None

    for base_type in gguf_base_formats:
        gguf_out = os.path.join(args.output_dir, f"{model_name}-{base_type}.gguf")
        convert_to_gguf(converter_script, args.merged_dir, gguf_out, base_type)
        if gguf_for_quant is None:
            gguf_for_quant = gguf_out

    # --- Temporary base GGUF for quantization (if none was explicitly requested) ---
    temp_gguf_path = None
    if quant_formats and gguf_for_quant is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.close()
        temp_gguf_path = tmp.name
        print(f"\n(Creating temporary f16 GGUF for quantization: {temp_gguf_path})")
        convert_to_gguf(converter_script, args.merged_dir, temp_gguf_path, "f16")
        gguf_for_quant = temp_gguf_path

    # --- GGUF quantizations ---
    for quant_type in quant_formats:
        quant_out = os.path.join(args.output_dir, f"{model_name}-{quant_type}.gguf")
        quantize_gguf(quantize_bin, gguf_for_quant, quant_out, quant_type)

    # --- Cleanup temporary GGUF ---
    if temp_gguf_path and os.path.exists(temp_gguf_path):
        os.remove(temp_gguf_path)
        print(f"\n(Removed temporary GGUF: {temp_gguf_path})")

    print("\nDone.")


if __name__ == "__main__":
    main()
