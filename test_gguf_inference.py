import argparse
import json
import os
import re
import subprocess
import sys


def require_file(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def build_prompt(user_prompt, tools_context):
    return (
        f"<start_of_turn>user\n{tools_context}\n\n{user_prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def run_llama_cli(llama_cli, model_path, prompt, ctx_size, n_predict, temperature, threads, gpu_layers):
    cmd = [
        llama_cli,
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-c",
        str(ctx_size),
        "--temp",
        str(temperature),
        "-t",
        str(threads),
        "-ngl",
        str(gpu_layers),
        "--no-display-prompt",
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def evaluate_output(text, required_tags):
    missing = [tag for tag in required_tags if tag not in text]
    return missing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a quick GGUF sanity check with llama.cpp llama-cli."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to GGUF model file.",
    )
    parser.add_argument(
        "--llama-cli",
        default="/home/jamie/llama.cpp/build/bin/llama-cli",
        help="Path to llama.cpp llama-cli binary.",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=8192,
        help="llama.cpp context size.",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=384,
        help="Maximum generated tokens per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, os.cpu_count() // 2 if os.cpu_count() else 8),
        help="CPU threads for llama-cli.",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=999,
        help="Number of layers to offload to GPU (-ngl).",
    )
    parser.add_argument(
        "--require-tags",
        default="<think>,<tool_call>",
        help="Comma-separated tags expected in output.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write structured sanity-check results as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    require_file(args.model, "GGUF model")
    require_file(args.llama_cli, "llama-cli binary")

    required_tags = [t.strip() for t in args.require_tags.split(",") if t.strip()]

    tools_context = (
        "<tools>\n"
        "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", "
        "\"description\": \"Get current weather\", \"parameters\": {\"type\": \"object\", "
        "\"properties\": {\"location\": {\"type\": \"string\"}}}}}]\n"
        "</tools>"
    )

    test_prompts = [
        "What is the weather in Paris right now?",
        "I need to know if I should wear a coat in London. Check the weather for me.",
    ]

    all_passed = True
    report = {
        "model": args.model,
        "llama_cli": args.llama_cli,
        "ctx_size": args.ctx_size,
        "n_predict": args.n_predict,
        "temperature": args.temperature,
        "threads": args.threads,
        "gpu_layers": args.gpu_layers,
        "required_tags": required_tags,
        "results": [],
    }
    for i, p in enumerate(test_prompts, start=1):
        full_prompt = build_prompt(p, tools_context)
        print(f"\n=== Prompt {i} ===")
        print(p)

        output = run_llama_cli(
            llama_cli=args.llama_cli,
            model_path=args.model,
            prompt=full_prompt,
            ctx_size=args.ctx_size,
            n_predict=args.n_predict,
            temperature=args.temperature,
            threads=args.threads,
            gpu_layers=args.gpu_layers,
        )

        # Show concise sample output and tag status.
        preview = re.sub(r"\s+", " ", output).strip()[:1000]
        print("Output preview:")
        print(preview)

        missing = evaluate_output(output, required_tags)
        if missing:
            all_passed = False
            print(f"Result: FAIL (missing tags: {missing})")
        else:
            print("Result: PASS")

        report["results"].append(
            {
                "prompt_index": i,
                "prompt": p,
                "passed": len(missing) == 0,
                "missing_tags": missing,
                "output_preview": preview,
                "output": output,
            }
        )

    report["all_passed"] = all_passed

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report: {args.json_out}")

    if not all_passed:
        print("\nSanity check finished with failures.")
        sys.exit(2)

    print("\nSanity check passed.")


if __name__ == "__main__":
    main()
