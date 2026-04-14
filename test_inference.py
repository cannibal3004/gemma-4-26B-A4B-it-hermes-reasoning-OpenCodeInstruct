import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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

# --- Configuration ---
BASE_MODEL_ID = "google/gemma-4-26B-A4B-it"
ADAPTER_DIR = "./gemma-hermes-reasoning-results-rocm/checkpoint-700" # Relative to where you run the script, or use absolute path

def test_inference(prompt, tools_context):
    print(f"\n{'='*50}")
    print(f"TEST PROMPT: {prompt}")
    print(f"{'='*50}\n")

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # 2. Load Base Model in BF16
    print(f"Loading base model ({BASE_MODEL_ID}) in BF16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 3. Convert custom layers, then load adapter
    print("Converting Gemma4ClippableLinear layers to standard Linear...")
    base_model = convert_custom_layers(base_model)

    print(f"Loading adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    # 4. Prepare Input using Gemma Template
    # We include the tools in the user turn so the model has the context it was trained on.
    full_prompt = f"<start_of_turn>user\n{tools_context}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # 5. Generate
    print("Generating response...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,  # Low temperature for structured output
            do_sample=True,
            top_p=0.9,
        )

    # 6. Decode and Print
    # We slice the output to only show the newly generated text (after the prompt)
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    
    print("--- MODEL RESPONSE ---")
    print(generated_text)
    print("\n--- END RESPONSE ---")

if __name__ == "__main__":
    # Example Tool Context (mimicking the dataset style)
    example_tools = """<tools>
[{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}]
</tools>"""

    # Test Case 1: Simple Tool Call
    test_prompt_1 = "What is the weather like in Paris?"
    
    # Test Case 2: Reasoning Trace (if the model was trained to think)
    test_prompt_2 = "I need to know if I should wear a coat in London. Check the weather for me."

    # Run tests
    # Note: Ensure ADAPTER_DIR points to the correct absolute path if running from elsewhere
    try:
        test_inference(test_prompt_1, example_tools)
        test_inference(test_prompt_2, example_tools)
    except Exception as e:
        print(f"Error during inference: {e}")
        print("\nTip: Make sure ADAPTER_DIR in the script matches the actual path where your adapter was saved.")
