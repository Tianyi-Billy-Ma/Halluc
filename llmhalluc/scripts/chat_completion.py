import argparse
import torch
from llmhalluc.models import get_model, get_tokenizer
from llmhalluc.extras.constant import SPECIAL_TOKEN_MAPPING


def detect_special_token(model_path: str):
    """Detect special backtrack token based on model path/name."""
    # Simple detection logic based on mapping keys in constant.py
    for key, mapping in SPECIAL_TOKEN_MAPPING.items():
        if key in model_path.lower():
            # Get the first token defined in the mapping for this model type
            return list(mapping.keys())[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Run a simple chat completion task.")
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Path to the model."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Path to the tokenizer. Defaults to model path.",
    )
    parser.add_argument(
        "--message", "-m", type=str, required=True, help="Input message/prompt."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Max new tokens to generate."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="Whether to use sampling."
    )
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer_path = (
        args.tokenizer_name_or_path
        if args.tokenizer_name_or_path
        else args.model_name_or_path
    )

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = get_tokenizer(tokenizer_path)

    print(f"Loading model from {args.model_name_or_path}...")

    # Detect special token
    backtrack_token = detect_special_token(args.model_name_or_path)
    if backtrack_token:
        print(f"Detected special backtrack token: {backtrack_token}")
        # Find its ID
        backtrack_id = tokenizer.convert_tokens_to_ids(backtrack_token)
        print(f"Backtrack Token ID: {backtrack_id}")
    else:
        print("No known special backtrack token detected for this model type.")

    # Use device_map="auto" if CUDA is available for efficient loading, otherwise fallback
    kwargs = {}
    if device == "cuda":
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = "auto"
    else:
        kwargs["device_map"] = "cpu"

    model = get_model(args.model_name_or_path, tokenizer=tokenizer, **kwargs)

    # Input
    prompt_text = args.message

    # Apply chat template
    messages = [{"role": "user", "content": prompt_text}]
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"Warning: Could not apply chat template ({e}). Using raw prompt.")
        formatted_prompt = prompt_text

    print(f"\n[Input]")
    print(f"Raw Message: {prompt_text}")
    print(f"Formatted Prompt:\n{formatted_prompt}")

    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids
    if device == "cuda":
        input_ids = input_ids.to(model.device)

    print(f"Token IDs: {input_ids.tolist()[0]}")

    # Generate
    print("\n[Generating...]")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Extract new tokens
    new_tokens = output_ids[0][input_ids.shape[1] :]
    full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    print(f"\n[Output]")
    print(f"Output Token IDs (Full): {output_ids.tolist()[0]}")
    print(f"Output Token IDs (New Only): {new_tokens.tolist()}")
    print("-" * 40)
    print(f"Generated Text:\n{generated_text}")
    print("-" * 40)
    print(f"Full Text:\n{full_output_text}")
    print("-" * 40)


if __name__ == "__main__":
    main()
