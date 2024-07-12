# main.py
import argparse
from llama_sae_generation import load_model, LlamaSaeGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_dir", type=str, required=True)
    parser.add_argument("--sae_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_batch_size", type=int, default=1)  # Reduced batch size for MPS
    parser.add_argument("--sae_layers", type=str, default="")
    parser.add_argument("--device", type=str, default=None, choices=['cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    sae_layers = [int(layer) for layer in args.sae_layers.split(",")] if args.sae_layers else None

    model, tokenizer = load_model(
        args.llama_dir,
        args.sae_dir,
        args.max_seq_len,
        args.max_batch_size,
        sae_layers
	args.device
    )

    generator = LlamaSaeGenerator(model, tokenizer)

    # Example usage
    prompts = [
        "Once upon a time",
        "The meaning of life is",
    ]

    for prompt in prompts:
        completion = generator.text_completion([prompt])[0]
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
        print()

    # Example chat completion
    dialogs = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the capital of France?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ],
    ]

    for dialog in dialogs:
        completion = generator.chat_completion([dialog])[0]
        print(f"Dialog: {dialog}")
        print(f"Assistant: {completion}")
        print()

if __name__ == "__main__":
    main()
