# main.py
import argparse
from llama_sae_generation import load_model, LlamaSaeGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--sae_layers", type=str, default="")
    args = parser.parse_args()

    sae_layers = [int(layer) for layer in args.sae_layers.split(",")] if args.sae_layers else None

    model, tokenizer = load_model(
        args.ckpt_dir,
        args.tokenizer_path,
        args.max_seq_len,
        args.max_batch_size,
        sae_layers
    )

    generator = LlamaSaeGenerator(model, tokenizer)

    # Example usage
    prompts = [
        "Once upon a time",
        "The meaning of life is",
    ]

    completions = generator.text_completion(prompts)
    for prompt, completion in zip(prompts, completions):
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

    chat_completions = generator.chat_completion(dialogs)
    for dialog, completion in zip(dialogs, chat_completions):
        print(f"Dialog: {dialog}")
        print(f"Assistant: {completion}")
        print()

if __name__ == "__main__":
    main()
