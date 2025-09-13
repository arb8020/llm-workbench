#!/usr/bin/env python3
import argparse, os
from nnsight import LanguageModel, CONFIG

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai-community/gpt2")  # HF repo id
    p.add_argument("--prompt", default="The Eiffel Tower is in the city of")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--remote", action="store_true", help="run on NDIF if you have an API key")
    p.add_argument("--api-key", default=os.getenv("NDIF_API_KEY"), help="NDIF API key (or env NDIF_API_KEY)")
    args = p.parse_args()

    if args.remote:
        if not args.api_key:
            raise SystemExit("remote=True but no NDIF API key. Pass --api-key or set NDIF_API_KEY.")
        # Configure NNsight for NDIF
        CONFIG.API.APIKEY = args.api_key  # uses NNsight’s CONFIG to set the key

    # Create the language model wrapper (local weights if remote=False; otherwise NDIF runs it)
    model = LanguageModel(args.model, device_map="auto")

    # One forward/generation run
    with model.generate(args.prompt, max_new_tokens=args.max_new_tokens, remote=args.remote) as run:
        out = model.generator.output.save()  # token IDs (prompt + generated)

    # Decode: split original prompt vs. newly generated continuation
    n = args.max_new_tokens
    decoded_prompt = model.tokenizer.decode(out[0][:-n].cpu(), skip_special_tokens=True)
    decoded_answer = model.tokenizer.decode(out[0][-n:].cpu(), skip_special_tokens=True)

    print("Prompt:", decoded_prompt)
    print("Generated:", decoded_answer)

if __name__ == "__main__":
    main()

