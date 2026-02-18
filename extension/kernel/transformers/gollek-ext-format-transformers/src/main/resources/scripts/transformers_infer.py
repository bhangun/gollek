#!/usr/bin/env python3
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    # Keep stdout clean for Java side parsing.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    try:
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            AutoModelForCausalLM,
            pipeline,
        )
    except Exception as e:
        print(f"Failed to import transformers: {e}", file=sys.stderr)
        return 2

    model_path = args.model_path
    if not os.path.isdir(model_path):
        print(f"Model directory not found: {model_path}", file=sys.stderr)
        return 3

    try:
        cfg = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if getattr(cfg, "is_encoder_decoder", False):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            task = "text2text-generation"
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            task = "text-generation"

        generator = pipeline(task=task, model=model, tokenizer=tokenizer, device=-1)

        out = generator(
            args.prompt,
            max_new_tokens=max(1, args.max_new_tokens),
            temperature=max(0.0, args.temperature),
            top_p=min(1.0, max(0.0, args.top_p)),
            do_sample=args.temperature > 0,
            return_full_text=False,
        )

        if not out:
            print("")
            return 0

        first = out[0]
        text = first.get("generated_text") or first.get("summary_text") or ""
        print(text)
        return 0

    except Exception as e:
        print(f"Transformers inference error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
