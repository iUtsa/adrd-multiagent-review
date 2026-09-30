"""Check that a local Ollama model works end to end through LangChain.

    python scripts/check_ollama.py                 # uses the default model below
    python scripts/check_ollama.py --model qwen3:4b

Exit code 0 = the model answered. See docs/LOCAL_MODELS.md for setup.
"""

from __future__ import annotations

import argparse
import shutil
import sys

import requests

DEFAULT_MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    if not shutil.which("ollama"):
        print("Ollama is not installed. Install it with: brew install ollama")
        return 1
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except requests.RequestException:
        print("Ollama is installed but not running. Start it with: brew services start ollama")
        return 1
    names = [m["name"] for m in tags.get("models", [])]
    if args.model not in names:
        print(f"Model {args.model!r} is not downloaded. Get it with: ollama pull {args.model}")
        print(f"Downloaded models: {names or 'none'}")
        return 1

    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=args.model, temperature=0)
    reply = llm.invoke("Answer with one word: what does MMSE stand for? Just the first word.")
    print(f"{args.model} answered: {reply.content.strip()[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
