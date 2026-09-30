# Local models (Ollama)

Ollama runs AI models on your own Mac for free. We use it only for **development**: checking that the pipeline runs end to end without paying for API calls. Final results use the pinned cloud models in the architecture doc, because small local models are too weak to be credible reviewers.

## Status

- Python connector installed (`langchain-ollama`, via `pip install -e ".[local]"`).
- Check script ready: `scripts/check_ollama.py`.
- **Ollama itself is not installed yet: the Mac's disk is full.** On 2026-09-30 there was about 0.5 GB free; Ollama plus one small model needs about 3–4 GB, and macOS needs headroom to stay stable. Free **at least 10 GB** first.

## What fits on this Mac

8 GB of memory (Apple M1). A model has to fit in memory alongside macOS, so use a model of about **4 billion parameters or smaller** (roughly a 2.5 GB download). Larger models will be extremely slow or fail.

| Model | Download | Use |
|---|---|---|
| `qwen3:4b` (default in the check script) | ~2.5 GB | pipeline testing |

## Setup (after freeing disk space)

```bash
brew install ollama
brew services start ollama        # runs in the background
ollama pull qwen3:4b
python scripts/check_ollama.py    # should print the model's answer
```

The check script explains exactly what is missing at each stage (not installed / not running / model not downloaded).

## Free alternative that needs no disk space

The **Gemini API free tier** (Google AI Studio) gives a much stronger model at no cost, with rate limits. For development on this Mac it is the better option; add `GOOGLE_API_KEY` to `.env` when you have it.
