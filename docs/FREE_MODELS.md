# Running on free model tiers

Decided 2026-10-11 (Utsa): run the professor's 1,855 search results on **free** model APIs only, accepting that it is slow. Config: [`configs/runs/PROF_search_free.yaml`](../configs/runs/PROF_search_free.yaml).

```bash
python -m adrd_review.preflight configs/runs/PROF_search_free.yaml     # keys, live test calls, days needed
scripts/run_until_done.sh configs/runs/PROF_search_free.yaml           # unattended: re-runs every 3 h until done
tail -f data/logs/until_done_PROF_search_free.log                      # progress (flow counts after each try)
python -m adrd_review.eval.flow data/runs/PROF_search_free/<latest>    # stage counts so far
```

## 1. What is free (checked 2026-10-11)

| Source | Free limits | Status here | Used for |
|---|---|---|---|
| **Groq** | Per model: 30 requests/min, 1,000 requests/day, **8,000 tokens/min, 200,000 tokens/day** | Key in `.env` ✓ | Titles and abstracts (3 models) |
| **NVIDIA NIM** (build.nvidia.com) | ~40 requests/min per account; about 1,000 free credits (5,000 on request) | Needs `NVIDIA_API_KEY` | Full text, reviewer A |
| **OpenRouter** `:free` models | 20 requests/min; **50 requests/day per account**, 1,000/day after a one-time $10 top-up | Needs `OPENROUTER_API_KEY` | Full text, reviewers B and C |
| Google AI Studio (personal Gmail) | Gemini Flash models free; limits only shown in the console; free-tier prompts may be used by Google | Not set up | Strongest free option if wanted |
| Mistral (Experiment plan) | ~$10/month of credit; phone verification | Not set up | Possible extra |
| GitHub Models | **Not available**: the endpoint now returns only "OK" (tested 2026-10-11) | – | – |
| Cloudflare Workers AI, Cohere, SambaNova | 10,000 "neurons"/day; 1,000 calls/month; 20 requests/day | Not worth it | – |
| Cerebras | Now a $5 trial with a card, not a free tier | – | – |

Sources: [OpenRouter's comparison](https://openrouter.ai/blog/tutorials/free-llm-apis-compared/), [October 2026 roundup](https://dev.to/tariqnasser/free-llm-api-tiers-in-october-2026-whats-left-and-how-i-chain-them-227l), [daily-checked limits](https://gravity.fast/data/free-llm-api-tiers/), [NVIDIA forum on credits](https://forums.developer.nvidia.com/t/credit-rate-limit-increase-request-1-000-5-000-credits-40-200-rpm/379108), [NVIDIA's account-wide cap](https://news.creeta.com/en/nvidia-nim-free-tier-limits-2026/). Free tiers change often; the preflight check calls each model before a run.

**What about MCP?** MCP (Model Context Protocol) connects a model to tools and data (files, search, databases). It does **not** run models, so it cannot replace a model API. No MCP server gives free model calls for a pipeline like this one.

## 2. Getting the free keys (no card needed)

Add each to `.env` (never committed). A run already in progress picks them up on its next try.

| Key | Where | `.env` line |
|---|---|---|
| OpenRouter | openrouter.ai → sign in → Keys → Create key | `OPENROUTER_API_KEY=sk-or-...` |
| NVIDIA | build.nvidia.com → sign in (NVIDIA Developer Program, free) → any model → "Get API Key" | `NVIDIA_API_KEY=nvapi-...` |
| Google AI Studio (optional) | aistudio.google.com/apikey with a personal Gmail | `GOOGLE_API_KEY=...` |

## 3. How the run fits in free limits

The limits are on **calls and tokens per day**, and every call repeats the same instructions and answer form (~2,500 tokens). So:

1. **Batch screening** ([agents/batch-screening.md](agents/batch-screening.md)): 50 titles per call, 4 abstracts per call. The decision rules are unchanged; each article is still decided on its own text. Titles: 1,855 × 3 reviewers = 5,565 calls become **114**.
2. **Each reviewer keeps its models for the whole run**, one per stage (`title_model`, `model`, `eligibility_model`), so the panel is consistent:

   | Reviewer | Titles | Abstracts | Full papers |
   |---|---|---|---|
   | A, clinical | Groq gpt-oss-120b (low reasoning) | Groq gpt-oss-120b (medium reasoning) | NVIDIA Nemotron 3 Super 120B |
   | B, machine learning | Groq Qwen 3.8 27B | Groq Qwen 3.8 27B | OpenRouter Gemma 4 31B (free) |
   | C, methods | Groq gpt-oss-20b (low) | Groq gpt-oss-20b (medium) | OpenRouter Inkling (free) |

   Groq cannot read a full paper: one full-text call (~13,000 tokens) is bigger than its 8,000 tokens/min limit.
3. **Calls saved:** no triage (it only mattered for full-text excerpts), no adjudicator (split votes go to the majority and are flagged), verifier checks quotes without a model call, no extraction or annotation (counts only).
4. **Unattended multi-day runs:** a model whose daily limit is used up is marked and skipped at once (`LimitReached`); its articles are *deferred*, not failed. A model whose key is not set yet makes only the articles that need it wait. `scripts/run_until_done.sh` re-runs every 3 hours; finished calls come from the cache, so each try only does the open work. Abstract batches are formed only after every title is decided, so their cache keys stay the same from try to try.

## 4. How long

| Stage | Free budget | Time |
|---|---|---|
| Titles | ~37 batch calls per Groq model (~170,000 tokens) | **About 1 day** |
| Abstracts | ~1,500 abstracts ≈ 380 batch calls per Groq model (~2.5–2.9 million tokens) at 200,000 tokens/day | **About 2 weeks** (the bottleneck) |
| Full text | ~400 papers × 3 reviewers; NVIDIA ~1,000 credits; OpenRouter 50/day (1,000/day after $10) | Days to weeks, depending on the keys |

**Ways to make it faster, if wanted:**
- A **one-time $10 top-up on OpenRouter** raises its free models to 1,000 calls/day. That alone makes the full-text stage take a day or two.
- A free **Google AI Studio key**: Gemini Flash could take over the abstract stage. Changing a stage's models means starting that stage again, so decide before the abstract stage is far along.

## 5. Quality checks made before choosing these models (2026-10-11)

- **Titles, one real batch of 50**: Qwen excluded 3 (a Parkinson's study and two reviews, all correct). Both gpt-oss models excluded about half the titles, mostly because a title did not *mention* dementia or AI. A title is excluded only if **all three** reviewers agree, so the panel excluded just those 3. The batch instructions now say plainly that not mentioning something is never a reason.
- **Abstracts, one batch of 5**: all three models agreed (Parkinson's study and an MCI-only study excluded, as the professor's rules require). gpt-oss with "low" reasoning skipped judgments on one article, so abstracts use "medium" reasoning.
