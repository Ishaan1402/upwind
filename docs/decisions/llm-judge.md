# LLM judge provider and model selection

- Status: Proposed (not yet implemented)
- Date: 2026-08-16
- Scope: `backend/llm_judge.py`, `backend/llm.py`, `.github/workflows/eval.yml`, and the eval corpus tooling.

## Problem

The judge process has three concrete failure modes:

1. **Groq rate limits.** `llama-3.3-70b-versatile` (`DEFAULT_JUDGE_MODEL`) hits
   HTTP 429 and falls back to `llama-3.1-8b-instant`, which is too weak and
   produces false positives (it has flagged "AQI" as leaked jargon). Research:
   8B models can handle verifiable facts but are position-biased and unreliable
   on subjective judgment.
2. **Weak structured output.** `response_format={"type":"json_object"}` is
   weaker than schema-enforced structured output (`responseSchema`) — Groq's
   weak point.
3. **DeepSeek price change** (effective 16:00 UTC 2026-08-16): peak/off-peak
   billing. Peak = 01:00–04:00 UTC and 06:00–10:00 UTC; off-peak is half price.
   `deepseek-v4-flash` off-peak $0.22/$0.66 per 1M in/out tokens, peak
   $0.44/$1.32.

## Decision

Adopt a forward-looking judge design: change the provider/model now, with a
roadmap for later improvements.

### Current state

- **Judge** (`backend/llm_judge.py`): Groq `openai/gpt-oss-120b` with
  fallback `openai/gpt-oss-20b` (Llama 3.3 70B / 3.1 8B were decommissioned
  2026-08-16). `response_format={"type":"json_object"}`, temperature 0.
  Emits JSON: `grounded`, `hallucinations[]`, `leaked_jargon[]`,
  `has_disallowed_headers`, `has_actionable_tip`, `verdict`, `reasoning`.
- **Briefer** (`backend/llm.py`): DeepSeek `deepseek-v4-flash`, max_tokens 600,
  temperature 0.4.
- **Judge runs**: (1) background on every `/api/why` query
  (`backend/routers/why.py`), (2) nightly corpus eval
  (`.github/workflows/eval.yml`, scheduled 11:00 UTC).

### Shortlist

| Model | Provider | $/1M in+out | Rate-limit note | Free tier | Judge fit (1-5) |
|---|---|---|---|---|---|
| Gemini 3.1 Flash-Lite | Google | $0.25 / $1.50 (batch ~$0.125/$0.75) | Free ~15 RPM / ~250K TPM / ~500-1.5K RPD; Tier 1 paid 10M TPM | Yes (free tokens) | 4 |
| deepseek-v4-flash | DeepSeek | $0.14/$0.28 (cache-hit $0.003); off-peak $0.22/$0.66, peak $0.44/$1.32 | Concurrency 2,500 (highest) | No | 4 |
| gpt-4.1-mini | OpenAI | $0.40/$1.60 | Tier 1 500 RPM / 200K TPM | No | 4 |
| gpt-oss-120b | Groq / Fireworks / Cerebras | Groq $0.15/$0.60; Fireworks $0.10/$0.60; Cerebras $0.35/$0.75 | Groq free 30 RPM but 1K RPD/8K TPM | tiny | 4 (Groq) / 3-4 (Cerebras) |

### Recommended change

- **Judge → Gemini 3.1 Flash-Lite**: free tokens, `responseSchema` structured
  output, high throughput, 429-resistant. Fallback chain: DeepSeek v4-flash
  (2,500 concurrency, effectively 429-proof) → Groq `gpt-oss-120b` (stronger
  120B judge than llama-3.1-8b). Keep the 8B only as last resort.
- **Briefer → keep `deepseek-v4-flash`**; schedule eval off-peak (done: nightly
  moved to 11:00 UTC).
- **Practical floor**: no 70B needed for rubric-scored fact-checking; the
  Flash-Lite / 4.1-mini / v4-flash class is the sweet spot. Use temp 0 + schema
  + few-shot exemplars; consider 2 cheap judges with majority vote (single-trial
  judging flips ~14% of verdicts).

## Future work

- Provider-agnostic judge abstraction (interface + config-driven model selection).
- Schema-enforced structured output (`responseSchema`) everywhere.
- Multi-judge majority voting + calibration of pass/fail thresholds.
- Judge-behavior observability (verdict drift, flake rate, rate-limit metrics)
  feeding the observability dashboard.

## Sources

Groq rate limits https://console.groq.com/docs/rate-limits · Groq pricing https://groq.com/pricing · Gemini pricing https://ai.google.dev/gemini-api/docs/pricing · Gemini rate limits https://ai.google.dev/gemini-api/docs/rate-limits · OpenAI pricing https://developers.openai.com/api/docs/pricing · DeepSeek pricing https://api-docs.deepseek.com/quick_start/pricing · Cerebras https://inference-docs.cerebras.ai/support/rate-limits · Fireworks https://fireworks.ai/pricing · Together https://www.together.ai/pricing · Mistral https://docs.mistral.ai/inference/pricing · OpenRouter https://openrouter.ai/pricing
