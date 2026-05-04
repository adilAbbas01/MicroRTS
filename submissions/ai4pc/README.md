# AI4PC

Hybrid LLM agent for the 2026 IEEE WCCI MicroRTS LLM Game AI Competition.

## Strategy

Every 100 game ticks, the LLM picks one of four rush doctrines based on the visible enemy composition:

- **WORKER_RUSH** — early game (tick < 200) when the enemy has no military units yet
- **LIGHT_RUSH** — mid-game default, or counter to enemy ranged
- **HEAVY_RUSH** — counter to enemy light
- **RANGED_RUSH** — counter to enemy heavy

A stickiness rule prevents thrashing: the agent only switches strategy when the enemy reveals a new unit type. The chosen strategy's scripted bot (WorkerRushPlusPlus, LightRush, HeavyRush, RangedRush) executes the per-tick actions.

## Model and runtime

- **Provider:** Ollama (local HTTP at `localhost:11434`)
- **Default model:** `llama3.1:8b` (set via `OLLAMA_MODEL` env var)
- **Determinism:** `temperature=0`, `seed=42`
- **LLM call interval:** 100 ticks (≈3 ms LLM overhead amortized per tick on the test hardware)

## Self-reported result

79.0 / **Grade B**, eliminated at Tiamat. See `results.json` for the full opponent breakdown.
