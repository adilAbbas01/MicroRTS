# 2026 IEEE WCCI MicroRTS LLM Game AI Competition

This competition is part of [2026 IEEE WCCI](https://attend.ieee.org/wcci-2026/competitions/).

---

## Overview

Build an LLM-powered agent that plays MicroRTS by understanding game state descriptions and generating strategic actions through prompts.

**What makes this different from traditional game AI?**
- No training or fine-tuning - the LLM uses only the prompt and game state
- Success depends on prompt engineering and strategy design
- Tests LLM reasoning, planning, and instruction-following capabilities

---

## Getting Started

### Prerequisites

- Java JDK 17+
- [Ollama](https://ollama.ai/) installed locally
- GPU recommended (CPU inference is too slow for real-time play)

### Step 1: Clone and Setup

```bash
git clone https://github.com/drchangliu/MicroRTS
cd MicroRTS
```

### Step 2: Start Ollama with a Model

```bash
# Download and run the model (leave this terminal open)
ollama run llama3.1:8b
```

### Step 3: Compile the Game

```bash
find src -name '*.java' > sources.list
javac -cp "lib/*:bin" -d bin @sources.list
```

### Step 4: Run a Test Game

```bash
chmod +x RunLoop.sh
./RunLoop.sh
```

---

## How to Compete

### The Main Task

Modify the prompt in `src/ai/abstraction/ollama.java` to make your LLM agent win against opponents.

**Key file locations:**

| File | Purpose |
|------|---------|
| `src/ai/abstraction/ollama.java` | LLM agent - **modify the PROMPT here** |
| `resources/config.properties` | Game settings and opponent selection |
| `RunLoop.sh` | Run multiple games for testing |

### Model Selection Policy

Submitters are welcome to **suggest a preferred model ID** in their agent code or in a comment/README. However, competition runs are executed on the organizers' server using locally available Ollama models. If your preferred model is not available on the server, your agent will be run with whatever model is available.

**Current server model:** `llama3.1:8b` (Meta Llama 3.1 8B, ~5 GB — runs well on CPU and Apple Silicon).

To specify your preferred model, set the default in your agent class:

```java
static String MODEL = System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
```

The `OLLAMA_MODEL` environment variable always takes precedence, so the organizers can override your default when running on the server. If your preferred model differs from what's available, note it in your submission README so organizers are aware.

### Changing the Opponent

In `resources/config.properties`:
```properties
AI2=ai.RandomBiasedAI
```

Available opponents:
- `ai.RandomBiasedAI` - Random but prefers useful actions
- `ai.RandomAI` - Purely random actions
- `ai.PassiveAI` - Does nothing
- `ai.abstraction.HeavyRush` - Builds heavy units aggressively
- `ai.abstraction.LightRush` - Builds light units aggressively

### Changing the Map

In `resources/config.properties`:
```properties
map_location=maps/8x8/basesWorkers8x8.xml
```

Maps are in the `maps/` folder organized by size (4x4, 8x8, 10x10, 12x12, 16x16, 24x24).

### Running Multiple Games

Edit `RunLoop.sh`:
```bash
TOTAL_RUNS=10                    # Number of games to run
RUN_TIME_PER_GAME_SEC=350        # Seconds per game before timeout
```

Then run:
```bash
./RunLoop.sh
```

---

## Understanding the LLM Agent

### How It Works

1. Each game tick, the agent receives the current game state as text
2. The static prompt (rules + strategy) is combined with the dynamic state
3. The LLM generates a JSON response with actions
4. The game engine validates and executes the actions

### Prompt Structure

See [LLM_PROMPTS.md](LLM_PROMPTS.md) for the complete prompt format.

**Static prompt:** Game rules, unit stats, action format, suggested strategy

**Dynamic prompt:** Current map state, unit positions, resources, turn number

**Response format:**
```json
{
  "thinking": "My strategic reasoning...",
  "moves": [
    {
      "raw_move": "(1, 1): worker harvest((0, 0), (2, 1))",
      "unit_position": [1, 1],
      "unit_type": "worker",
      "action_type": "harvest"
    }
  ]
}
```

---

## Checking Results

### During the Game

Watch the console output for LLM responses and game state.

### After the Game

Open the generated CSV file:
```
Response<TIMESTAMP>_<AI1>_<AI2>_<MODEL>.csv
```

The `Score_in_every_run` column shows:
- If P0 score > P1 score: Player 0 wins
- If P1 score > P0 score: Player 1 wins

(AI1 is Player 0, AI2 is Player 1)

---

## Running on HPC Clusters

If using SLURM-managed clusters:

### Interactive Session
```bash
srun --gres=gpu:1 --pty bash
ollama serve &
./RunLoop.sh
```

### Batch Job
```bash
sbatch run_loop.slurm
```

Monitor with:
```bash
squeue -u $USER      # Check job status
ls -lrt slurm-*.out  # View logs
scancel <JOBID>      # Cancel job
```

See [GPU_SETUP.md](GPU_SETUP.md) for detailed HPC instructions.

---

## Tips for Competition

1. **Start simple** - Get a working agent before optimizing
2. **Test against PassiveAI first** - Ensure basic functionality works
3. **Log everything** - Check the `logs/` folder for detailed game traces
4. **Iterate on prompts** - Small changes can have big effects
5. **Consider the response format** - Invalid JSON means no actions

---

## Self-Reported Tournament Results (Optional)

Submitters are encouraged to include self-reported benchmark results with their submission. These let you share performance numbers obtained using your preferred model on your own hardware — which may differ from what is available on the official competition server.

### What to include

Add a `results.json` file alongside your agent code (e.g., `src/ai/abstraction/submissions/your_team/results.json`). Run `benchmark_arena.py` against the six built-in leaderboard opponents (not against other submissions) and fill in the results.

The format mirrors the official `benchmark_results/leaderboard.json` schema with a few extra metadata fields:

```json
{
  "self_reported": true,
  "submitter": "your_team_name",
  "agent_class": "ai.abstraction.submissions.your_team.YourAgent",
  "model": "qwen3:14b",
  "hardware": "NVIDIA RTX 4090, 64 GB RAM",
  "date": "2026-03-01",
  "notes": "Optional: anything relevant about your setup.",

  "version": "2.0",
  "format": "single-elimination",
  "map": "maps/8x8/basesWorkers8x8.xml",
  "max_cycles": 5000,
  "games_per_matchup": 1,

  "score": 96.0,
  "grade": "A+",
  "eliminated_at": "CoacAI",

  "opponents": {
    "RandomBiasedAI": { "wins": 1, "draws": 0, "losses": 0, "avg_game_score": 1.2, "weighted_points": 12.0 },
    "HeavyRush":      { "wins": 1, "draws": 0, "losses": 0, "avg_game_score": 1.2, "weighted_points": 24.0 },
    "LightRush":      { "wins": 1, "draws": 0, "losses": 0, "avg_game_score": 1.2, "weighted_points": 18.0 },
    "WorkerRush":     { "wins": 1, "draws": 0, "losses": 0, "avg_game_score": 1.0, "weighted_points": 15.0 },
    "Tiamat":         { "wins": 1, "draws": 0, "losses": 0, "avg_game_score": 1.2, "weighted_points": 24.0 },
    "CoacAI":         { "wins": 0, "draws": 0, "losses": 1, "avg_game_score": 0.0, "weighted_points": 0.0  }
  }
}
```

See `src/ai/abstraction/submissions/example_team/results.json` for a filled-in example.

### How to generate your results

```bash
# Run the benchmark arena against the six built-in opponents
export OLLAMA_MODEL="your-preferred-model"
python3 benchmark_arena.py

# Results are written to benchmark_results/benchmark_<timestamp>.json
# Copy the relevant entry into your submission's results.json
```

### Important caveats

- Self-reported results are **not verified** and are separate from official competition scores.
- Official scores are always run by the organizers on the competition server using the server's available model (`llama3.1:8b` currently).
- Self-reported results are useful context — they let reviewers see how your agent performs with a more capable model, and help identify whether a weaker official score is a model-capability issue vs. a prompt/strategy issue.
- Only report results against the six built-in leaderboard opponents. Do not report results against other submissions.

---

## Submission

Competition submission details will be announced on the [IEEE WCCI 2026 website](https://attend.ieee.org/wcci-2026/competitions/).

---

## Questions?

- Check the [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for codebase navigation
- Review the [LLM_PROMPTS.md](LLM_PROMPTS.md) for prompt format details
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
