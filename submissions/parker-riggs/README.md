# Parker-Riggs AdaptiveRushBot

`AdaptiveRushBot` is a hybrid MicroRTS agent that tries to counter whatever
the opponent is building. It picks between four scripted rush strategies each
game and updates that choice as it gains more information about the enemy army
and the map size.

The four strategies and the reasoning behind each:

- **WorkerRush** - floods the map with worker units straight from the base,
  skipping the barracks ramp. Best for very small maps (4x4) and for emergency
  base defense when the enemy is already on our doorstep.
- **LightRush** - trains cheap, fast Light units from a Barracks. Light units
  destroy Workers very efficiently and close the gap on Ranged units.
- **HeavyRush** - trains Heavy units. Slower to build but win direct combat
  against Light units.
- **RangedRush** - trains Ranged units. Kite Heavy units from distance and are
  the strongest default opener on big maps (16x16 and larger) because range
  matters more when units have further to travel.

The counter triangle: Light beats Ranged, Heavy beats Light, Ranged beats Heavy.
WorkerRush sits outside the triangle as an emergency / tiny-map option.

### Decision making

Strategy selection works in two layers:

1. **LLM layer** - every 25 ticks the bot sends a compact game state summary to
   a local Ollama instance and asks it to pick one of the four strategies. The
   prompt includes the current tick, map dimensions, a map-size tier
   (tiny/small/medium/large), your own unit counts, and a per-type breakdown of
   visible enemy units including Ranged so the model can apply the counter
   triangle correctly.
2. **Heuristic fallback** - if Ollama is not running, times out, or returns
   something unparseable, a hand-written rule set takes over. It checks for
   emergency defense first, then commitment to an existing military type, then
   the visible enemy composition, then economy/time thresholds scaled to the
   map size.

## Files

- `AdaptiveRushBot.java` - main agent
- `metadata.json` - submission metadata

## Prerequisites

- Java JDK 17+
- Ollama installed locally: <https://ollama.ai/>
- MicroRTS repository root as working directory

## 1) Start Ollama and load a model

In a separate terminal:

```bash
ollama serve
```

Then pull the model used by this submission (default):

```bash
ollama pull llama3.1:8b
```

Optional environment variables (defaults shown):

```bash
export OLLAMA_ENDPOINT="http://localhost:11434/api/generate"
export OLLAMA_MODEL="llama3.1:8b"
```

## 2) Compile MicroRTS (full project)

From the repository root (`MicroRTS`):

```bash
find src -name '*.java' > sources.list
javac -cp "lib/*:bin" -d bin @sources.list
```

## 3) Compile this submission agent

From the repository root:

```bash
javac -cp "lib/*:bin:src" -d bin submissions/parker-riggs/AdaptiveRushBot.java
```

## 4) Configure game to use this agent

Edit `resources/config.properties` and set one side to:

```properties
AI1=ai.abstraction.submissions.parker_riggs.AdaptiveRushBot
```

Example opponent:

```properties
AI2=ai.abstraction.LightRush
```

Pick a map in the same file, for example:

```properties
map_location=maps/8x8/basesWorkers8x8.xml
```

## 5) Run games

From repository root:

```bash
javac -cp "lib/*:bin:src" -d bin submissions/parker-riggs/AdaptiveRushBot.java
java -cp "lib/*:bin" rts.MicroRTS -f resources/config.properties
```

If you want to watch the game window while still using `config.properties`, set:

```properties
headless=false
```

Then run the same command:

```bash
java -cp "lib/*:bin" rts.MicroRTS -f resources/config.properties
```

## 6) Where results are stored

Direct run (`rts.MicroRTS -f resources/config.properties`):
- Primary output is printed to the terminal where you launched the game.
- No automatic `results/` file is created by default for this single-run command.

GUI traces (optional):
- If you run the FrontEnd and check `Save Trace`, replay files are written as `trace1.xml`, `trace2.xml`, etc.
- These trace files are saved in the current working directory (typically the repository root).

Scripted loop runs:
- `./RunLoop.sh` writes per-run logs to `logs/run_YYYY-MM-DD_HH-MM-SS.log`.

Experiment/benchmark artifacts in this repo:
- Curated experiment folders are under `results/`.
- Benchmark JSON and leaderboard files are under `benchmark_results/`.

## How the heuristic works

When the LLM is unavailable the bot evaluates these rules in order and uses the
first one that fires:

1. **Emergency defense** -- an enemy attacker is within 8 tiles of our base AND
   we have zero military built: WorkerRush (skip the barracks, arm workers now).
2. **Commitment** -- already have 2+ of one combat type built: keep producing
   that type (Heavy/Light/Ranged).
3. **Counter the enemy**: enemy Heavy visible -> RangedRush; enemy Ranged
   visible -> LightRush; enemy Light visible -> HeavyRush.
4. **Pure-worker enemy** (workers but no military): LightRush.
5. **Pressure near base** (with some military already built): LightRush.
6. **Economy is set up OR past the time cutoff**: map-tier default opener.
7. **Default**: map-tier default opener.

### Map-size tiers

The bot bins maps into tiers and scales its worker threshold, time-to-commit
threshold, and default opener to each tier:

| Tier   | Map width | Worker threshold | Time cutoff (ticks) | Default opener |
|--------|-----------|------------------|---------------------|----------------|
| tiny   | <= 4      | 1                | 75                  | WorkerRush     |
| small  | <= 8      | 2                | 150                 | LightRush      |
| medium | <= 12     | 3                | 300                 | LightRush      |
| large  | >= 16     | 4                | 450                 | RangedRush     |

Bigger maps get a higher worker threshold because the longer supply lines need
more harvesters to sustain military production, and more time before the bot
commits to an opener because those games develop slower.

## Notes

- Package name uses underscore (`parker_riggs`) to match Java naming rules.
- The LLM is only queried once every 25 ticks to keep the game loop responsive.
- If Ollama is not running the bot works entirely on the heuristic described above.
- The end-of-game log line shows what percentage of decisions came from the LLM
  vs the heuristic, which is useful for verifying Ollama is actually being used.
- `gui.frontend.FrontEnd` has a fixed AI dropdown and does not auto-list submission classes.