# PenguinBot High-Level Documentation

## Overview

`PenguinBot` is a hybrid strategy agent:
- Core planner: `NaiveMCTS` (Monte Carlo Tree Search)
- Deterministic safety logic: opening build order + rush/advantage heuristics
- Optional strategic guidance: periodic Ollama calls that can switch global stance

Implementation class:
- `ai.mcts.submissions.penguin_bot.MCTSAgent`

Metadata:
- `submissions/penguin-bot/metadata.json`

## Design Goals

- Keep early game stable with a scripted opening.
- Use deterministic heuristics to avoid obvious tactical mistakes.
- Use an LLM only as a high-level stance controller (`DEFEND` vs `ATTACK`), not for low-level action generation.
- Keep runtime robust: if LLM or MCTS fails, fall back to built-in scripted policies.

## Strategy Pipeline (Per Decision Tick)

1. Opening phase check
- Until opening is complete, the bot runs a scripted macro opening.
- Opening ends when either:
  - Time passes `OPENING_END_TICK` (360), and
  - Opening goals are met.

2. Deterministic state assessment
- Computes military/economy counts and rush pressure.
- May force stance to `ATTACK` on clear advantage or to `DEFEND` when pressured.
- Sets `preferredUnit` (`HEAVY` or `RANGED`) from enemy composition and own mix.

3. Periodic LLM consultation (optional)
- Every `MCTS_LLM_INTERVAL` ticks (default 60).
- If being rushed, consults more frequently: `max(10, interval/4)`.
- LLM receives compact game summary and returns strict JSON stance advice.

4. Stance-to-MCTS biasing
- Tunes MCTS simulation horizon and exploration values based on stance.
- Swaps playout policy:
  - `DEFEND` -> `WorkerDefense`
  - `ATTACK` -> `HeavyRush` or `RangedRush` depending on `preferredUnit`
- Updates preferred action categories used to rank candidate actions.

5. MCTS search + stance-filtered action selection
- Runs MCTS from current state.
- Overrides action choice scoring to prefer stance-consistent actions and production.
- Rejects mixed actions that violate current global stance intent.

6. Runtime fallback
- If MCTS throws at action time:
  - `ATTACK` -> fallback rush policy (`HeavyRush`/`RangedRush`)
  - `DEFEND` -> `WorkerDefense`
- If fallback also fails, returns empty `PlayerAction`.

## Opening Build Logic

Opening script prioritizes:
- Early worker production (`OPENING_WORKERS_BEFORE_BARRACKS = 1`)
- First barracks as soon as feasible
- Economy target: 4 workers
- Military seed: at least 1 ranged + 1 heavy

The opening merges scripted production with `WorkerDefense` behavior for unassigned units, while checking resource-usage consistency before adding each action.

## Stance Model

Global stance is binary:
- `DEFEND`: no offensive splitting, stronger base safety/economy preferences.
- `ATTACK`: prefers forward pressure and combat production.

The bot intentionally forbids mixed/offense-defense splitting in a single policy mode. This is enforced both in LLM prompting and MCTS action filtering.

## Deterministic Triggers

Important deterministic triggers include:
- `forceAttack` when:
  - Enemy has effectively collapsed (no barracks, no combat, near-zero workers), or
  - Bot has strong combat/structural/economic lead.
- `forceDefend` when:
  - Under rush pressure, and
  - No decisive attack trigger is active, and
  - Own combat is not superior.

Rush detection uses enemy threatening units near own bases (`RUSH_ALERT_RADIUS = 7`).

## LLM Integration

### Purpose
- LLM does not output concrete commands.
- It only advises whether to switch stance and which combat unit type to prefer.

### Prompt Contract
Prompt asks for JSON with fields like:
- `switch_required`
- `target_stance`
- `necessity`
- `preferred_unit`
- `reason`
- optional `wholly_necessary`

A stance change is accepted only when:
- `switch_required = true`
- and necessity indicates a wholly necessary change
- and target differs from current stance

### Model Resolution and Fallback
On startup/use, the agent:
- Queries `OLLAMA_HOST/api/tags` for installed models.
- Prefers configured model if installed.
- Otherwise checks fallback candidates.
- If generation fails, retries once using another installed fallback model.

If Ollama calls fail entirely, the bot silently keeps deterministic behavior.

## MCTS Action Scoring Layer

`getMostVisitedActionIdx()` is customized to combine:
- Visit count
- Mean evaluation
- Preference score from stance-specific action intents

Intent classification tags candidate unit actions as:
- `OFFENSE`
- `DEFENSE`
- `ECONOMY`
- `NEUTRAL`

This keeps the MCTS final choice aligned with current strategic mode.

## Configuration

Environment variables:
- `MCTS_LLM_INTERVAL` (default `60`)
- `OLLAMA_HOST` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (default `llama3.1:8b`)
- `OLLAMA_FALLBACK_MODELS` (default `llama3.2:3b,mistral:7b,qwen2.5:7b`)

## Practical Behavior Summary

- Early game: safe macro opening + defense cover.
- Mid game: deterministic heuristics dominate stance unless LLM provides a strictly justified switch.
- Late game: MCTS remains primary executor, but its final action choice is strongly stance-shaped.
- Reliability: designed to degrade gracefully when LLM endpoint/model is unavailable.
