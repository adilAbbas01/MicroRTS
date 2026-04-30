#!/usr/bin/env python3
"""
Generate consolidated LLM leaderboard from benchmark result files.

Reads all benchmark_results/benchmark_*.json files, finds the best score
per model across runs, and generates:
  - benchmark_results/leaderboard.json  (consolidated best-score-per-model)
  - benchmark_results/LEADERBOARD.md    (rich per-opponent breakdown table)

Handles both v1.0 (2 opponents) and v2.0 (6 opponents) result files.

Usage:
    python3 generate_leaderboard.py
"""

import json
import glob
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = "benchmark_results"


def _score_to_grade(score):
    """Convert a benchmark score (0-100) to a letter grade."""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


def load_all_results():
    """Load all benchmark result files and return list of parsed entries."""
    pattern = os.path.join(RESULTS_DIR, "benchmark_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No benchmark files found in {RESULTS_DIR}/")
        return []

    all_entries = []

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: skipping {filepath}: {e}")
            continue

        version = data.get("version", "1.0")
        date = data.get("date", "")
        config = data.get("config", {})
        scores = data.get("benchmark_scores", {})

        # v2.0 has opponent_breakdown directly
        opponent_breakdowns = data.get("opponent_breakdown", {})
        agent_classes = data.get("agent_classes", {})

        # v1.0 may not have opponent_breakdown, build from detailed_results
        if not opponent_breakdowns and "detailed_results" in data:
            anchors = data.get("anchors", {})
            for model_name, results in data["detailed_results"].items():
                ref_games = results.get("reference_games", {})
                breakdown = {}
                for anchor_class, games in ref_games.items():
                    # Look up anchor info from the file's anchors or guess
                    if anchor_class in anchors:
                        info = anchors[anchor_class]
                    else:
                        # v1.0 fallback: known anchors
                        v1_anchors = {
                            "ai.RandomBiasedAI": {"name": "RandomBiasedAI", "weight": 40},
                            "ai.abstraction.WorkerRush": {"name": "WorkerRush", "weight": 60},
                        }
                        info = v1_anchors.get(anchor_class, {
                            "name": anchor_class.split(".")[-1],
                            "weight": 0
                        })
                    wins = sum(1 for g in games if g["result"] == "win")
                    draws = sum(1 for g in games if g["result"] == "draw")
                    losses = len(games) - wins - draws
                    breakdown[info.get("name", anchor_class)] = {
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                    }
                opponent_breakdowns[model_name] = breakdown

        elim_at = data.get("eliminated_at", {})
        fmt = data.get("format", "full")

        for model_name, score in scores.items():
            entry = {
                "model": model_name,
                "score": score,
                "version": version,
                "format": fmt,
                "date": date,
                "map": config.get("map", ""),
                "games_per_matchup": config.get("games_per_matchup", 1),
                "opponents": opponent_breakdowns.get(model_name, {}),
                "eliminated_at": elim_at.get(model_name),
                "agent_class": agent_classes.get(model_name, ""),
                "source_file": os.path.basename(filepath),
            }
            all_entries.append(entry)

    print(f"Loaded {len(all_entries)} entries from {len(files)} result files")
    return all_entries


def find_best_per_model(entries):
    """Find the best score per model. Prefer v2.0 entries when scores are equal."""
    best = {}
    for entry in entries:
        model = entry["model"]
        if model not in best:
            best[model] = entry
        else:
            prev = best[model]
            # Prefer higher score; on tie prefer v2.0 over v1.0; on tie prefer newer
            if (entry["score"] > prev["score"] or
                (entry["score"] == prev["score"] and entry["version"] > prev["version"]) or
                (entry["score"] == prev["score"] and entry["version"] == prev["version"]
                 and entry["date"] > prev["date"])):
                best[model] = entry
    return best


def generate_leaderboard_json(best_per_model):
    """Generate consolidated leaderboard.json."""
    entries = []
    for model, entry in sorted(best_per_model.items(), key=lambda x: x[1]["score"], reverse=True):
        entries.append({
            "model": entry["model"],
            "score": entry["score"],
            "grade": _score_to_grade(entry["score"]),
            "version": entry["version"],
            "format": entry.get("format", "full"),
            "eliminated_at": entry.get("eliminated_at"),
            "agent_class": entry.get("agent_class", ""),
            "date": entry["date"],
            "map": entry["map"],
            "games_per_matchup": entry["games_per_matchup"],
            "opponents": entry["opponents"],
            "source_file": entry["source_file"],
        })

    output = {
        "generated": datetime.now().isoformat(),
        "description": "Best benchmark score per model across all runs",
        "entries": entries
    }

    outfile = os.path.join(RESULTS_DIR, "leaderboard.json")
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Written: {outfile}")
    return output


def generate_leaderboard_markdown(best_per_model):
    """Generate LEADERBOARD.md with rich per-opponent breakdown."""
    sorted_models = sorted(best_per_model.values(), key=lambda x: x["score"], reverse=True)

    # Collect all opponent names across all entries (ordered)
    all_opponents = []
    seen = set()
    for entry in sorted_models:
        for opp_name in entry.get("opponents", {}):
            if opp_name not in seen:
                all_opponents.append(opp_name)
                seen.add(opp_name)

    lines = []
    lines.append("# MicroRTS LLM Leaderboard")
    lines.append("")
    lines.append("Best benchmark score per model across all runs.")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    # Summary table
    lines.append("## Rankings")
    lines.append("")
    header = "| Rank | Model | Score | Grade | Eliminated at |"
    sep = "|------|-------|-------|-------|---------------|"
    for opp in all_opponents:
        header += f" {opp} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for rank, entry in enumerate(sorted_models, 1):
        grade = _score_to_grade(entry["score"])
        elim = entry.get("eliminated_at")
        elim_str = elim if elim else "cleared all"
        row = f"| {rank} | {entry['model']} | **{entry['score']}** | {grade} | {elim_str} |"
        opponents = entry.get("opponents", {})
        is_elim_format = entry.get("format") == "single-elimination"
        past_elim = False
        for opp in all_opponents:
            if past_elim:
                row += " -- |"
                continue
            data = opponents.get(opp, {})
            if data:
                w = data.get("wins", 0)
                d = data.get("draws", 0)
                l = data.get("losses", 0)
                row += f" {w}W/{d}D/{l}L |"
                if is_elim_format and w == 0:
                    past_elim = True
            else:
                row += " - |"
        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-model detail cards
    lines.append("## Detailed Breakdown")
    lines.append("")

    for entry in sorted_models:
        grade = _score_to_grade(entry["score"])
        elim = entry.get("eliminated_at")
        elim_note = f" -- eliminated at {elim}" if elim else " -- cleared all"
        lines.append(f"### {entry['model']} - {entry['score']} pts ({grade}{elim_note})")
        lines.append("")
        lines.append(f"- **Date:** {entry['date'][:10]}")
        lines.append(f"- **Arena version:** v{entry['version']}")
        lines.append(f"- **Format:** {entry.get('format', 'full')}")
        lines.append(f"- **Map:** `{entry['map']}`")
        lines.append(f"- **Games per matchup:** {entry['games_per_matchup']}")
        lines.append(f"- **Source:** `{entry['source_file']}`")
        lines.append("")

        opponents = entry.get("opponents", {})
        if opponents:
            lines.append("| Opponent | W | D | L | Weighted Pts |")
            lines.append("|----------|---|---|---|-------------|")
            for opp_name, data in opponents.items():
                w = data.get("wins", 0)
                d = data.get("draws", 0)
                l = data.get("losses", 0)
                pts = data.get("weighted_points", "-")
                lines.append(f"| {opp_name} | {w} | {d} | {l} | {pts} |")
            lines.append("")

    lines.append("---")
    lines.append("")

    # Grade scale
    lines.append("## Grade Scale")
    lines.append("")
    lines.append("| Grade | Score Range | Description |")
    lines.append("|-------|-------------|-------------|")
    lines.append("| A+ | 90-100 | Excellent - beats hard AIs consistently |")
    lines.append("| A | 80-89 | Very Good - competes with hard AIs |")
    lines.append("| B | 70-79 | Good - beats medium, challenges hard |")
    lines.append("| C | 60-69 | Average - beats easy and some medium |")
    lines.append("| D | 40-59 | Below Average - draws common |")
    lines.append("| F | 0-39 | Failing - losses/timeouts |")
    lines.append("")

    # Version note
    lines.append("## Version Notes")
    lines.append("")
    lines.append("- **v1.0**: 2 opponents, play all (RandomBiasedAI 40pts + WorkerRush 60pts)")
    lines.append("- **v2.0**: 6 opponents, single-elimination (RandomBiasedAI 10 -> HeavyRush 20 -> LightRush 15 -> WorkerRush 15 -> Tiamat 20 -> CoacAI 20)")
    lines.append("")
    lines.append("Scores from different versions are **not directly comparable** due to different opponent sets, weights, and format.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `generate_leaderboard.py`*")
    lines.append("")

    outfile = os.path.join(RESULTS_DIR, "LEADERBOARD.md")
    with open(outfile, 'w') as f:
        f.write("\n".join(lines))

    print(f"Written: {outfile}")


def main():
    print("=" * 50)
    print("MicroRTS Leaderboard Generator")
    print("=" * 50)
    print()

    Path(RESULTS_DIR).mkdir(exist_ok=True)

    entries = load_all_results()
    if not entries:
        print("No results to process.")
        return

    best = find_best_per_model(entries)
    print(f"Found {len(best)} unique models")
    print()

    # Print quick summary
    for model, entry in sorted(best.items(), key=lambda x: x[1]["score"], reverse=True):
        grade = _score_to_grade(entry["score"])
        print(f"  {model}: {entry['score']} ({grade}) [v{entry['version']}]")
    print()

    generate_leaderboard_json(best)
    generate_leaderboard_markdown(best)

    print("\nDone!")


if __name__ == "__main__":
    main()
