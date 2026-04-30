#!/usr/bin/env python3
"""
Auto-update the FALLBACK_AGENTS "Submission" array in docs/match.html.

Reads all valid submissions from submissions/ directory, builds the
fully-qualified class name for each, and patches the Submission entries
in match.html in-place.

Usage:
    python3 tournament/update_match_agents.py
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MATCH_HTML = PROJECT_ROOT / "docs" / "match.html"
SKIP_DIRS = {"_template"}


def discover_submissions():
    """Scan submissions/ and return a sorted list of agent entry dicts."""
    entries = []
    if not SUBMISSIONS_DIR.exists():
        return entries

    for meta_path in sorted(SUBMISSIONS_DIR.glob("*/metadata.json")):
        if meta_path.parent.name.startswith("_"):
            continue
        if meta_path.parent.name in SKIP_DIRS:
            continue

        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        team = meta.get("team_name", "")
        agent_class = meta.get("agent_class", "")
        agent_file = meta.get("agent_file", "")
        if not team or not agent_class:
            continue

        # Determine FQCN by reading the package declaration from the Java source
        java_file = meta_path.parent / agent_file
        fq_class = None
        if java_file.exists():
            try:
                src = java_file.read_text()
                pkg_match = re.search(r'^\s*package\s+([\w.]+)\s*;', src, re.MULTILINE)
                if pkg_match:
                    fq_class = pkg_match.group(1) + "." + agent_class
            except OSError:
                pass

        if not fq_class:
            team_pkg = team.replace("-", "_")
            fq_class = f"ai.abstraction.submissions.{team_pkg}.{agent_class}"

        requires_llm = meta.get("model_provider", "none") != "none"

        entries.append({
            "class": fq_class,
            "name": meta.get("display_name", team),
            "description": meta.get("description", ""),
            "requires_llm": requires_llm,
        })

    # Sort by display name for consistent ordering
    entries.sort(key=lambda e: e["name"].lower())
    return entries


def format_entry(entry, is_last):
    """Format a single agent entry as a JavaScript object literal line."""
    cls = json.dumps(entry["class"])
    name = json.dumps(entry["name"])
    desc = json.dumps(entry["description"])
    llm = "true" if entry["requires_llm"] else "false"
    comma = "" if is_last else ","
    return f'                {{ "class": {cls}, "name": {name}, "description": {desc}, "requires_llm": {llm} }}{comma}'


def update_match_html(entries):
    """Replace the Submission array in match.html with the given entries."""
    if not MATCH_HTML.exists():
        print(f"ERROR: {MATCH_HTML} not found")
        return False

    html = MATCH_HTML.read_text()

    # Match the "Submission": [ ... ] block
    # Pattern: "Submission": [ <entries across multiple lines> ]
    pattern = r'("Submission":\s*\[)\s*\n.*?\n(\s*\])'
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        print("ERROR: Could not find Submission array in match.html")
        return False

    # Build the replacement block
    lines = []
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        lines.append(format_entry(entry, is_last))

    replacement = match.group(1) + "\n" + "\n".join(lines) + "\n" + match.group(2)

    html = html[:match.start()] + replacement + html[match.end():]
    MATCH_HTML.write_text(html)
    return True


def main():
    entries = discover_submissions()
    if not entries:
        print("No valid submissions found.")
        return

    print(f"Found {len(entries)} submissions:")
    for e in entries:
        llm_tag = " [LLM]" if e["requires_llm"] else ""
        print(f"  {e['name']}: {e['class']}{llm_tag}")

    if update_match_html(entries):
        print(f"\nUpdated {MATCH_HTML} with {len(entries)} submission entries.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
