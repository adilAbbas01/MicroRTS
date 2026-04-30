#!/usr/bin/env python3
"""
MicroRTS Match Server - On-demand head-to-head matches via HTTP.

Provides a simple REST API for running games between any two agents.
The docs/match.html page uses this to let visitors pick agents and watch results.

Usage:
    python3 tournament/match_server.py [--port 8080] [--host 0.0.0.0]

Endpoints:
    GET  /api/agents        List available agents (built-in + submissions)
    GET  /api/maps           List available maps
    POST /api/match          Run a match: {"ai1": "...", "ai2": "...", "map": "...", "max_cycles": 3000}
    GET  /api/match/<id>     Poll match status/result
    GET  /                   Redirect to docs/match.html

Requires: Java compiled (ant build), MicroRTS bin/ directory present.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# --- Configuration ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
LIB_DIR = PROJECT_ROOT / "lib"
MAPS_DIR = PROJECT_ROOT / "maps"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
CONFIG_TEMPLATE = PROJECT_ROOT / "resources" / "config.properties"

TRACES_DIR = PROJECT_ROOT / "tournament" / "traces"
DEFAULT_MAP = "maps/8x8/basesWorkers8x8.xml"
DEFAULT_MAX_CYCLES = 3000
GAME_TIMEOUT = 300  # 5 minutes max per match
MAX_CONCURRENT = 3  # Max simultaneous matches

# --- Built-in agents ---

BUILTIN_AGENTS = {
    "ai.RandomBiasedAI": {
        "name": "RandomBiasedAI",
        "category": "Baseline",
        "description": "Random but prefers useful actions",
        "requires_llm": False,
    },
    "ai.RandomAI": {
        "name": "RandomAI",
        "category": "Baseline",
        "description": "Purely random actions",
        "requires_llm": False,
    },
    "ai.PassiveAI": {
        "name": "PassiveAI",
        "category": "Baseline",
        "description": "Does nothing",
        "requires_llm": False,
    },
    "ai.abstraction.WorkerRush": {
        "name": "WorkerRush",
        "category": "Rush",
        "description": "Aggressive worker rush",
        "requires_llm": False,
    },
    "ai.abstraction.LightRush": {
        "name": "LightRush",
        "category": "Rush",
        "description": "Builds light units aggressively",
        "requires_llm": False,
    },
    "ai.abstraction.HeavyRush": {
        "name": "HeavyRush",
        "category": "Rush",
        "description": "Builds heavy units aggressively",
        "requires_llm": False,
    },
    "ai.abstraction.RangedRush": {
        "name": "RangedRush",
        "category": "Rush",
        "description": "Builds ranged units aggressively",
        "requires_llm": False,
    },
    "ai.competition.tiamat.Tiamat": {
        "name": "Tiamat",
        "category": "Competition Winner",
        "description": "CoG 2017/2018 champion",
        "requires_llm": False,
    },
    "ai.coac.CoacAI": {
        "name": "CoacAI",
        "category": "Competition Winner",
        "description": "CoG 2020 champion",
        "requires_llm": False,
    },
    "ai.abstraction.ollama": {
        "name": "Ollama LLM",
        "category": "LLM",
        "description": "Pure LLM agent via Ollama",
        "requires_llm": True,
    },
    "ai.abstraction.HybridLLMRush": {
        "name": "HybridLLMRush",
        "category": "LLM",
        "description": "Rule-based + periodic LLM strategy switching",
        "requires_llm": True,
    },
    "ai.abstraction.StrategicLLMAgent": {
        "name": "StrategicLLMAgent",
        "category": "LLM",
        "description": "8 strategies + tactical params from LLM",
        "requires_llm": True,
    },
    "ai.mcts.llmguided.LLMInformedMCTS": {
        "name": "LLMInformedMCTS",
        "category": "LLM",
        "description": "MCTS with LLM policy priors",
        "requires_llm": True,
    },
}

# --- Match state ---

matches = {}  # id -> match dict
matches_lock = threading.Lock()
running_count = 0
running_lock = threading.Lock()


def discover_submission_agents():
    """Scan submissions/ for valid agents."""
    agents = {}
    if not SUBMISSIONS_DIR.exists():
        return agents
    for meta_path in SUBMISSIONS_DIR.glob("*/metadata.json"):
        if meta_path.parent.name.startswith("_"):
            continue
        try:
            meta = json.loads(meta_path.read_text())
            team = meta.get("team_name", "")
            agent_class = meta.get("agent_class", "")
            if not team or not agent_class:
                continue
            # Determine fully qualified class name
            team_pkg = team.replace("-", "_")
            agent_file = meta.get("agent_file", "")
            # Check if the agent .java exists
            java_file = meta_path.parent / agent_file
            if not java_file.exists():
                continue
            # Detect package from the agent source
            fq_class = None
            try:
                src = java_file.read_text()
                pkg_match = re.search(r'^\s*package\s+([\w.]+)\s*;', src, re.MULTILINE)
                if pkg_match:
                    fq_class = pkg_match.group(1) + "." + agent_class
            except Exception:
                pass
            if not fq_class:
                fq_class = f"ai.abstraction.submissions.{team_pkg}.{agent_class}"
            requires_llm = meta.get("model_provider", "none") != "none"
            agents[fq_class] = {
                "name": meta.get("display_name", team),
                "category": "Submission",
                "description": meta.get("description", ""),
                "requires_llm": requires_llm,
            }
        except (json.JSONDecodeError, KeyError):
            continue
    return agents


def get_all_agents():
    """Return merged dict of built-in + submission agents."""
    agents = dict(BUILTIN_AGENTS)
    agents.update(discover_submission_agents())
    return agents


def discover_maps():
    """List available maps."""
    maps = []
    if not MAPS_DIR.exists():
        return maps
    for xml in sorted(MAPS_DIR.rglob("*.xml")):
        rel = xml.relative_to(PROJECT_ROOT)
        maps.append(str(rel))
    return maps


def parse_map_xml(map_path):
    """Parse a map XML file into a JSON-friendly dict."""
    full_path = PROJECT_ROOT / map_path
    if not full_path.exists() or not full_path.is_file():
        return None
    # Ensure path is under maps/ to prevent traversal
    try:
        full_path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    tree = ET.parse(str(full_path))
    root = tree.getroot()
    width = int(root.get("width", 0))
    height = int(root.get("height", 0))
    terrain = ""
    terrain_el = root.find("terrain")
    if terrain_el is not None and terrain_el.text:
        terrain = terrain_el.text.strip()
    units = []
    for u in root.iter("rts.units.Unit"):
        units.append({
            "type": u.get("type", ""),
            "player": int(u.get("player", -1)),
            "x": int(u.get("x", 0)),
            "y": int(u.get("y", 0)),
            "resources": int(u.get("resources", 0)),
        })
    players = []
    for p in root.iter("rts.Player"):
        players.append({
            "ID": int(p.get("ID", 0)),
            "resources": int(p.get("resources", 0)),
        })
    return {
        "width": width,
        "height": height,
        "terrain": terrain,
        "units": units,
        "players": players,
    }


def write_config(ai1, ai2, map_location, max_cycles):
    """Write a temporary config.properties for this match."""
    fd, path = tempfile.mkstemp(suffix=".properties", prefix="match_")
    with os.fdopen(fd, 'w') as f:
        f.write(f"launch_mode=STANDALONE\n")
        f.write(f"map_location={map_location}\n")
        f.write(f"max_cycles={max_cycles}\n")
        f.write(f"headless=true\n")
        f.write(f"partially_observable=false\n")
        f.write(f"UTT_version=2\n")
        f.write(f"conflict_policy=1\n")
        f.write(f"AI1={ai1}\n")
        f.write(f"AI2={ai2}\n")
    return path


def parse_game_result(output, returncode=0):
    """Parse structured output from Game.java."""
    result = {"winner": None, "final_tick": None, "raw_output": "", "error": None}
    # Keep last 50 lines for context
    lines = output.strip().split('\n')
    result["raw_output"] = '\n'.join(lines[-50:])

    for line in lines:
        line = line.strip()
        if line.startswith("WINNER:"):
            try:
                result["winner"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("FINAL_TICK:"):
            try:
                result["final_tick"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        # RecordLLMGame uses different labels
        elif line.startswith("Final tick:"):
            try:
                result["final_tick"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("Winner: Player 0"):
            result["winner"] = 0
        elif line.startswith("Winner: Player 1"):
            result["winner"] = 1
        elif line.startswith("Result: Draw"):
            result["winner"] = -1
        elif line.startswith("CRASHED: Player"):
            try:
                crashed_player = int(line.split("Player")[1].strip())
                result["crashed"] = crashed_player
            except (ValueError, IndexError):
                pass
        elif line.startswith("ERROR:"):
            result["error"] = line[6:].strip()

    # Detect failed games: non-zero exit or no result parsed
    if returncode != 0 and result["winner"] is None:
        if not result["error"]:
            # Extract error from output
            for line in lines:
                if "Exception" in line or "Error" in line:
                    result["error"] = line.strip()
                    break
            if not result["error"]:
                result["error"] = f"Game process exited with code {returncode}"

    return result


def run_match(match_id):
    """Execute a match in a subprocess."""
    global running_count

    with matches_lock:
        match = matches[match_id]
        match["status"] = "running"
        match["started_at"] = time.time()

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_prefix = str(TRACES_DIR / match_id)

    try:
        classpath = f"{LIB_DIR}/*:{BIN_DIR}"
        cmd = [
            "java", "-cp", classpath,
            "tests.trace.RecordLLMGame",
            match["ai1"], match["ai2"],
            trace_prefix,
            str(match["max_cycles"]),
            match["map"],
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=GAME_TIMEOUT,
            cwd=str(PROJECT_ROOT),
        )

        output = proc.stdout + proc.stderr
        result = parse_game_result(output, proc.returncode)

        # Check if trace JSON was saved
        trace_json = trace_prefix + ".json"
        has_trace = os.path.exists(trace_json)

        with matches_lock:
            if result["error"]:
                match["status"] = "error"
                match["error"] = result["error"]
            else:
                match["status"] = "completed"
            match["completed_at"] = time.time()
            match["duration"] = match["completed_at"] - match["started_at"]
            match["winner"] = result["winner"]
            match["final_tick"] = result["final_tick"]
            if result.get("crashed") is not None:
                crashed = result["crashed"]
                match["crashed"] = crashed
                crash_name = match["ai1_name"] if crashed == 0 else match["ai2_name"]
                match["winner_label"] = f"{crash_name} crashed"
            elif result["winner"] == 0:
                match["winner_label"] = "Player 1 (AI1)"
            elif result["winner"] == 1:
                match["winner_label"] = "Player 2 (AI2)"
            else:
                match["winner_label"] = "Draw"
            match["game_log"] = result["raw_output"]
            match["has_trace"] = has_trace

    except subprocess.TimeoutExpired:
        with matches_lock:
            match["status"] = "timeout"
            match["completed_at"] = time.time()
            match["duration"] = match["completed_at"] - match["started_at"]
            match["winner_label"] = "Timeout"
            match["game_log"] = f"Game timed out after {GAME_TIMEOUT}s"
    except Exception as e:
        with matches_lock:
            match["status"] = "error"
            match["completed_at"] = time.time()
            match["error"] = str(e)
            match["game_log"] = str(e)
    finally:
        # Clean up XML trace (keep JSON for replay)
        xml_trace = trace_prefix + ".xml"
        if os.path.exists(xml_trace):
            os.unlink(xml_trace)
        with running_lock:
            running_count -= 1


# --- HTTP Handler ---

class MatchHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Quieter logging
        print(f"[{self.log_date_time_string()}] {format % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "" or path == "/index.html":
            self.send_response(302)
            self.send_header("Location", "/match.html")
            self.end_headers()
            return

        # Serve static files from docs/
        if path.endswith((".html", ".css", ".js", ".json")):
            file_path = PROJECT_ROOT / "docs" / path.lstrip("/")
            if file_path.exists() and file_path.is_file():
                content = file_path.read_bytes()
                self.send_response(200)
                if path.endswith(".html"):
                    ct = "text/html"
                elif path.endswith(".css"):
                    ct = "text/css"
                elif path.endswith(".js"):
                    ct = "application/javascript"
                else:
                    ct = "application/json"
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        if path == "/api/agents":
            agents = get_all_agents()
            # Group by category
            grouped = {}
            for class_name, info in sorted(agents.items(), key=lambda x: (x[1]["category"], x[1]["name"])):
                cat = info["category"]
                if cat not in grouped:
                    grouped[cat] = []
                grouped[cat].append({
                    "class": class_name,
                    "name": info["name"],
                    "description": info["description"],
                    "requires_llm": info["requires_llm"],
                })
            self.send_json({"agents": grouped})
            return

        if path == "/api/maps":
            maps = discover_maps()
            self.send_json({"maps": maps, "default": DEFAULT_MAP})
            return

        if path == "/api/map":
            qs = parse_qs(parsed.query)
            map_path = qs.get("path", [None])[0]
            if not map_path:
                self.send_json({"error": "path parameter required"}, 400)
                return
            data = parse_map_xml(map_path)
            if not data:
                self.send_json({"error": "Map not found"}, 404)
                return
            self.send_json(data)
            return

        if path.startswith("/api/trace/"):
            match_id = path.split("/")[-1]
            # Sanitize: only allow alphanumeric/dash to prevent path traversal
            if not re.match(r'^[a-zA-Z0-9-]+$', match_id):
                self.send_json({"error": "Invalid match ID"}, 400)
                return
            trace_path = TRACES_DIR / (match_id + ".json")
            if not trace_path.exists():
                self.send_json({"error": "Trace not found"}, 404)
                return
            content = trace_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path.startswith("/api/match/"):
            match_id = path.split("/")[-1]
            with matches_lock:
                match = matches.get(match_id)
            if not match:
                self.send_json({"error": "Match not found"}, 404)
                return
            # Return safe subset
            safe = {k: v for k, v in match.items() if k != "thread"}
            self.send_json(safe)
            return

        self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/match":
            global running_count

            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self.send_json({"error": "Invalid JSON"}, 400)
                return

            ai1 = data.get("ai1", "").strip()
            ai2 = data.get("ai2", "").strip()
            map_loc = data.get("map", DEFAULT_MAP).strip()
            max_cycles = int(data.get("max_cycles", DEFAULT_MAX_CYCLES))

            if not ai1 or not ai2:
                self.send_json({"error": "ai1 and ai2 are required"}, 400)
                return

            # Validate agents exist
            all_agents = get_all_agents()
            if ai1 not in all_agents:
                self.send_json({"error": f"Unknown agent: {ai1}"}, 400)
                return
            if ai2 not in all_agents:
                self.send_json({"error": f"Unknown agent: {ai2}"}, 400)
                return

            # Clamp max_cycles
            max_cycles = max(100, min(max_cycles, 10000))

            # Check concurrency limit
            with running_lock:
                if running_count >= MAX_CONCURRENT:
                    self.send_json({"error": "Server busy. Try again in a moment."}, 429)
                    return
                running_count += 1

            match_id = str(uuid.uuid4())[:8]
            match = {
                "id": match_id,
                "ai1": ai1,
                "ai2": ai2,
                "ai1_name": all_agents[ai1]["name"],
                "ai2_name": all_agents[ai2]["name"],
                "map": map_loc,
                "max_cycles": max_cycles,
                "status": "queued",
                "created_at": time.time(),
            }

            with matches_lock:
                matches[match_id] = match

            t = threading.Thread(target=run_match, args=(match_id,), daemon=True)
            match["thread"] = t
            t.start()

            self.send_json({"match_id": match_id, "status": "queued"}, 202)
            return

        self.send_json({"error": "Not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description="MicroRTS Match Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Verify build exists
    if not BIN_DIR.exists():
        print(f"ERROR: {BIN_DIR} not found. Run 'ant build' first.")
        sys.exit(1)

    print(f"MicroRTS Match Server starting on http://{args.host}:{args.port}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Agents: {len(get_all_agents())} available")
    print(f"  Maps: {len(discover_maps())} available")
    print(f"  Max concurrent matches: {MAX_CONCURRENT}")
    print(f"  Game timeout: {GAME_TIMEOUT}s")
    print()
    print(f"Open http://localhost:{args.port}/match.html to run matches.")

    server = HTTPServer((args.host, args.port), MatchHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
