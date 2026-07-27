#!/usr/bin/env python3
"""
AQI & Attribution Briefing CLI (Upwind)
Quick CLI path for inspecting AQI status and testing LLM narrative briefings.
"""

import sys
import os
import argparse
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.services.geocode import geocode_location
from backend.services.airnow import fetch_airnow_observation
from backend.services.openmeteo import fetch_openmeteo_aqi
from backend.engine.signals import assemble_evidence_signals
from backend.engine.score import score_hypotheses
from backend.llm import generate_narrative_briefing
from backend.db import get_cached_narrative, set_cached_narrative

# Try importing Rich for clean terminal formatting
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def get_aqi_color(aqi: int) -> str:
    """Return color style string based on AQI value."""
    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "yellow"
    elif aqi <= 150:
        return "orange3"
    elif aqi <= 200:
        return "red"
    elif aqi <= 300:
        return "magenta"
    else:
        return "bold magenta"


def parse_zip_inputs(raw_inputs: List[str]) -> List[str]:
    """Extract clean 5-digit US ZIP codes or location strings from raw input list."""
    zips = []
    for item in raw_inputs:
        # Split by comma or whitespace
        parts = [p.strip() for p in re.split(r'[,;\s]+', item) if p.strip()]
        for part in parts:
            if part not in zips:
                zips.append(part)
    return zips


async def fetch_zip_briefing(zip_code: str, use_cache: bool = True) -> Dict[str, Any]:
    """Fetch observation, assemble signals, score hypotheses, and generate briefing for a ZIP code."""
    location = geocode_location(zip_code)
    if not location:
        return {
            "zip": zip_code,
            "error": f"Could not geocode '{zip_code}'. Please provide a valid 5-digit US ZIP code."
        }

    lat, lon = location["lat"], location["lon"]
    loc_key = location.get("zip_code") or f"{lat:.2f}_{lon:.2f}"

    # 1. Fetch AQI observation
    observation = await fetch_airnow_observation(lat, lon)
    if not observation:
        observation = await fetch_openmeteo_aqi(lat, lon)

    if not observation:
        return {
            "zip": zip_code,
            "location": location.get("name", zip_code),
            "error": "Air quality data is currently unavailable for this location."
        }

    # 2. Assemble signals & score hypotheses
    signals, execution_trace = await assemble_evidence_signals(location, observation)
    hypotheses, open_questions = score_hypotheses(observation, signals)

    # 3. Handle Narrative (Cache vs Fresh)
    hour_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
    cache_key = f"why_{loc_key}_{hour_stamp}"

    narrative = None
    if use_cache:
        narrative = get_cached_narrative(cache_key)

    if not narrative:
        narrative = await generate_narrative_briefing(
            location, observation, signals, hypotheses, open_questions
        )
        if use_cache:
            set_cached_narrative(cache_key, narrative, {
                "location": location,
                "observation": observation,
                "signals": signals,
                "hypotheses": hypotheses,
                "open_questions": open_questions,
                "narrative": narrative
            })

    top_hypothesis = hypotheses[0] if hypotheses else None

    return {
        "zip": zip_code,
        "location": location.get("name", zip_code),
        "lat": lat,
        "lon": lon,
        "observation": observation,
        "primary_hypothesis": top_hypothesis["id"] if top_hypothesis else None,
        "hypothesis_title": top_hypothesis["title"] if top_hypothesis else None,
        "confidence": top_hypothesis["confidence"] if top_hypothesis else None,
        "open_questions": open_questions,
        "narrative": narrative
    }


def render_terminal_output(results: List[Dict[str, Any]], no_color: bool = False):
    """Render clean, human-readable terminal output for briefing results."""
    console = Console(highlight=False) if (HAS_RICH and not no_color) else None

    for i, res in enumerate(results):
        zip_code = res["zip"]
        if "error" in res:
            if console:
                console.print(f"[bold red]❌ [{zip_code}]: {res['error']}[/bold red]\n")
            else:
                print(f"❌ [{zip_code}]: {res['error']}\n")
            continue

        loc_name = res.get("location", zip_code)
        obs = res.get("observation", {})
        aqi_val = obs.get("aqi", 0)
        category = obs.get("category", "Unknown")
        pollutant = obs.get("primary_pollutant", "PM2.5")
        narrative = res.get("narrative", "")

        if console:
            color = get_aqi_color(aqi_val)
            header_text = Text()
            header_text.append(f"[{zip_code}] ", style="bold cyan")
            header_text.append(f"{loc_name} ", style="bold white")
            header_text.append(f"(AQI {aqi_val} • {category}, {pollutant})", style=f"bold {color}")

            panel_content = f"{narrative}"
            if res.get("open_questions"):
                questions_str = "\n".join([f"❓ {q}" for q in res["open_questions"]])
                panel_content += f"\n\n[italic dim]{questions_str}[/italic dim]"

            console.print(Panel(
                panel_content,
                title=header_text,
                title_align="left",
                border_style=color,
                padding=(1, 2)
            ))
            console.print()
        else:
            print(f"[{zip_code}] {loc_name} (AQI {aqi_val} - {category}, {pollutant})")
            print("-" * 60)
            print(narrative)
            if res.get("open_questions"):
                print()
                for q in res["open_questions"]:
                    print(f"❓ {q}")
            print()


async def main_async():
    parser = argparse.ArgumentParser(
        description="AQI & Attribution Briefing CLI (Upwind)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m backend.cli 90210
  python -m backend.cli 90210 94103
  python -m backend.cli --zip 90210,94103 --no-cache
  python -m backend.cli 90210 --json
  python -m backend.cli 90210 --raw
"""
    )
    parser.add_argument(
        "positional_zips",
        nargs="*",
        help="One or more ZIP codes or location queries (comma or space separated)"
    )
    parser.add_argument(
        "-z", "--zip",
        dest="flag_zips",
        help="ZIP code(s) separated by commas"
    )
    parser.add_argument(
        "-f", "--no-cache", "--force",
        dest="no_cache",
        action="store_true",
        help="Bypass SQLite cache and force fresh LLM narrative generation (useful for prompt testing)"
    )
    parser.add_argument(
        "-r", "--raw",
        action="store_true",
        help="Output raw briefing text only (no header panels or metadata)"
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output results as clean JSON array"
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Evaluate generated narrative via LLM Judge (Groq API) and display verdict"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable color rendering in terminal output"
    )

    args = parser.parse_args()

    # Gather ZIP inputs from positionals or flag
    raw_inputs = list(args.positional_zips)
    if args.flag_zips:
        raw_inputs.append(args.flag_zips)

    zips = parse_zip_inputs(raw_inputs)

    # Prompt interactively if no ZIP code passed
    if not zips:
        try:
            user_input = input("Enter US ZIP code(s) (comma-separated): ").strip()
            if not user_input:
                print("No ZIP code provided. Exiting.")
                sys.exit(0)
            zips = parse_zip_inputs([user_input])
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            sys.exit(0)

    use_cache = not args.no_cache

    # Fetch briefings
    results = []
    if HAS_RICH and not (args.json or args.raw or args.no_color) and sys.stdout.isatty():
        console = Console()
        with console.status("[bold cyan]Fetching AQI & generating narrative briefings...", spinner="dots"):
            for zip_code in zips:
                res = await fetch_zip_briefing(zip_code, use_cache=use_cache)
                results.append(res)
    else:
        for zip_code in zips:
            res = await fetch_zip_briefing(zip_code, use_cache=use_cache)
            results.append(res)

    # Evaluate via LLM Judge if requested
    if args.judge:
        from backend.llm_judge import judge_narrative
        for res in results:
            if "narrative" in res and "error" not in res:
                loc_obj = geocode_location(res["zip"])
                obs_obj = res.get("observation", {})
                signals, _ = await assemble_evidence_signals(loc_obj, obs_obj)
                hypotheses, open_questions = score_hypotheses(obs_obj, signals)
                payload = {
                    "location": loc_obj, "observation": obs_obj,
                    "signals": signals, "hypotheses": hypotheses, "open_questions": open_questions,
                    "narrative": res["narrative"]
                }
                verdict = await judge_narrative(payload, res["narrative"])
                res["judge_verdict"] = verdict

    # Render output based on selected mode
    if args.json:
        print(json.dumps(results, indent=2))
    elif args.raw:
        for res in results:
            if "narrative" in res:
                print(res["narrative"])
                if "judge_verdict" in res:
                    print("\n--- LLM JUDGE VERDICT ---")
                    print(json.dumps(res["judge_verdict"], indent=2))
            elif "error" in res:
                print(f"Error: {res['error']}")
            print()
    else:
        render_terminal_output(results, no_color=args.no_color)
        if args.judge:
            for res in results:
                v = res.get("judge_verdict")
                if v:
                    status_emoji = "✅ PASS" if v.get("verdict") == "pass" else ("❌ FAIL" if v.get("verdict") == "fail" else "⚠️ " + str(v.get("verdict")).upper())
                    print(f"⚖️ LLM Judge Verdict [{res['zip']}]: {status_emoji}")
                    print(f"   Reasoning: {v.get('reasoning')}")
                    if v.get("hallucinations"):
                        print(f"   Hallucinations: {v.get('hallucinations')}")
                    if v.get("leaked_jargon"):
                        print(f"   Leaked Jargon: {v.get('leaked_jargon')}")
                    print()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
