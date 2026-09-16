#!/usr/bin/env python3
"""
main.py
=======

This is the command-line entry point for the AI Research Assistant.

Its job is deliberately narrow: handle user input/output (reading the topic,
showing progress, printing/saving the final report) using the `rich`
library for nice terminal formatting, while delegating all of the actual
"thinking" to `agent.py`. This separation (interface vs. logic) is a common
and useful pattern -- it means you could swap this CLI for a web UI later
without touching agent.py or tools.py at all.

Usage:
    python main.py "history of the Roman aqueducts"
    python main.py                # will prompt you for a topic interactively
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status

from agent import run_research_agent

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Where finished reports get saved. Relative to this file, so it works no
# matter what directory you run `python main.py` from.
REPORTS_DIR = Path(__file__).parent / "reports"


def load_api_key(console: Console) -> str:
    """
    Load the Anthropic API key from the environment (or a local .env file).

    We use `python-dotenv` so that a developer can keep their key in a
    `.env` file (which is gitignored -- see .gitignore) instead of ever
    typing it into a terminal or committing it to source control.
    """
    # load_dotenv() reads a `.env` file in the current directory (if present)
    # and copies its KEY=VALUE lines into os.environ, where os.getenv can
    # find them. If there's no .env file, this simply does nothing.
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] ANTHROPIC_API_KEY is not set.\n"
            "Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and add your "
            "API key, or export it in your shell before running this script."
        )
        sys.exit(1)

    return api_key


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Research Assistant -- autonomously researches a topic "
        "and writes a Markdown report."
    )
    parser.add_argument(
        "topic",
        nargs="?",  # "?" means this argument is optional
        default=None,
        help="The research topic/question. If omitted, you'll be prompted for it.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    """
    Turn a research topic into a filesystem-safe filename fragment.

    e.g. "The Rise of Renewable Energy!" -> "the-rise-of-renewable-energy"
    """
    text = text.lower().strip()
    # Replace anything that isn't a letter, digit, or whitespace with nothing.
    text = re.sub(r"[^\w\s-]", "", text)
    # Collapse whitespace/underscores into single hyphens.
    text = re.sub(r"[\s_]+", "-", text)
    # Trim to a reasonable length so filenames don't get absurd.
    return text[:60].strip("-") or "research-report"


def save_report(topic: str, report_markdown: str) -> Path:
    """Save the report to reports/<slug>-<timestamp>.md and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{slugify(topic)}-{timestamp}.md"
    filepath = REPORTS_DIR / filename

    filepath.write_text(report_markdown, encoding="utf-8")
    return filepath


def make_progress_callback(console: Console, status: Status):
    """
    Build an `on_event` callback (matching agent.EventCallback) that renders
    the agent's progress live in the terminal using `rich`.

    We build this as a closure (a function that returns a function) so it
    can capture `console` and `status` without agent.py needing to know
    anything about rich at all -- agent.py just calls `on_event(type, data)`
    and doesn't care who's listening or how it's displayed.
    """

    def on_event(event_type: str, data: dict) -> None:
        if event_type == "turn_start":
            status.update(
                f"[bold cyan]Thinking...[/bold cyan] "
                f"(turn {data['turn']}/{data['max_turns']})"
            )

        elif event_type == "tool_call":
            name = data["name"]
            tool_input = data["input"]
            if name == "web_search":
                console.print(
                    f"  [yellow]🔎 Searching:[/yellow] {tool_input.get('query')!r}"
                )
            elif name == "fetch_page":
                console.print(
                    f"  [yellow]📄 Reading page:[/yellow] {tool_input.get('url')}"
                )
            else:
                console.print(f"  [yellow]🔧 Calling tool:[/yellow] {name}({tool_input})")

        elif event_type == "tool_result":
            # Keep result previews short -- the full text still goes to
            # Claude, we just don't want to flood the terminal with it.
            preview = data["result"].replace("\n", " ")[:100]
            console.print(f"    [dim]-> {preview}...[/dim]")

        elif event_type == "final_report":
            status.update("[bold green]Writing final report...[/bold green]")

        elif event_type == "max_turns_reached":
            console.print(
                "[bold red]Warning:[/bold red] agent hit its turn limit "
                "before finishing."
            )

    return on_event


def main() -> None:
    console = Console()
    args = parse_args()

    console.print(
        Panel.fit(
            "[bold]AI Research Assistant[/bold]\n"
            "Autonomous web research powered by Claude",
            border_style="blue",
        )
    )

    api_key = load_api_key(console)

    # If no topic was passed as a command-line argument, ask for one
    # interactively. rich.console.Console.input() works like the built-in
    # input(), but supports the same markup as console.print().
    topic = args.topic or console.input(
        "[bold]What topic would you like researched?[/bold] "
    )
    topic = topic.strip()
    if not topic:
        console.print("[bold red]No topic provided. Exiting.[/bold red]")
        sys.exit(1)

    console.print(f"\n[bold]Researching:[/bold] {topic}\n")

    # rich's Status context manager shows a spinner + message while work
    # happens, and cleanly disappears once we exit the `with` block.
    with console.status("[bold cyan]Starting up...[/bold cyan]") as status:
        on_event = make_progress_callback(console, status)
        try:
            report_markdown = run_research_agent(topic, api_key, on_event=on_event)
        except Exception as exc:  # noqa: BLE001 - top-level CLI error boundary
            status.stop()
            console.print(f"[bold red]The agent failed unexpectedly:[/bold red] {exc}")
            sys.exit(1)

    console.print()  # blank line for spacing
    console.print(Panel.fit("[bold green]Research complete![/bold green]"))

    # Render the Markdown report nicely in the terminal...
    console.print(Markdown(report_markdown))

    # ...and also save it to disk so it isn't lost when the terminal closes.
    saved_path = save_report(topic, report_markdown)
    console.print(f"\n[bold]Report saved to:[/bold] [cyan]{saved_path}[/cyan]")


if __name__ == "__main__":
    main()
