"""
agent.py
========

This file contains the "brain" of the AI Research Assistant: the agentic
loop that talks to Claude, lets it call tools (see tools.py), feeds the
results back, and keeps going until Claude decides it has done enough
research and writes a final Markdown report.

What is an "agentic loop" (a.k.a. the ReAct pattern)?
-------------------------------------------------------
A single call to an LLM is a one-shot question-and-answer. An *agent* is an
LLM wrapped in a loop that lets it take multiple steps: **Re**ason about
what it needs, **Act** by calling a tool, observe the result, and repeat --
this Reason+Act cycle is exactly what the "ReAct" prompting pattern
describes, and it's what `run_research_agent` below implements:

    1. Send Claude the conversation so far (system prompt + user request +
       any previous tool results).
    2. Claude replies. Its reply is either:
         a) a request to call one or more tools ("tool_use" blocks), or
         b) a final plain-text answer (no more tools needed).
    3. If (a), we actually run those tools in Python (see tools.py),
       package up the results, and add them to the conversation as a new
       message. Then we go back to step 1.
    4. If (b), we're done -- that text *is* the final research report.

This file deliberately does NOT import `rich` or print anything itself.
Keeping the agent logic free of presentation code means it could be reused
in a web app, a Slack bot, tests, etc. Instead, it accepts an optional
callback function so the caller (main.py) can display progress however it
likes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anthropic

from tools import TOOL_DEFINITIONS, call_tool

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# The Claude model this agent uses. Centralizing it here means you only have
# to change it in one place if you want to try a different model.
MODEL_NAME = "claude-sonnet-4-6"

# Safety valve: the maximum number of back-and-forth turns (Claude reply +
# our tool results = 1 turn) before we force the loop to stop. Without this,
# a confused agent could loop forever, burning API credits.
MAX_AGENT_TURNS = 8

# The maximum number of tokens Claude can generate in a single reply. The
# final report can be long, so we give this a generous budget.
MAX_OUTPUT_TOKENS = 4096

# The system prompt is where we tell Claude *who it is* and *how to behave*
# for the entire conversation. This is the single most important piece of
# "prompt engineering" in this project -- it's what turns a generic chatbot
# into a disciplined research assistant that reliably produces a well
# structured report instead of a rambling chat reply.
SYSTEM_PROMPT = """\
You are an autonomous AI research assistant. You will be given a research \
topic. Your job is to investigate it thoroughly using the tools available \
to you, and then produce a single, well-structured Markdown research report.

Follow this process (this is the "ReAct" pattern -- alternate between \
reasoning about what you need and acting by calling a tool):
1. Use the `web_search` tool to find promising sources on the topic. You may \
search more than once with different queries to cover different angles \
(background, recent developments, counterarguments, statistics, etc.).
2. Use the `fetch_page` tool to read the full content of the most promising \
search results. Prioritize primary sources, reputable publications, and \
pages that look substantive rather than thin or promotional.
3. Read enough sources (typically 3-6) to be confident you understand the \
topic from multiple angles before writing anything.
4. Once you have gathered enough information, STOP calling tools and \
instead respond with the final report as plain Markdown text. Do not call \
any more tools once you begin writing the report.

The final report MUST use this Markdown structure:

# <Report Title>

## Executive Summary
A short (3-5 sentence) summary of the topic and the key takeaway.

## Key Findings
A bulleted list of the most important facts or insights you found, each one \
citing the source number it came from, like: "- Some finding [1]."

## Detailed Analysis
Several paragraphs of prose that go deeper into the topic, synthesizing \
information across your sources rather than just listing facts. Use \
subheadings (###) to break up distinct themes if helpful.

## Sources
A numbered list of every source you actually used, formatted as:
1. Page Title - https://the-url-you-fetched

Rules:
- Only cite sources you actually fetched with `fetch_page`; do not cite a \
search result you never opened.
- Be objective and note disagreement between sources if you find any.
- If your searches turn up little useful information, say so honestly in \
the report rather than inventing facts.
"""


# ---------------------------------------------------------------------------
# Event callback type
# ---------------------------------------------------------------------------

# main.py can pass in a function matching this shape to receive progress
# updates (e.g. to print a spinner or log each tool call) without agent.py
# needing to know anything about *how* those updates are displayed.
#
# The callback receives:
#   event_type: a short string like "tool_call", "tool_result", "final_report"
#   data: a dict with details relevant to that event type
EventCallback = Callable[[str, dict[str, Any]], None]


def _noop_callback(event_type: str, data: dict[str, Any]) -> None:
    """Default callback used when the caller doesn't want progress updates."""
    return None


def run_research_agent(
    topic: str,
    api_key: str,
    on_event: EventCallback = _noop_callback,
) -> str:
    """
    Run the full agentic research loop for a given topic and return the
    final Markdown report as a string.

    Args:
        topic: The research topic/question provided by the user.
        api_key: The Anthropic API key to authenticate with.
        on_event: Optional callback invoked with progress events, so the
            caller (e.g. main.py) can display live progress with `rich`.

    Returns:
        The final research report, as a Markdown-formatted string.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # `messages` is the full conversation history we send to Claude on every
    # turn. The Anthropic API is stateless between calls -- it has no memory
    # of previous requests -- so *we* are responsible for building up and
    # re-sending the entire conversation each time, including past tool
    # calls and their results. This growing list is what gives the agent
    # its "memory" across the loop.
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Research the following topic and produce a Markdown "
                f"report as instructed in your system prompt:\n\n{topic}"
            ),
        }
    ]

    for turn in range(1, MAX_AGENT_TURNS + 1):
        on_event("turn_start", {"turn": turn, "max_turns": MAX_AGENT_TURNS})

        # Ask Claude for its next move. `tools=TOOL_DEFINITIONS` is what
        # enables tool use at all -- without it, Claude could never request
        # a tool call, no matter how we phrased the prompt.
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Claude's reply becomes part of the conversation history, exactly
        # as the API returned it (role="assistant", content=list of blocks).
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Claude has decided it's done calling tools. Its text content
            # is (per our system prompt) the final Markdown report.
            report_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            on_event("final_report", {"turn": turn})
            return report_text

        # If we get here, response.content contains one or more
        # "tool_use" blocks (Claude may ask for several tools at once).
        # Anthropic's API requires that every tool_use block be answered
        # with a matching tool_result block, all bundled into a single new
        # "user" message before we can continue the conversation.
        tool_result_blocks = []

        for block in response.content:
            if block.type != "tool_use":
                # Claude can mix explanatory text in with tool_use blocks
                # (e.g. "I'll search for X now."). We don't need to do
                # anything with plain text blocks here, just skip them.
                continue

            on_event(
                "tool_call",
                {"turn": turn, "name": block.name, "input": block.input},
            )

            # This is the actual Python execution step: we take the tool
            # name and arguments Claude requested and run the real function
            # from tools.py. Claude never runs code itself -- it only ever
            # *asks* us to, and we stay in full control of what executes.
            result_text = call_tool(block.name, block.input)

            on_event(
                "tool_result",
                {"turn": turn, "name": block.name, "result": result_text},
            )

            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    # tool_use_id links this result back to the specific
                    # tool_use request it's answering -- required by the API.
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )

        # Send all of this turn's tool results back as a new user message,
        # then loop around so Claude can decide what to do next.
        messages.append({"role": "user", "content": tool_result_blocks})

    # If we exit the loop without Claude producing a final answer, we've
    # hit MAX_AGENT_TURNS. Rather than silently failing, return whatever
    # useful text we can plus a clear note about what happened.
    on_event("max_turns_reached", {"max_turns": MAX_AGENT_TURNS})
    return (
        f"# Research incomplete\n\n"
        f"The agent used all {MAX_AGENT_TURNS} allotted turns without "
        f"producing a final report on:\n\n> {topic}\n\n"
        f"Try increasing MAX_AGENT_TURNS in agent.py, or narrowing the topic."
    )
