"""
tools.py
========

This file defines the "tools" our AI agent is allowed to use.

In Anthropic's tool-use (a.k.a. "function calling") system, a *tool* is just a
regular Python function, plus a JSON description of what it does and what
arguments it takes. We hand that JSON description to Claude, and when Claude
decides it needs more information than it already has, it responds asking us
to run one of these functions with specific arguments. We run the function
ourselves (Claude never executes code directly) and send the result back.

This file has three parts:
    1. The actual Python functions (web_search, fetch_page) that do the work.
    2. TOOL_DEFINITIONS: the JSON schemas describing those functions to Claude.
    3. A small dispatcher (call_tool) that agent.py uses to run a tool by name.

Keeping tools in their own file (separate from the agent loop in agent.py)
is good practice: it means the "what can the agent do" logic is cleanly
separated from the "how does the agent decide what to do" logic.
"""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# How many search results to return by default. Keeping this small helps
# keep the conversation short (and cheap) for a beginner-friendly demo.
DEFAULT_SEARCH_RESULTS = 5

# We don't want to dump an entire webpage (which could be tens of thousands
# of words) into Claude's context window. This caps how many characters of
# page text we send back per fetch_page call.
MAX_PAGE_CHARS = 6000

# A realistic browser User-Agent so that more sites don't reject our request
# outright as an obvious bot.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

# How long (in seconds) we'll wait for a webpage to respond before giving up.
REQUEST_TIMEOUT_SECONDS = 10


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def web_search(query: str, max_results: int = DEFAULT_SEARCH_RESULTS) -> str:
    """
    Search the web using DuckDuckGo and return a readable summary of results.

    Args:
        query: The search query, e.g. "latest advances in solid-state batteries".
        max_results: How many results to fetch (kept small on purpose).

    Returns:
        A human-readable string listing each result's title, URL, and short
        snippet. We return a *string* (not a Python object) because tool
        results sent back to Claude must be plain text.
    """
    try:
        # DDGS() opens a DuckDuckGo search session. Using it as a context
        # manager ("with") makes sure network resources are cleaned up.
        with DDGS() as ddgs:
            # .text() runs a normal web search (as opposed to .images(),
            # .news(), etc.) and returns a list of dicts.
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001 - report *any* failure back to the agent
        # If DuckDuckGo is unreachable, rate-limits us, etc., we don't want
        # to crash the whole program -- we want Claude to see the error and
        # decide what to do next (e.g. try a different query).
        return f"web_search failed for query '{query}': {exc}"

    if not results:
        return f"No search results found for query: '{query}'"

    # Build a numbered, readable summary. Claude reads plain text, so we
    # format this the way we'd want a human research assistant to hand us
    # a list of leads: title, link, and a one-line description of each.
    lines = [f"Search results for '{query}':"]
    for i, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("href", "")
        snippet = result.get("body", "")
        lines.append(f"{i}. {title}\n   URL: {url}\n   Snippet: {snippet}")

    return "\n".join(lines)


def fetch_page(url: str) -> str:
    """
    Download a webpage and extract its main readable text content.

    Args:
        url: The full URL of the page to fetch, e.g. "https://example.com/article".

    Returns:
        A string containing the cleaned-up text of the page (truncated to
        MAX_PAGE_CHARS characters), or an error message if the fetch failed.
    """
    try:
        response = requests.get(
            url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS
        )
        # raise_for_status() turns HTTP error codes (404, 500, etc.) into
        # a Python exception, which we catch below.
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"fetch_page failed for '{url}': {exc}"

    # BeautifulSoup parses the raw HTML into a tree we can navigate. The
    # "html.parser" backend ships with Python, so no extra system
    # dependencies are required.
    soup = BeautifulSoup(response.text, "html.parser")

    # Strip out elements that are never useful as "article text" -- scripts,
    # stylesheets, navigation bars, etc. Removing them before extracting
    # text prevents things like JavaScript code from polluting our output.
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    # get_text() concatenates all the remaining visible text. `separator="\n"`
    # keeps paragraph-like breaks instead of squashing everything onto one line.
    raw_text = soup.get_text(separator="\n")

    # Collapse the (often huge) amount of blank/whitespace-only lines that
    # HTML tends to produce into a single clean block of text.
    lines = [line.strip() for line in raw_text.splitlines()]
    cleaned_text = "\n".join(line for line in lines if line)

    if not cleaned_text:
        return f"fetch_page retrieved '{url}' but found no readable text content."

    truncated = cleaned_text[:MAX_PAGE_CHARS]
    if len(cleaned_text) > MAX_PAGE_CHARS:
        truncated += "\n... [content truncated]"

    return f"Content fetched from {url}:\n\n{truncated}"


# ---------------------------------------------------------------------------
# Tool schemas (this is what we actually send to the Claude API)
# ---------------------------------------------------------------------------

# Each entry follows Anthropic's tool-use JSON schema format:
#   - "name": must match a key in TOOL_FUNCTIONS below.
#   - "description": Claude reads this to decide *when* to use the tool, so
#     be specific and honest about what it does and doesn't do.
#   - "input_schema": a standard JSON Schema describing the expected arguments.
TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": (
            "Search the public web using DuckDuckGo and return a list of "
            "matching pages (title, URL, and short snippet) for a query. "
            "Use this first to discover candidate sources on a topic before "
            "reading any single page in depth."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run, e.g. 'causes of the 2008 financial crisis'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Number of results to return (default {DEFAULT_SEARCH_RESULTS}).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": (
            "Fetch a specific webpage by URL and return its readable text "
            "content (HTML tags, scripts, and navigation chrome are stripped "
            "out). Use this after web_search to actually read a promising "
            "source in depth before citing it in your report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the page to fetch, including https://.",
                }
            },
            "required": ["url"],
        },
    },
]

# Maps a tool name (as Claude will send it) to the Python function that
# actually implements it. agent.py uses this to dispatch tool calls without
# needing an if/elif chain of tool names.
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "fetch_page": fetch_page,
}


def call_tool(name: str, tool_input: dict) -> str:
    """
    Look up and run a tool by name, given the arguments Claude provided.

    Args:
        name: The tool name Claude asked to call (must be a key in TOOL_FUNCTIONS).
        tool_input: A dict of arguments Claude provided, matching input_schema.

    Returns:
        The tool's string output, or an error message if the tool name is
        unknown or the function raised an unexpected exception.
    """
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"Error: unknown tool '{name}'"

    try:
        # We use **tool_input so that e.g. {"query": "x", "max_results": 3}
        # is passed as web_search(query="x", max_results=3).
        return function(**tool_input)
    except TypeError as exc:
        # This typically means Claude supplied arguments that don't match
        # the function's signature -- surface that clearly instead of
        # crashing the whole agent loop.
        return f"Error calling tool '{name}' with input {tool_input}: {exc}"
