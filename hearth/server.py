"""Hearth MCP server (Streamable HTTP) for Alexa+.

Run:  python -m hearth.server          # serves http://127.0.0.1:8765/mcp
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__
from .service import Hearth

INSTRUCTIONS = """\
Hearth is a private end-of-day voice diary for people who live alone and are stretched thin.
Use log_journal_entry whenever the person is simply talking about their day, venting, or reflecting. \
Use get_commitments when they ask what they said they would do, what's due, or what's on their list. \
Use prioritize_today when they ask what matters most or what to do first. \
Use complete_commitment when they say they finished something they had promised or planned. \
Use add_commitment only when they explicitly ask you to remember or remind them of something. \
Use get_reflection when they ask how they've been or what patterns you've noticed.
Hearth only acknowledges and remembers. It never gives advice, therapy, or crisis support. \
Read the 'speech' field of each result back to the person as-is.
"""

mcp = MCPServer("hearth", title="Hearth", instructions=INSTRUCTIONS, version=__version__)
app_logic = Hearth()

UserId = Annotated[str, Field(description="Account identifier. Alexa+ account-linking user id; defaults to a single user.")]


@mcp.tool()
def log_journal_entry(
    text: Annotated[str, Field(description="What the person said about their day, verbatim.")],
    user_id: UserId = "me",
) -> dict[str, Any]:
    """Save a spoken diary entry. Acknowledges it warmly, tags mood and themes, and automatically
    notices anything the person said they will do or promised someone, adding it to their list."""
    return app_logic.log_journal_entry(text, user_id)


@mcp.tool()
def get_commitments(
    timeframe: Annotated[
        str, Field(description="One of: today, tomorrow, this_week, overdue, undated, all.")
    ] = "all",
    user_id: UserId = "me",
) -> dict[str, Any]:
    """List what the person said they would do. 'today' and 'this_week' include overdue items."""
    return app_logic.get_commitments(timeframe, user_id)


@mcp.tool()
def prioritize_today(
    limit: Annotated[int, Field(description="How many items to return (1-5).", ge=1, le=5)] = 3,
    user_id: UserId = "me",
) -> dict[str, Any]:
    """Say which open commitment matters most right now, and why (overdue, due soon, promised to a person)."""
    return app_logic.prioritize_today(limit, user_id)


@mcp.tool()
def complete_commitment(
    description: Annotated[str, Field(description="Spoken description of what they finished, e.g. 'called my sister'.")] = "",
    commitment_id: Annotated[int | None, Field(description="Exact id, if known from a previous result.")] = None,
    user_id: UserId = "me",
) -> dict[str, Any]:
    """Cross something off the list. Matches by meaning; asks which one if it is ambiguous."""
    return app_logic.complete_commitment(description, commitment_id, user_id)


@mcp.tool()
def add_commitment(
    text: Annotated[str, Field(description="The thing to remember, e.g. 'book the dentist'.")],
    due: Annotated[str | None, Field(description="Natural-language date such as 'friday' or 'tomorrow'.")] = None,
    user_id: UserId = "me",
) -> dict[str, Any]:
    """Explicitly add a to-do when the person says 'remind me to...' or 'add ... to my list'."""
    return app_logic.add_commitment(text, due, user_id)


@mcp.tool()
def get_reflection(
    days: Annotated[int, Field(description="How many days back to look (1-60).", ge=1, le=60)] = 7,
    focus: Annotated[
        str | None,
        Field(description="Optional topic to reflect on, e.g. 'work' or 'my family' ('How have I been about work?'). Picks the relevant evenings by meaning."),
    ] = None,
    user_id: UserId = "me",
) -> dict[str, Any]:
    """Notice patterns across recent entries (recurring topics, heavy vs. good evenings, tasks piling up).
    Observations only, never advice."""
    return app_logic.get_reflection(days, user_id, focus)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "name": "hearth", "version": __version__, "llm_engine": app_logic.brain.engine})


def build_app():
    """ASGI app exposing the MCP endpoint at /mcp (Streamable HTTP)."""
    return mcp.streamable_http_app(host=os.environ.get("HEARTH_HOST", "127.0.0.1"))


def main() -> None:
    import uvicorn

    host = os.environ.get("HEARTH_HOST", "127.0.0.1")
    port = int(os.environ.get("HEARTH_PORT", "8765"))
    uvicorn.run(build_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
