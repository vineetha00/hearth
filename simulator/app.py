"""Simulated Alexa+ web app.

Plays the part of Alexa+: takes what you say, picks a Hearth tool, calls the *real* Hearth MCP server over
Streamable HTTP with an MCP client, and speaks the result back. Every call is returned in a trace so the demo
shows exactly what crossed the wire.

Run:  python -m simulator.app        # http://127.0.0.1:8080 (needs the Hearth server on :8765)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import uvicorn
from mcp.client import Client
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from datetime import date, timedelta

from hearth import dates

from . import router

log = logging.getLogger("hearth.simulator")


def _group(c: dict[str, Any]) -> str:
    """Bucket a commitment for display (uses the same clock as the Hearth server)."""
    if not c.get("due_date"):
        return "none"
    delta = (date.fromisoformat(c["due_date"]) - dates.today()).days
    if delta < 0:
        return "overdue"
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    return "week" if delta <= 6 else "later"


def _decorate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**c, "group": _group(c)} for c in items]


def _today_label() -> str:
    return dates.today().strftime("%A, %B %-d")
STATIC = Path(__file__).parent / "static"
MCP_URL = os.environ.get("HEARTH_MCP_URL", "http://127.0.0.1:8765/mcp")
MODEL_ID = os.environ.get("HEARTH_ROUTER_MODEL", os.environ.get("HEARTH_BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"))

_bedrock = router.make_bedrock_client()


async def _call(client: Client, tool: str, arguments: dict[str, Any], trace: list[dict]) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = await client.call_tool(tool, arguments)
    ms = round((time.perf_counter() - t0) * 1000)
    data = result.structured_content or {}
    trace.append({"tool": tool, "arguments": arguments, "ms": ms, "is_error": bool(result.is_error), "result": data})
    if result.is_error:
        raise RuntimeError(f"{tool} returned an error")
    return data


async def _tools(client: Client) -> list[dict[str, Any]]:
    listed = await client.list_tools()
    return [{"name": t.name, "description": t.description or "", "inputSchema": t.input_schema} for t in listed.tools]


async def say(request: Request) -> JSONResponse:
    body = await request.json()
    utterance = (body.get("utterance") or "").strip()
    if not utterance:
        return JSONResponse({"error": "Say something first."}, status_code=400)
    trace: list[dict] = []
    try:
        async with Client(MCP_URL) as client:
            plan = router.route_rules(utterance)
            if _bedrock is not None:
                try:
                    plan = router.route_bedrock(utterance, await _tools(client), _bedrock, MODEL_ID)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Bedrock routing failed (%s); using rules", exc)
            data = await _call(client, plan.tool, plan.arguments, trace)
            if plan.fallback_to_journal and not data.get("completed") and not data.get("needs_clarification"):
                data = await _call(client, "log_journal_entry", {"text": utterance}, trace)
            listing = await _call(client, "get_commitments", {"timeframe": "all"}, [])
            return JSONResponse({
                "speech": data.get("speech", ""),
                "tool": trace[-1]["tool"],
                "router": plan.router,
                "engine": data.get("engine"),
                "protocol": client.protocol_version,
                "trace": trace,
                "commitments": _decorate(listing.get("commitments", [])),
                "today": _today_label(),
            })
    except Exception as exc:  # noqa: BLE001
        log.exception("say failed")
        return JSONResponse(
            {"error": f"Couldn't reach the Hearth MCP server at {MCP_URL}. Start it with: python -m hearth.server ({exc})"},
            status_code=502,
        )


async def state(_: Request) -> JSONResponse:
    try:
        async with Client(MCP_URL) as client:
            listing = await _call(client, "get_commitments", {"timeframe": "all"}, [])
            tools = await _tools(client)
            return JSONResponse({
                "commitments": _decorate(listing.get("commitments", [])),
                "today": _today_label(),
                "tools": [t["name"] for t in tools],
                "protocol": client.protocol_version,
                "router": "bedrock" if _bedrock is not None else "rules",
                "mcp_url": MCP_URL,
            })
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"Hearth MCP server not reachable at {MCP_URL} ({exc})"}, status_code=502)


async def index(_: Request) -> FileResponse:
    return FileResponse(STATIC / "index.html")


app = Starlette(routes=[
    Route("/", index),
    Route("/api/say", say, methods=["POST"]),
    Route("/api/state", state),
])


def main() -> None:
    uvicorn.run(app, host=os.environ.get("SIM_HOST", "127.0.0.1"), port=int(os.environ.get("SIM_PORT", "8080")), log_level="info")


if __name__ == "__main__":
    main()
