"""End-to-end: boot the real server and talk to it as an MCP client over Streamable HTTP."""

import asyncio
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
from mcp.client import Client

EXPECTED_TOOLS = {
    "log_journal_entry", "get_commitments", "prioritize_today",
    "complete_commitment", "add_commitment", "get_reflection",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url(tmp_path_factory):
    port = _free_port()
    env = {
        **os.environ,
        "HEARTH_PORT": str(port),
        "HEARTH_LLM": "heuristic",
        "HEARTH_NOW": "2026-10-20T20:30:00",
        "HEARTH_DB": str(tmp_path_factory.mktemp("db") / "t.db"),
    }
    proc = subprocess.Popen([sys.executable, "-m", "hearth.server"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5)
                break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError("server did not start")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_full_flow_over_streamable_http(server_url):
    async def run():
        async with Client(server_url) as c:
            # Required by the hackathon: spec 2025-11-25 or later.
            assert c.protocol_version >= "2025-11-25", c.protocol_version
            tools = {t.name for t in (await c.list_tools()).tools}
            assert tools == EXPECTED_TOOLS

            r = await c.call_tool("log_journal_entry", {"text": "Rough day. I need to call the landlord tomorrow and I have to finish my thesis draft by friday."})
            assert not r.is_error and r.structured_content["saved"]
            assert len(r.structured_content["commitments_added"]) == 2

            tomorrow = await c.call_tool("get_commitments", {"timeframe": "tomorrow"})
            assert tomorrow.structured_content["count"] == 1

            top = await c.call_tool("prioritize_today", {})
            assert "landlord" in top.structured_content["speech"].lower()

            done = await c.call_tool("complete_commitment", {"description": "called the landlord"})
            assert done.structured_content["completed"]

            left = await c.call_tool("get_commitments", {"timeframe": "all"})
            assert left.structured_content["count"] == 1

    asyncio.run(run())
