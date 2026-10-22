# Friction log

Entries are things that actually happened while building Hearth. Format: task, steps, expected vs. actual, severity, workaround, suggestion.

## 1. MCP Python SDK 2.x: `FastMCP` import is gone

- **Task:** Stand up a Streamable HTTP MCP server using the current official Python SDK.
- **Steps:** `pip install mcp` (resolved to 2.3.0), then `from mcp.server.fastmcp import FastMCP`, the import used by most tutorials and examples.
- **Expected:** The tutorial import works.
- **Actual:** `ModuleNotFoundError`. The message was excellent: it says FastMCP was renamed `MCPServer` (`from mcp.server.mcpserver import MCPServer`), links the migration guide, and offers `mcp<2` as a pin.
- **Severity:** Low (2 minutes), but it will hit everyone following older material.
- **Workaround:** Use `MCPServer`; `@mcp.tool()` and `mcp.streamable_http_app()` work as before.
- **Suggestion:** Keep that error message. Also worth a banner on the older docs and examples pointing at the rename.

## 2. Tools returning a bare `dict` produce no structured content

- **Task:** Return machine-readable results from tools (plus a `speech` string for voice).
- **Steps:** Annotate tools `-> dict`, call from an MCP client, read `result.structured_content`.
- **Expected:** The returned dict arrives as structured content.
- **Actual:** `structured_content` was `None`; the data only appeared as serialized text content. No warning.
- **Severity:** Medium. It looks like it works until a client reads the structured field.
- **Workaround:** Annotate as `dict[str, Any]`, which makes the SDK emit structured output.
- **Suggestion:** Warn at registration time when a tool's return annotation can't produce an output schema, or document the `dict` vs `dict[str, Any]` difference where tools are introduced.

## 3. Localhost binding rejects other `Host` headers (HTTP 421)

- **Task:** Check the server will work behind a public hostname.
- **Steps:** Bound to `127.0.0.1`, sent an `initialize` request with `Host: hearth.example.com`.
- **Expected:** Request accepted, or a clear explanation.
- **Actual:** HTTP 421 with no mention of DNS-rebinding protection in the response I saw. Binding to `0.0.0.0` accepted the same request.
- **Severity:** Medium for anyone deploying; it is a sensible default but an easy way to lose an evening.
- **Workaround:** `HEARTH_HOST=0.0.0.0` in deployment (documented in the README and `.env.example`).
- **Suggestion:** Return a body that names the Host allow-list and how to extend it, and mention it in the deployment docs.

## 4. Bedrock on a new account: every model throttled, and the quota page was the only clue

- **Task:** Call Claude on Amazon Bedrock (Converse API) from a brand-new AWS account for extraction and replies.
- **Steps:** Created an IAM user and access key, set `AWS_REGION`, ran a one-line Converse call against Claude Sonnet 5.5, then Claude Sonnet 4.5, Claude Haiku 4.5 and the Amazon Nova models.
- **Expected:** At least one model responds, or the error says what to enable.
- **Actual:** Sonnet 5.5 returned `AccessDeniedException`. Every other model returned `ThrottlingException` on the very first request, which looked like a rate limit on a call that had never succeeded. Service Quotas showed the *applied* tokens/requests-per-minute quota at **0** against a default of 10,000 requests and 5,000,000 tokens per minute. Raising it to a small number (50) was rejected because the new value must be at least the default.
- **Severity:** High for a first-time user. It cost most of an evening, and nothing in the error said "your quota is zero".
- **Workaround:** Request the quota increase at the default value (10,000 requests / 5,000,000 tokens per minute) for the cross-region Haiku 4.5 profile, then use the `us.` inference profile ID. It was approved the same afternoon. Hearth also falls back to an offline engine per call, so the app kept working while the quota was pending.
- **Suggestion:** Make the throttling error say when the applied quota is zero and link the exact Service Quotas entry. Also add a "check my setup" button or CLI command that lists usable models and their quotas, like the small `scripts/check_bedrock.py` in this repo does.
