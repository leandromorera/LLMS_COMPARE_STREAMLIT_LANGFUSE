"""
Thin Python client for the Langfuse MCP server at http://localhost:8005/mcp/.

The MCP server handles Langfuse authentication internally — no LANGFUSE_* keys
needed in .env. Every public method is a direct wrapper over a named MCP tool.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

MCP_URL = "http://localhost:8005/mcp/"
_CONTENT_TYPE_JSON = "application/json"


class LangfuseMCPClient:
    """HTTP client for the Langfuse MCP server (Streamable HTTP / SSE transport)."""

    def __init__(self, url: str = MCP_URL, timeout: int = 60) -> None:
        self.url = url
        self.timeout = timeout
        self._session_id: str | None = None

    # ------------------------------------------------------------------
    # Low-level plumbing
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Perform the MCP initialize handshake and capture the session ID."""
        payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "testthekey-client", "version": "1.0"},
            },
        }
        headers = {
            "Content-Type": _CONTENT_TYPE_JSON,
            "Accept": f"{_CONTENT_TYPE_JSON}, text/event-stream",
        }
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.post(self.url, json=payload, headers=headers)
        if "mcp-session-id" in resp.headers:
            self._session_id = resp.headers["mcp-session-id"]

    def _call(self, method: str, params: dict) -> Any:
        """Send a JSON-RPC request and return the parsed result."""
        if not self._session_id:
            self._initialize()

        headers = {
            "Content-Type": _CONTENT_TYPE_JSON,
            "Accept": f"{_CONTENT_TYPE_JSON}, text/event-stream",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.post(self.url, json=payload, headers=headers)

        if "mcp-session-id" in resp.headers:
            self._session_id = resp.headers["mcp-session-id"]

        content_type = resp.headers.get("content-type", "")

        # ── Plain JSON response ───────────────────────────────────────────────
        if _CONTENT_TYPE_JSON in content_type:
            try:
                data = resp.json()
                if "result" in data:
                    return data["result"]
                if "error" in data:
                    raise RuntimeError(f"MCP error: {data['error']}")
            except Exception as exc:
                raise RuntimeError(f"Could not parse MCP JSON response: {exc}\n{resp.text[:400]}")

        # ── SSE / text event-stream response ─────────────────────────────────
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "result" in data:
                        return data["result"]
                    if "error" in data:
                        raise RuntimeError(f"MCP error: {data['error']}")
                except json.JSONDecodeError:
                    continue

        # ── Last resort: try the whole body as JSON ───────────────────────────
        try:
            data = json.loads(resp.text)
            if "result" in data:
                return data["result"]
            if "error" in data:
                raise RuntimeError(f"MCP error: {data['error']}")
        except Exception:
            pass

        raise RuntimeError(f"Could not parse MCP response:\n{resp.text[:400]}")

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call an MCP tool and return its text result string."""
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        if isinstance(result, dict):
            content = result.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", json.dumps(result))
            return json.dumps(result)
        return str(result)

    def _parse_result(self, raw: str) -> dict:
        """Parse the JSON string returned by MCP tools."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    # ------------------------------------------------------------------
    # Auth / connectivity
    # ------------------------------------------------------------------

    def auth_check(self, verbose: bool = False) -> dict:
        raw = self.call_tool("langfuse_auth_check", {"verbose": verbose})
        return self._parse_result(raw)

    # ------------------------------------------------------------------
    # Trace / ID helpers
    # ------------------------------------------------------------------

    def create_trace_id(self, seed: str | None = None) -> str:
        """Return a deterministic trace UUID (hex string) for a given seed."""
        raw = self.call_tool("langfuse_create_trace_id", {"seed": seed})
        data = self._parse_result(raw)
        return data.get("trace_id", raw)

    def get_trace(self, trace_id: str) -> dict:
        raw = self.call_tool("langfuse_get_trace", {"trace_id": trace_id})
        return self._parse_result(raw)

    def list_traces(self, name: str | None = None, limit: int = 10) -> dict:
        raw = self.call_tool("langfuse_list_traces", {"name": name, "limit": limit})
        return self._parse_result(raw)

    # ------------------------------------------------------------------
    # Observation logging
    # ------------------------------------------------------------------

    def log_observation(
        self,
        name: str,
        as_type: str = "span",
        trace: dict | None = None,
        trace_context: dict | None = None,
        observation: dict | None = None,
    ) -> dict:
        """Log a trace + one observation (span / generation / event / etc)."""
        raw = self.call_tool("langfuse_log_observation", {
            "name": name,
            "as_type": as_type,
            "trace": trace,
            "trace_context": trace_context,
            "observation": observation or {},
            "flush": True,
            "return_trace_url": True,
        })
        return self._parse_result(raw)

    def log_generation(
        self,
        observation: dict,
        trace: dict | None = None,
        latency_ms: int | None = None,
    ) -> dict:
        """Log an LLM generation (prompt → completion) with timing."""
        raw = self.call_tool("langfuse_log_generation", {
            "observation": observation,
            "trace": trace,
            "latency_ms": latency_ms,
            "flush": True,
            "return_trace_url": True,
        })
        return self._parse_result(raw)

    def log_batch(self, records: list[dict]) -> dict:
        """Log multiple observations in a single call."""
        raw = self.call_tool("langfuse_log_batch", {"records": records, "flush": True})
        return self._parse_result(raw)

    def log_error_event(
        self,
        message: str,
        trace: dict | None = None,
        error_type: str | None = None,
        data: dict | None = None,
    ) -> dict:
        raw = self.call_tool("langfuse_log_error_event", {
            "message": message,
            "trace": trace,
            "error_type": error_type,
            "data": data,
            "flush": True,
            "return_trace_url": True,
        })
        return self._parse_result(raw)

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def create_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Attach a numeric score to a trace."""
        raw = self.call_tool("langfuse_create_score", {
            "trace_id": trace_id,
            "name": name,
            "value": value,
            "comment": comment,
            "metadata": metadata,
            "flush": True,
        })
        return self._parse_result(raw)

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def dataset_create(
        self,
        name: str,
        description: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        raw = self.call_tool("langfuse_dataset_create", {
            "name": name,
            "description": description,
            "metadata": metadata,
        })
        return self._parse_result(raw)

    def dataset_add_item(
        self,
        dataset_name: str,
        input: dict,
        expected_output: dict | None = None,
        item_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        raw = self.call_tool("langfuse_dataset_add_item", {
            "dataset_name": dataset_name,
            "input": input,
            "expected_output": expected_output,
            "item_id": item_id,
            "metadata": metadata,
        })
        return self._parse_result(raw)

    def dataset_run_log(
        self,
        run_name: str,
        dataset_item_id: str,
        trace_id: str | None = None,
        observation_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        raw = self.call_tool("langfuse_dataset_run_log", {
            "run_name": run_name,
            "dataset_item_id": dataset_item_id,
            "trace_id": trace_id,
            "observation_id": observation_id,
            "metadata": metadata,
        })
        return self._parse_result(raw)

    # ------------------------------------------------------------------
    # Prompt management
    # ------------------------------------------------------------------

    def prompt_create_version(
        self,
        name: str,
        prompt: str,
        labels: list[str] | None = None,
        tags: list[str] | None = None,
        commit_message: str | None = None,
    ) -> dict:
        raw = self.call_tool("langfuse_prompt_create_version", {
            "name": name,
            "prompt": prompt,
            "type": "text",
            "labels": labels,
            "tags": tags,
            "commit_message": commit_message,
        })
        return self._parse_result(raw)

    def prompt_get(self, prompt_name: str, label: str | None = None) -> dict:
        raw = self.call_tool("langfuse_prompt_get", {
            "prompt_name": prompt_name,
            "label": label,
        })
        return self._parse_result(raw)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self) -> None:
        self.call_tool("langfuse_flush", {})
