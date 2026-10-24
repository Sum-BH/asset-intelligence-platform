# %% [markdown]
# # NB22 — MCP Server: Platform Capabilities for Any MCP Client (R30)
# Exposes the three tools (read-only) plus the optimizer over the Model Context
# Protocol, so any MCP-capable client — Claude Desktop, an IDE agent, a colleague's
# assistant — can query the platform. Work-order *creation* stays behind an explicit
# flag: read-only by default is the governance posture.
# Verified here by importing the real `mcp` SDK, building the server, and invoking
# every handler in-process. Conversion target: `aip_agents/mcp_server.py`.

# %%
import asyncio
import json
from pathlib import Path

import mcp.types as types
from mcp.server import Server

DATA = Path("../../data")

# tools + scheduler from earlier notebooks (file handoff)
_t = Path("nb19_tools.py").read_text().split("# ## 5. Fixture tests")[0]
exec("\n".join(ln for ln in _t.splitlines() if not ln.startswith(("# %%", "# #"))))
_s = Path("nb21_scheduler.py").read_text().split("# ## 2. Assertion suite")[0]
exec("\n".join(ln for ln in _s.splitlines()
                if not ln.startswith(("# %%", "# #", "from __future__"))))

ALLOW_WRITES = False          # explicit flag; see governance note (Phase 5)

# %% [markdown]
# ## 1. Server definition — tool manifest

# %%
server = Server("asset-intelligence-platform")

TOOL_SPECS = [
    types.Tool(
        name="query_signals",
        description="Condition-monitoring state for an asset: health index, trend, "
                    "RUL estimate and the model that produced it.",
        inputSchema={"type": "object",
                     "properties": {"asset_id": {"type": "string"},
                                    "window": {"type": "integer", "default": 30}},
                     "required": ["asset_id"]}),
    types.Tool(
        name="query_vision",
        description="Visual and thermal inspection findings for an asset.",
        inputSchema={"type": "object",
                     "properties": {"asset_id": {"type": "string"}},
                     "required": ["asset_id"]}),
    types.Tool(
        name="query_knowledge",
        description="Search the local maintenance knowledge base; returns cited chunks.",
        inputSchema={"type": "object",
                     "properties": {"query": {"type": "string"},
                                    "k": {"type": "integer", "default": 3}},
                     "required": ["query"]}),
    types.Tool(
        name="optimize_schedule",
        description="Constraint-optimal maintenance schedule for approved work orders.",
        inputSchema={"type": "object",
                     "properties": {"work_orders": {"type": "array", "items": {"type": "object"}},
                                    "technicians": {"type": "array", "items": {"type": "object"}}},
                     "required": ["work_orders", "technicians"]}),
]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return TOOL_SPECS


async def _dispatch(name: str, args: dict) -> dict:
    if name == "query_signals":
        return query_signals(args["asset_id"], args.get("window", 30)).model_dump()
    if name == "query_vision":
        return query_vision(args["asset_id"]).model_dump()
    if name == "query_knowledge":
        return query_knowledge(args["query"], args.get("k", 3)).model_dump()
    if name == "optimize_schedule":
        wos = [WorkOrder(**w) for w in args["work_orders"]]
        techs = [Technician(t["id"], frozenset(t["skills"]), tuple(t["shift"]))
                 for t in args["technicians"]]
        sch = optimize(wos, techs)
        return {"status": sch.status, "assignments": sch.assignments,
                "unscheduled": sch.unscheduled, "objective": sch.objective}
    if name.startswith("create_") and not ALLOW_WRITES:
        raise PermissionError("write tools disabled (ALLOW_WRITES=False)")
    raise ValueError(f"unknown tool: {name}")


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    result = await _dispatch(name, arguments or {})
    return [types.TextContent(type="text", text=json.dumps(result, indent=1))]

# %% [markdown]
# ## 2. In-process verification — manifest + every handler actually invoked

# %%
async def verify():
    tools = await list_tools()
    names = [t.name for t in tools]
    print("advertised tools:", names)
    assert names == ["query_signals", "query_vision", "query_knowledge", "optimize_schedule"]
    for t in tools:
        assert t.description and t.inputSchema["type"] == "object"

    sig = await _dispatch("query_signals", {"asset_id": "press-1"})
    vis = await _dispatch("query_vision", {"asset_id": "press-1"})
    kno = await _dispatch("query_knowledge", {"query": "bearing envelope BPFO", "k": 2})
    sch = await _dispatch("optimize_schedule", {
        "work_orders": [{"id": "W1", "asset_id": "pump-3", "duration_h": 3, "priority": 3,
                         "earliest_h": 0, "deadline_h": 24, "required_skill": "hydraulics"}],
        "technicians": [{"id": "T1", "skills": ["hydraulics"], "shift": [0, 24]}]})
    print("signals  :", sig["health_index"], sig["trend"])
    print("vision   :", vis["n_findings"], "findings")
    print("knowledge:", [h["chunk_id"] for h in kno["hits"]])
    print("schedule :", sch["status"], sch["assignments"])

    assert sig["asset_id"] == "press-1" and sig["rul_model"]
    assert vis["n_findings"] > 0 and kno["hits"]
    assert sch["status"] in ("OPTIMAL", "FEASIBLE") and sch["assignments"]

    # content wrapper is what a real client receives
    payload = await call_tool("query_signals", {"asset_id": "press-1"})
    assert payload[0].type == "text" and json.loads(payload[0].text)["asset_id"] == "press-1"

    try:
        await _dispatch("create_work_order", {})
        raise SystemExit("write tool should have been refused")
    except PermissionError as e:
        print("write guard OK:", e)
    return {"tools": names, "signals": sig, "schedule_status": sch["status"]}

def _run(coro):
    """Jupyter kernels already own an event loop; fall back to a worker thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()

result = _run(verify())

# %% [markdown]
# ## 3. Client configuration (workstation)

# %%
STDIO_ENTRYPOINT = '''\
# aip_agents/mcp_server.py — run with: uv run python -m aip_agents.mcp_server
import asyncio
from mcp.server.stdio import stdio_server
async def main():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())
asyncio.run(main())
'''
CLIENT_CONFIG = {"mcpServers": {"asset-intelligence": {
    "command": "uv",
    "args": ["run", "python", "-m", "aip_agents.mcp_server"],
    "cwd": "/path/to/asset-intelligence-platform"}}}
print(json.dumps(CLIENT_CONFIG, indent=2))

json.dump({"tools": result["tools"], "verified_in_process": True,
           "write_tools_enabled": ALLOW_WRITES, "client_config": CLIENT_CONFIG},
          open(DATA / "processed/nb22_mcp.json", "w"), indent=1)
print("NB22 complete.")
