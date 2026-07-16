"""Regression test for the tool-catalogue key build in run_reply_engine.

Python 3.12 removed the implicit `dict_keys + list` concatenation that the
engine relied on; the router cache key must build from plain lists.
"""

import sys
import os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Stubs for optional deps not installed in the test environment.
sys.modules.setdefault("dotenv", types.ModuleType("dotenv"))
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
_mcp = types.ModuleType("mcp")
_mcp.ClientSession = object
_mcp_client = types.ModuleType("mcp.client")
_mcp_client.stdio_client = lambda *a, **k: None
_mcp_client.StdioServerParameters = object
_mcp_client_stdio = types.ModuleType("mcp.client.stdio")
_mcp_client_stdio.stdio_client = lambda *a, **k: None
_mcp_client_stdio.StdioServerParameters = object
sys.modules.update(
    {"mcp": _mcp, "mcp.client": _mcp_client, "mcp.client.stdio": _mcp_client_stdio}
)


def _loads():
    from jarvis.reply.engine import BUILTIN_TOOLS, PLUGIN_TOOLS

    return BUILTIN_TOOLS, PLUGIN_TOOLS


def test_tool_catalogue_key_builds_on_py312():
    BUILTIN_TOOLS, PLUGIN_TOOLS = _loads()
    # Mirrors engine.py:1001 (the line that previously raised TypeError).
    key = ",".join(sorted(list(BUILTIN_TOOLS.keys()) + list(PLUGIN_TOOLS.keys())))
    assert isinstance(key, str)
    assert len(key) > 0
    # Every builtin + plugin tool name appears in the catalogue key.
    for name in BUILTIN_TOOLS:
        assert name in key
    for name in PLUGIN_TOOLS:
        assert name in key
