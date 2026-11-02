"""The agent tools as an MCP server, for Claude Desktop, Cursor and other MCP clients. Needs mcp
(pip install "pageindex-search[mcp]").

    pageindex-search-mcp --collection col_...  # over stdio; the key from PAGEINDEX_SEARCH_API_KEY
"""
from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ._tools import Tools, ToolResult


def _mcp() -> Any:
    try:
        import mcp
        import mcp.server.lowlevel  # noqa: F401
        import mcp.types  # noqa: F401
    except ImportError:
        raise ImportError('The MCP server needs mcp: pip install "pageindex-search[mcp]"') from None
    return mcp


def content(result: ToolResult) -> list[Any]:
    """The tool result as MCP content: the text, then each picture after its page number."""
    types = _mcp().types
    out: list[Any] = [types.TextContent(type="text", text=result.text)]
    for image in result.images:
        out.append(types.TextContent(type="text", text=image.label))
        out.append(types.ImageContent(type="image", data=image.base64, mime_type="image/png"))
    return out


def _failed(result: ToolResult) -> bool:
    """Whether the tool gave back {"error": ...}, for the agent to correct."""
    if result.images:
        return False
    try:
        return set(json.loads(result.text)) == {"error"}
    except (ValueError, TypeError):
        return False


def server(tools: Tools | Callable[[Any], Tools], *, name: str = "page-search") -> Any:
    """A low-level MCP Server with the tools; run it over stdio, or serve it over HTTP. tools may be a function
    of the request context giving each request's own, as a server for many callers needs."""
    import anyio
    mcp = _mcp()
    types = mcp.types
    from ._version import __version__

    tools_for = tools if callable(tools) else lambda ctx: tools

    async def list_tools(ctx: Any, params: Any) -> Any:
        return types.ListToolsResult(tools=[
            types.Tool(name=d["name"], description=d["description"], input_schema=d["input_schema"],
                       annotations=types.ToolAnnotations(read_only_hint=True, open_world_hint=False))
            for d in tools_for(ctx).definitions])

    async def call_tool(ctx: Any, params: Any) -> Any:
        result = await anyio.to_thread.run_sync(tools_for(ctx).run, params.name, dict(params.arguments or {}))
        return types.CallToolResult(content=content(result), is_error=_failed(result))

    return mcp.server.lowlevel.Server(name, version=__version__, on_list_tools=list_tools, on_call_tool=call_tool)


def main(argv: list[str] | None = None) -> None:
    """pageindex-search-mcp: serve the tools over stdio."""
    parser = argparse.ArgumentParser(prog="pageindex-search-mcp",
                                     description="Page Search tools as an MCP server, over stdio. The API key comes "
                                                 "from PAGEINDEX_SEARCH_API_KEY.")
    parser.add_argument("--collection", help="Only this collection's documents.")
    parser.add_argument("--filter", type=json.loads, help='Only documents whose metadata meets this, as JSON: '
                                                          '\'{"year": 2024}\'.')
    args = parser.parse_args(argv)

    import anyio
    _mcp()
    from mcp.server.stdio import stdio_server

    from ._client import PageSearch, PageSearchError
    try:
        app = server(PageSearch().tools(collection=args.collection, filter=args.filter))
    except PageSearchError as error:  # a wrong key or collection: say so, where the MCP client logs it
        raise SystemExit(f"pageindex-search-mcp: {error}") from None

    async def serve() -> None:
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())

    anyio.run(serve)
