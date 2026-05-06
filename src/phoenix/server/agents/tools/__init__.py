from __future__ import annotations

from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, ExternalToolset

from phoenix.server.agents.dependencies import ChatDependencies
from phoenix.server.agents.tools.external_tools import (
    ASK_USER_TOOL_DEFINITION,
    BASH_TOOL_DEFINITION,
    SET_SPANS_FILTER_TOOL_DEFINITION,
    SET_TIME_RANGE_TOOL_DEFINITION,
)


def build_pxi_toolsets(deps: ChatDependencies) -> AbstractToolset[ChatDependencies]:
    """Build the combined PXI toolset from request dependencies."""
    external_tools: list[ToolDefinition] = [
        BASH_TOOL_DEFINITION,
        ASK_USER_TOOL_DEFINITION,
        SET_TIME_RANGE_TOOL_DEFINITION,
    ]
    project = deps.contexts.project
    if project is not None and project.span_filter is not None:
        external_tools.append(SET_SPANS_FILTER_TOOL_DEFINITION)
    toolsets: list[AbstractToolset[ChatDependencies]] = [ExternalToolset(external_tools)]
    if deps.docs_mcp_toolset is not None:
        toolsets.append(deps.docs_mcp_toolset)
    return CombinedToolset(toolsets)


__all__ = ["build_pxi_toolsets"]
