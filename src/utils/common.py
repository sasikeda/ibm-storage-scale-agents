"""Common utility functions for agent configuration and MCP setup."""

import configparser
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/agent.log",
    log_format: str = "json",
    max_bytes: int = 10485760,
    backup_count: int = 5,
) -> logging.Logger:
    """Setup logging based on configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Format type ('json' or 'text')
        max_bytes: Maximum size of log file before rotation in bytes
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)

    if log_format == "json":
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","message":"%(message)s"}'
        )
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_value)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(log_level_value)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


class MCPClient:
    """Client for interacting with MCP server via HTTP."""

    def __init__(self, base_url: str):
        """Initialize MCP client.

        Args:
            base_url: Base URL of the MCP server
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            timeout=30.0, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        )
        self._tools_cache = None
        self._session_id = None
        self.logger = logging.getLogger(__name__)

    def _parse_sse_response(self, text: str) -> Dict[str, Any]:
        """Parse Server-Sent Events response format.

        Args:
            text: Raw SSE response text

        Returns:
            Parsed JSON data from the SSE message
        """
        import json

        self.logger.debug(f"Parsing SSE response, length: {len(text)}")

        lines = text.strip().split("\n")
        data_lines = []

        for line in lines:
            if line.startswith("data: "):
                json_str = line[6:]
                data_lines.append(json_str)

        if data_lines:
            self.logger.debug(f"Found {len(data_lines)} data lines")
            for idx, data_line in enumerate(data_lines):
                parsed = json.loads(data_line)
                self.logger.debug(
                    f"Data line {idx}: method={parsed.get('method', 'N/A')}, has result={('result' in parsed)}"
                )

            return json.loads(data_lines[-1])

        self.logger.debug("No data lines found, parsing entire response")
        return json.loads(text)

    async def _ensure_session(self):
        """Ensure we have a valid session ID."""
        if self._session_id is not None:
            return

        # Initialize session with the server
        response = await self.client.post(
            f"{self.base_url}",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "scale-agents", "version": "1.0.0"},
                },
            },
        )
        response.raise_for_status()

        # Extract session ID from response headers
        self._session_id = response.headers.get("mcp-session-id")
        if not self._session_id:
            raise Exception("Server did not provide session ID")

        # Update client headers with session ID
        self.client.headers["mcp-session-id"] = self._session_id

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from MCP server.

        Returns:
            List of tool definitions
        """
        if self._tools_cache is not None:
            return self._tools_cache

        await self._ensure_session()

        response = await self.client.post(
            f"{self.base_url}", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        response.raise_for_status()

        # Parse SSE response
        result = self._parse_sse_response(response.text)

        if "result" in result and "tools" in result["result"]:
            self._tools_cache = result["result"]["tools"]
            return self._tools_cache
        return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result (extracted from content)
        """
        await self._ensure_session()

        response = await self.client.post(
            f"{self.base_url}",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        response.raise_for_status()

        result = self._parse_sse_response(response.text)

        if "error" in result:
            raise Exception(f"MCP tool error: {result['error']}")

        if "result" in result:
            mcp_result = result["result"]
            self.logger.debug(f"Raw MCP result keys: {mcp_result.keys()}")

            if "content" in mcp_result and isinstance(mcp_result["content"], list):
                self.logger.debug(f"Found content array with {len(mcp_result['content'])} items")
                for idx, content_item in enumerate(mcp_result["content"]):
                    self.logger.debug(f"Content item {idx}: type={content_item.get('type')}")
                    if content_item.get("type") == "text" and "text" in content_item:
                        import json

                        text_content = content_item["text"]
                        self.logger.debug(f"Text content (first 200 chars): {text_content[:200]}")

                        try:
                            parsed = json.loads(text_content)
                            self.logger.debug("Successfully parsed text content as JSON")
                            return parsed
                        except json.JSONDecodeError:
                            self.logger.debug("Text content is not JSON, returning as plain text")
                            return {"text": text_content}

            if "structuredContent" in mcp_result:
                self.logger.debug("Using structuredContent")
                return mcp_result["structuredContent"]

            self.logger.debug("Fallback: returning entire result")
            return mcp_result

        return result

    async def cleanup(self):
        """Cleanup client resources."""
        await self.client.aclose()


def load_agent_config(config_path: Path) -> configparser.ConfigParser:
    """Load and validate agent configuration from INI file.

    Args:
        config_path: Path to the configuration file

    Returns:
        ConfigParser object with loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required sections or keys are missing
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\nPlease create the file with [llm] and [mcp] sections."
        )

    config = configparser.ConfigParser()
    config.read(config_path)

    # Validate required sections
    required_sections = ["llm", "mcp"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section [{section}] in config file")

    # Validate required keys
    if "model_name" not in config["llm"]:
        raise ValueError("Missing 'model_name' in [llm] section")

    # Validate MCP configuration based on transport
    transport = config["mcp"].get("transport", "http").lower()
    if transport in ("sse", "http") and "url" not in config["mcp"]:
        raise ValueError(f"Missing 'url' in [mcp] section for {transport.upper()} transport")

    return config


def create_mcp_client(mcp_config: configparser.SectionProxy) -> MCPClient:
    """Create an MCP client based on configuration.

    Args:
        mcp_config: MCP section from configuration

    Returns:
        Configured MCPClient instance

    Raises:
        ValueError: If transport type is unsupported
    """
    transport = mcp_config.get("transport", "http").lower()

    if transport in ("sse", "http"):
        # HTTP/SSE transport
        mcp_url = mcp_config["url"].strip('"')  # Remove quotes if present
        return MCPClient(base_url=mcp_url)
    else:
        raise ValueError(
            f"Unsupported transport type: {transport}. Currently only 'http' and 'sse' are supported for LangChain integration."
        )


def create_langchain_tool_with_confirmation_simple(tool_name: str, mcp_client: MCPClient):
    """Create a LangChain tool wrapper with human confirmation requirement.

    This tool will interrupt execution and wait for human approval before proceeding.

    Args:
        tool_name: Name of the MCP tool
        mcp_client: MCP client for execution

    Returns:
        LangChain StructuredTool instance with confirmation requirement
    """
    # Tool-specific descriptions and schemas
    tool_configs = {
        "create_independent_fileset": {
            "description": "Create an INDEPENDENT fileset with its own inode space (can have snapshots)",
            "args": {
                "filesystem": {"type": str, "description": "The filesystem name (e.g., 'fs1')"},
                "fileset_data": {
                    "type": dict,
                    "description": "Fileset configuration data including filesetName, path, etc.",
                },
                "domain": {"type": str, "description": "Domain for authorization", "optional": True},
            },
        },
        "create_dependent_fileset": {
            "description": "Create a DEPENDENT fileset that shares parent's inode space (more efficient, no independent snapshots)",
            "args": {
                "filesystem": {"type": str, "description": "The filesystem name (e.g., 'fs1')"},
                "fileset_data": {
                    "type": dict,
                    "description": "Fileset configuration data including filesetName, path, etc.",
                },
                "domain": {"type": str, "description": "Domain for authorization", "optional": True},
            },
        },
        "delete_fileset": {
            "description": "Delete a fileset from a filesystem",
            "args": {
                "filesystem": {"type": str, "description": "The filesystem name"},
                "fileset_name": {"type": str, "description": "The fileset name to delete"},
            },
        },
        "create_fileset_snapshot": {
            "description": "Create a snapshot for a fileset",
            "args": {
                "filesystem": {"type": str, "description": "The filesystem name"},
                "fileset": {"type": str, "description": "The fileset name"},
                "snapshot_data": {"type": dict, "description": "Snapshot configuration data including snapshotName"},
                "domain": {"type": str, "description": "Domain for authorization", "optional": True},
            },
        },
        "delete_fileset_snapshot": {
            "description": "Delete a fileset snapshot",
            "args": {
                "filesystem": {"type": str, "description": "The filesystem name"},
                "fileset": {"type": str, "description": "The fileset name"},
                "snapshot_name": {"type": str, "description": "The snapshot name to delete"},
                "domain": {"type": str, "description": "Domain for authorization", "optional": True},
            },
        },
    }

    config = tool_configs.get(tool_name, {})
    tool_description = config.get("description", f"Execute {tool_name} operation (requires confirmation)")

    # Dynamically create Pydantic model for the tool's input schema
    fields = {}
    annotations = {}
    for arg_name, arg_config in config.get("args", {}).items():
        arg_type = arg_config["type"]
        arg_desc = arg_config["description"]
        is_optional = arg_config.get("optional", False)

        if is_optional:
            annotations[arg_name] = Optional[arg_type]
            fields[arg_name] = Field(default=None, description=arg_desc)
        else:
            annotations[arg_name] = arg_type
            fields[arg_name] = Field(description=arg_desc)

    # Create dynamic Pydantic model with proper annotations and defaults
    InputModel = type(f"{tool_name}_input", (BaseModel,), {"__annotations__": annotations, **fields})

    logger = logging.getLogger(__name__)

    async def tool_func(**kwargs) -> str:
        """Execute MCP tool with human confirmation requirement."""
        if mcp_client is None:
            return f"Error: MCP client not initialized for tool {tool_name}"

        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        print(f"\n{'=' * 70}")
        print(f"CONFIRMATION REQUIRED: {tool_name}")
        print(f"{'=' * 70}")
        print("Arguments:")
        for key, value in filtered_kwargs.items():
            print(f"  {key}: {value}")
        print(f"{'=' * 70}")

        approval = interrupt(
            {
                "type": "human_confirmation",
                "tool_name": tool_name,
                "arguments": filtered_kwargs,
                "message": f"Do you approve execution of {tool_name}?",
            }
        )

        if approval is None or not approval.get("approved", False):
            return json.dumps({"status": "cancelled", "message": f"Operation {tool_name} cancelled by user"})

        logger.info(f"Approved. Calling {tool_name} with args: {filtered_kwargs}")
        try:
            result = await mcp_client.call_tool(tool_name, filtered_kwargs)
            logger.debug(f"Result from {tool_name} (type: {type(result)})")
            logger.debug(f"Full result: {json.dumps(result, indent=2)}")

            return json.dumps(result, indent=2)
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})

    tool = StructuredTool(name=tool_name, description=tool_description, args_schema=InputModel, coroutine=tool_func)

    return tool


def create_langchain_tool_no_confirmation_simple(tool_name: str, mcp_client: MCPClient):
    """Create a LangChain tool wrapper without fetching tool definition.

    Args:
        tool_name: Name of the MCP tool
        mcp_client: MCP client for execution

    Returns:
        LangChain StructuredTool instance
    """
    # Tool-specific descriptions and schemas
    tool_configs = {
        "list_filesets": {
            "description": "List all filesets in a filesystem",
            "args": {
                "filesystem": {"type": str, "description": "The filesystem name"},
                "domain": {"type": str, "description": "Domain for authorization", "optional": True},
            },
        },
        "link_fileset": {
            "description": "Link a fileset to a junction path",
            "args": {
                "filesystem": {"type": str, "description": "The filesystem name"},
                "fileset_name": {"type": str, "description": "The fileset name to link"},
                "link_data": {"type": dict, "description": "Link configuration data including junction path"},
                "domain": {"type": str, "description": "Domain for authorization", "optional": True},
            },
        },
        "unlink_fileset": {
            "description": "Unlink a fileset from its junction path",
            "args": {
                "filesystem": {"type": str, "description": "The filesystem name"},
                "fileset_name": {"type": str, "description": "The fileset name to unlink"},
                "unlink_data": {"type": dict, "description": "Unlink configuration data", "optional": True},
                "domain": {"type": str, "description": "Domain for authorization", "optional": True},
            },
        },
        "list_fileset_snapshots": {
            "description": "List snapshots for a fileset",
            "args": {
                "filesystem": {"type": str, "description": "The filesystem name"},
                "fileset": {"type": str, "description": "The fileset name"},
                "domain": {"type": str, "description": "Domain for authorization", "optional": True},
            },
        },
    }

    config = tool_configs.get(tool_name, {})
    tool_description = config.get("description", f"Execute {tool_name} operation")

    # Dynamically create Pydantic model for the tool's input schema
    fields = {}
    annotations = {}
    for arg_name, arg_config in config.get("args", {}).items():
        arg_type = arg_config["type"]
        arg_desc = arg_config["description"]
        is_optional = arg_config.get("optional", False)

        if is_optional:
            annotations[arg_name] = Optional[arg_type]
            fields[arg_name] = Field(default=None, description=arg_desc)
        else:
            annotations[arg_name] = arg_type
            fields[arg_name] = Field(description=arg_desc)

    # Create dynamic Pydantic model with proper annotations and defaults
    InputModel = type(f"{tool_name}_input", (BaseModel,), {"__annotations__": annotations, **fields})

    logger = logging.getLogger(__name__)

    async def tool_func(**kwargs) -> str:
        """Execute MCP tool without confirmation."""
        if mcp_client is None:
            return f"Error: MCP client not initialized for tool {tool_name}"

        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        logger.info(f"Calling {tool_name} with args: {filtered_kwargs}")
        try:
            import json

            result = await mcp_client.call_tool(tool_name, filtered_kwargs)
            logger.debug(f"Result from {tool_name} (type: {type(result)})")
            logger.debug(f"Full result: {json.dumps(result, indent=2)}")

            return json.dumps(result, indent=2)
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    tool = StructuredTool(name=tool_name, description=tool_description, args_schema=InputModel, coroutine=tool_func)

    return tool
