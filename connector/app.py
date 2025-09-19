#!/usr/bin/env python3
"""
HTTP-basierter MCP Server für Minima Connector
Streaming HTTP implementation für direkte ChatGPT Integration
"""

import asyncio
import logging
from typing import Any, Dict, List
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
import mcp.server.stdio
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions
from mcp.types import (
    Tool,
    TextContent,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, Field
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os

# Read VERBOSE env var (default false)
VERBOSE = os.environ.get("VERBOSE", "false").lower() in ("1", "true", "yes")

# FastAPI app für HTTP-based MCP
app = FastAPI(
    title="Minima MCP Server", 
    description="HTTP-based Model Context Protocol server for local document search",
    version="1.0.0"
)

# CORS für ChatGPT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if VERBOSE:
    # Middleware to log incoming request headers and bodies via logger so they
    # appear in Docker/container logs (stdout). This avoids reliance on container
    # filesystem permissions and makes it easy to tail logs from the host.
    @app.middleware("http")
    async def log_requests_middleware(request: Request, call_next):
        try:
            raw = await request.body()
        except Exception:
            raw = b""

        try:
            headers = dict(request.headers)
            logger.info("---- REQUEST START ----")
            logger.info("PATH: %s", request.url.path)
            try:
                logger.info("HEADERS: %s", json.dumps(headers))
            except Exception:
                logger.info("HEADERS: %s", headers)
            # Log a truncated body to avoid huge binary dumps
            logger.info("BODY: %r", raw[:4000])
            logger.info("---- REQUEST END ----")
        except Exception:
            logger.exception("Failed to log incoming request via logger")

        # Recreate the request stream for downstream handlers by setting _receive
        async def receive():
            return {"type": "http.request", "body": raw}

        request._receive = receive  # type: ignore[attr-defined]

        response = await call_next(request)
        return response

# MCP Server instance
server = Server("minima_connector")

# Request models
class SearchRequest(BaseModel):
    query: str = Field(description="Search query for local documents")

class QueryRequest(BaseModel):
    query: str = Field(description="Query text for document search")

class EmbeddingRequest(BaseModel):
    query: str = Field(description="Text to get embedding vector for")

# Indexer base URL (internal Docker network)
INDEXER_BASE_URL = "http://indexer:8000"

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="search_documents",
            description="Search through locally indexed documents (PDF, CSV, DOCX, MD, TXT files)",
            inputSchema=SearchRequest.model_json_schema(),
        ),
        Tool(
            name="query_documents", 
            description="Query the document index and get detailed results with links",
            inputSchema=QueryRequest.model_json_schema(),
        ),
        Tool(
            name="get_embedding",
            description="Get embedding vector for a text query",
            inputSchema=EmbeddingRequest.model_json_schema(),
        ),
        # Add MCP-standard tool names as aliases so ChatGPT connector discovery
        # finds the expected "search" and "fetch" tools. These delegate to
        # our existing handlers above.
        Tool(
            name="search",
            description="(alias) Search through locally indexed documents - returns search results array",
            inputSchema=SearchRequest.model_json_schema(),
        ),
        Tool(
            name="fetch",
            description="(alias) Fetch full document content by id",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    try:
        if name == "search_documents":
            return await handle_search(arguments)
        elif name == "search":
            # alias for MCP compatibility
            return await handle_search(arguments)
        elif name == "query_documents":
            return await handle_query(arguments)
        elif name == "fetch":
            return await handle_fetch(arguments)
        elif name == "get_embedding":
            return await handle_embedding(arguments)
        else:
            raise McpError(INVALID_PARAMS, f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        raise McpError(INTERNAL_ERROR, str(e))

async def handle_search(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle search_documents tool"""
    try:
        request = SearchRequest(**arguments)
    except Exception as e:
        raise McpError(INVALID_PARAMS, f"Invalid arguments: {e}")
    
    url = f"{INDEXER_BASE_URL}/queryid"
    payload = {"query": request.query}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            # Extract result data
            result = data.get("result", {})
            output = result.get("output", "No content found")
            links = result.get("links", [])
            
            # Build a results array according to the MCP docs:
            # {"results": [{"id":"...","title":"...","url":"..."}, ...]}
            if not output or output == "No content found":
                results = []
            else:
                results = []
                # Prefer detailed per-hit results from the indexer (these should include vectordb ids)
                indexer_results = result.get("results") if isinstance(result, dict) else None
                if isinstance(indexer_results, list) and len(indexer_results) > 0:
                    for item in indexer_results:
                        # The indexer should provide a stable vectordb id (point id) in 'id' or similar
                        hit_id = item.get("id") or item.get("point_id") or item.get("hit_id") or ""
                        url = item.get("url", "") or ""
                        # Prefer an explicit text/title from the indexer, else fall back to the snippet
                        text_snippet = item.get("text") or item.get("page_content") or output
                        title = (text_snippet[:120] + '...') if text_snippet and len(text_snippet) > 120 else (text_snippet or "")
                        results.append({"id": str(hit_id), "title": title, "url": url})
                elif isinstance(links, list) and links:
                    # Fallback: use links but don't invent stable ids (use their index)
                    for i, link in enumerate(links, 1):
                        title = (output[:120] + '...') if len(output) > 120 else output
                        results.append({"id": str(i), "title": title, "url": link})
                else:
                    # Last resort: include a single result pointing to content; leave id empty to avoid misleading 'result-1'
                    title = (output[:120] + '...') if len(output) > 120 else output
                    results.append({"id": "", "title": title, "url": ""})

            # MCP requires a content array with one text item whose text is a JSON-encoded string
            payload = {"results": results}
            return [TextContent(type="text", text=json.dumps(payload))]
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during search: {e}")
            raise McpError(INTERNAL_ERROR, f"Search failed: {e}")

async def handle_query(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle query_documents tool"""
    try:
        request = QueryRequest(**arguments)
    except Exception as e:
        raise McpError(INVALID_PARAMS, f"Invalid arguments: {e}")
    
    url = f"{INDEXER_BASE_URL}/queryid"
    payload = {"query": request.query}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            # Extract result data
            result = data.get("result", {})
            output = result.get("output", "No content found")
            links = result.get("links", [])
            
            # Format for MCP
            # For query_documents we return a human readable text block (keeps existing behavior)
            result_text = f"**Query Results for: '{request.query}'**\n\n"
            result_text += f"**Content:**\n{output}\n\n"
            if links:
                result_text += f"**Source Files:**\n"
                for i, link in enumerate(links, 1):
                    result_text += f"{i}. {link}\n"

            return [TextContent(type="text", text=result_text)]
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during query: {e}")
            raise McpError(INTERNAL_ERROR, f"Query failed: {e}")

async def handle_embedding(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle get_embedding tool"""
    try:
        request = EmbeddingRequest(**arguments)
    except Exception as e:
        raise McpError(INVALID_PARAMS, f"Invalid arguments: {e}")
    
    url = f"{INDEXER_BASE_URL}/embedding"
    payload = {"query": request.query}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            # Extract embedding vector
            embedding = data.get("result", [])
            if not embedding:
                return [TextContent(type="text", text="No embedding vector generated")]
            
            # Format embedding info (don't return the full vector as it's very long)
            result_text = f"**Embedding Vector Generated**\n"
            result_text += f"Query: '{request.query}'\n"
            result_text += f"Vector dimensions: {len(embedding)}\n"
            result_text += f"First 5 values: {embedding[:5]}\n"
            result_text += f"Vector type: float32 array suitable for similarity search"
            
            return [TextContent(type="text", text=result_text)]
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during embedding: {e}")
            raise McpError(INTERNAL_ERROR, f"Embedding failed: {e}")


async def handle_fetch(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle fetch tool for MCP: accepts {"id": "..."} and returns the full document object.

    This implementation attempts to call the indexer service to retrieve a document by id.
    If the indexer does not expose a fetch endpoint, we return a minimal document object
    using available fields.
    """
    doc_id = None
    if isinstance(arguments, dict):
        doc_id = arguments.get("id") or arguments.get("document_id")

    if not doc_id:
        raise McpError(INVALID_PARAMS, "fetch requires an 'id' argument")

    # Try a few common indexer endpoints for fetching a document by id
    candidates = [f"{INDEXER_BASE_URL}/document/{doc_id}", f"{INDEXER_BASE_URL}/fetch"]
    async with httpx.AsyncClient() as client:
        for url in candidates:
            try:
                if url.endswith('/fetch'):
                    resp = await client.post(url, json={"id": doc_id}, timeout=20.0)
                else:
                    resp = await client.get(url, timeout=20.0)

                if resp.status_code == 200:
                    data = resp.json()
                    # Try to normalize the returned document shape
                    doc = data.get("result") if isinstance(data, dict) and data.get("result") else data
                    # Ensure required fields
                    doc_obj = {
                        "id": doc.get("id", doc_id) if isinstance(doc, dict) else doc_id,
                        "title": doc.get("title", "") if isinstance(doc, dict) else "",
                        "text": doc.get("text", str(doc)) if isinstance(doc, dict) else str(doc),
                        "url": doc.get("url", "") if isinstance(doc, dict) else "",
                        "metadata": doc.get("metadata", {}) if isinstance(doc, dict) else {},
                    }
                    return [TextContent(type="text", text=json.dumps(doc_obj))]
            except Exception:
                # Try next candidate
                logger.debug("fetch: candidate URL failed: %s", url)

    # Fallback: return a minimal document object indicating the id
    doc_obj = {"id": doc_id, "title": "", "text": "", "url": "", "metadata": {}}
    return [TextContent(type="text", text=json.dumps(doc_obj))]

# HTTP endpoints for MCP over HTTP
@app.get("/")
async def root():
    """Root endpoint - MCP server info"""
    return {
        "name": "minima_connector",
        "version": "1.0.0",
        "protocol": "mcp",
        "description": "Minima Local Document Search MCP Server over HTTP"
    }


@app.post("/")
async def root_post(request: Request):
    """Accept POSTs to root (some clients POST to base URL during plugin registration).

    Log the body and return the same JSON as GET / to avoid 405 responses.
    """
    try:
        raw = await request.body()
        if raw:
            logger.info(f"POST / received with body: {raw[:2000]!r}")
        else:
            logger.info("POST / received with empty body")
    except Exception:
        logger.exception("Failed to read POST / body")

    # Some clients (observed: openai-mcp) send a JSON-RPC initialize message.
    # Attempt to parse JSON and respond with a minimal JSON-RPC success response
    try:
        payload = json.loads(raw.decode('utf-8')) if raw else {}
    except Exception:
        payload = None

    if isinstance(payload, dict) and payload.get("method") in ("initialize", "initialized"):
        # Build a richer capabilities object so clients (openai-mcp) can discover
        # the server's MCP endpoints and available tools. This mirrors the
        # manifest's mcp section and includes a simple tools list.
        try:
            public_base = "https://bastapiprompting.ils-geomonitoring.de"
            req_params = payload.get("params", {}) if isinstance(payload, dict) else {}

            # Attach full tool definitions so the Connectors UI can list tools
            tools = [t.model_dump() for t in await list_tools()]

            resp = {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "result": {
                    "capabilities": {
                        # Provide both a mapping (name->tool) and a tools array
                        # Some clients expect a mapping, others expect an array.
                        "tools": {t["name"]: t for t in tools},
                        "tools_list": tools,
                        "resources": {},
                        "prompts": {},
                        "logging": {}
                    },
                    "protocolVersion": req_params.get("protocolVersion", "2024-11-05"),
                    "serverInfo": {
                        "name": "minima_connector",
                        "version": "1.0.0"
                    },
                    # Helpful URIs for clients that expect them inline
                    "mcp": {
                        "server_url": public_base,
                        "tools_url": f"{public_base}/tools",
                        "protocol_url": f"{public_base}/mcp",
                        "events_url": f"{public_base}/events",
                    }
                },
            }

            # Log the response we will send (truncated)
            try:
                logger.info("JSON-RPC initialize response: %s", json.dumps(resp)[:2000])
            except Exception:
                logger.exception("Failed to log JSON-RPC initialize response")

            logger.info("Replying to JSON-RPC initialize on POST /")
            return JSONResponse(content=resp)
        except Exception:
            logger.exception("Failed to build JSON-RPC initialize response")
            # Fallback to minimal response
            resp = {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "result": {"capabilities": {"tools": {}, "resources": {}, "prompts": {}, "logging": {}}},
            }
            return JSONResponse(content=resp)

    # Handle JSON-RPC method calls (tools/list, tools/call)
    if isinstance(payload, dict) and payload.get("method"):
        method = payload.get("method")
        req_id = payload.get("id")
        try:
            if method == "tools/list":
                tools = await list_tools()
                result = {"tools": [t.model_dump() for t in tools]}
            elif method == "tools/call":
                params = payload.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                contents = await call_tool(tool_name, arguments)
                result = {"content": [c.model_dump() for c in contents]}
            else:
                # Unknown method -> JSON-RPC error
                err = {"code": -32601, "message": f"Method not found: {method}"}
                return JSONResponse(content={"jsonrpc": "2.0", "id": req_id, "error": err})

            return JSONResponse(content={"jsonrpc": "2.0", "id": req_id, "result": result})
        except McpError as e:
            err = {"code": -32000, "message": str(e)}
            return JSONResponse(content={"jsonrpc": "2.0", "id": req_id, "error": err})
        except Exception as e:
            logger.exception("Exception handling JSON-RPC method on POST /: %s", e)
            err = {"code": -32603, "message": "Internal error"}
            return JSONResponse(content={"jsonrpc": "2.0", "id": req_id, "error": err})

    return await root()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "minima-mcp-server"}

@app.get("/tools")
async def get_tools():
    """Get available tools"""
    tools = await list_tools()
    resp = {"tools": [tool.model_dump() for tool in tools]}
    try:
        logger.info("/tools response: %s", json.dumps(resp)[:2000])
    except Exception:
        logger.exception("Failed to log /tools response")
    return resp

@app.post("/tool/{tool_name}")
async def call_tool_http(tool_name: str, arguments: Dict[str, Any]):
    """Call a specific tool"""
    try:
        result = await call_tool(tool_name, arguments)
        resp = {"result": [content.model_dump() for content in result]}
        try:
            logger.info("/tool/%s response: %s", tool_name, json.dumps(resp)[:2000])
        except Exception:
            logger.exception("Failed to log /tool response")
        return resp
    except McpError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """MCP protocol endpoint over HTTP

    This handler reads the raw body and headers, logs them for debugging, then
    attempts to parse JSON and handle MCP methods. It gracefully handles
    non-JSON payloads and logs full tracebacks on unexpected errors.
    """
    try:
        # Read raw body and headers
        raw = await request.body()
        headers = dict(request.headers)
        try:
            payload = json.loads(raw.decode('utf-8')) if raw else {}
        except Exception:
            payload = None

        logger.info(f"Incoming /mcp request headers: {json.dumps(headers)}")
        logger.info(f"Incoming /mcp raw body: {raw[:2000]!r}")

        if payload is None:
            logger.warning("/mcp request body is not valid JSON")
            raise HTTPException(status_code=400, detail="Request body must be JSON")

        method = payload.get("method")
        params = payload.get("params", {})

        if method == "tools/list":
            tools = await list_tools()
            resp = {"tools": [tool.model_dump() for tool in tools]}
            try:
                logger.info("/mcp tools/list response: %s", json.dumps(resp)[:2000])
            except Exception:
                logger.exception("Failed to log /mcp tools/list response")
            return resp
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = await call_tool(tool_name, arguments)
            resp = {"content": [content.model_dump() for content in result]}
            try:
                logger.info("/mcp tools/call response for %s: %s", tool_name, json.dumps(resp)[:2000])
            except Exception:
                logger.exception("Failed to log /mcp tools/call response")
            return resp
        else:
            logger.warning(f"Unknown MCP method received: {method}")
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

    except HTTPException:
        # Re-raise HTTPException as-is so FastAPI uses its status and detail
        raise
    except Exception as e:
        logger.exception(f"Exception handling /mcp request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Accept requests that include a trailing slash (clients sometimes POST /mcp/)
@app.post("/mcp/")
async def mcp_endpoint_slash(request: Request):
    return await mcp_endpoint(request)


# Support requests when the user mistakenly set the server base URL to include a path
# e.g. https://host.tld/mcp -> ChatGPT will request https://host.tld/mcp/.well-known/ai-plugin.json
@app.get("/{prefix:path}/.well-known/ai-plugin.json", include_in_schema=False)
async def mcp_manifest_with_prefix(prefix: str):
    return await mcp_manifest()

@app.get("/{prefix:path}/.well-known/openapi.json", include_in_schema=False)
async def openapi_schema_with_prefix(prefix: str):
    return await openapi_schema()

@app.get("/{prefix:path}/tools")
async def get_tools_with_prefix(prefix: str):
    return await get_tools()

@app.post("/{prefix:path}/tool/{tool_name}")
async def call_tool_http_with_prefix(prefix: str, tool_name: str, arguments: Dict[str, Any]):
    return await call_tool_http(tool_name, arguments)

@app.get("/{prefix:path}/events")
async def stream_events_with_prefix(prefix: str):
    return await stream_events()

@app.post("/{prefix:path}/mcp")
async def mcp_endpoint_with_prefix(prefix: str, request: Request):
    return await mcp_endpoint(request)

# Server-Sent Events endpoint for streaming
@app.get("/events")
async def stream_events():
    """SSE endpoint for real-time communication"""
    async def event_generator():
        yield f"data: {json.dumps({'type': 'connected', 'server': 'minima_connector'})}\n\n"
        # Keep connection alive
        while True:
            await asyncio.sleep(30)
            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# MCP Manifest für ChatGPT Integration
@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
async def mcp_manifest() -> JSONResponse:
    """MCP Manifest für ChatGPT Integration"""
    public_base = "https://bastapiprompting.ils-geomonitoring.de"
    
    manifest = {
        "schema_version": "v1",
        "name_for_human": "Minima Local Search",
        "name_for_model": "minima_local_search",
        "description_for_human": "Search through locally indexed documents using Minima's intelligent search engine.",
        "description_for_model": "Search and retrieve relevant content from locally indexed PDF, CSV, DOCX, MD, and TXT files. Returns snippets and file locations matching the search query. Uses Model Context Protocol (MCP) over HTTP.",
        "auth": {
            "type": "none"
        },
        "api": {
            "type": "openapi",
            "url": f"{public_base}/.well-known/openapi.json",
            "is_user_authenticated": False,
        },
        "mcp": {
            "server_url": f"{public_base}",
            "tools_url": f"{public_base}/tools",
            "protocol_url": f"{public_base}/mcp",
            "events_url": f"{public_base}/events",
            "transport": "http",
            "description": "HTTP-based Model Context Protocol server for local document search"
        },
        "logo_url": f"{public_base}/logo.svg",
        "contact_email": "support@ils-geomonitoring.de",
        "legal_info_url": f"{public_base}/legal",
    }
    return JSONResponse(content=manifest)

# OpenAPI Schema für MCP
@app.get("/.well-known/openapi.json", include_in_schema=False)
async def openapi_schema() -> Dict[str, Any]:
    """OpenAPI Schema für MCP Tools"""
    public_base = "https://bastapiprompting.ils-geomonitoring.de"
    
    schema = {
        "openapi": "3.0.0",
        "info": {
            "title": "Minima MCP Server",
            "description": "HTTP-based Model Context Protocol server for local document search",
            "version": "1.0.0"
        },
        "servers": [
            {
                "url": public_base
            }
        ],
        "paths": {
            "/tools": {
                "get": {
                    "summary": "List available MCP tools",
                    "responses": {
                        "200": {
                            "description": "List of available tools",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "tools": {
                                                "type": "array",
                                                "items": {
                                                    "$ref": "#/components/schemas/Tool"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/tool/{tool_name}": {
                "post": {
                    "summary": "Execute a MCP tool",
                    "parameters": [
                        {
                            "name": "tool_name",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "string"
                            },
                            "description": "Name of the tool to execute"
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "description": "Tool arguments"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Tool execution result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ToolResult"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "Tool": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string"
                        },
                        "description": {
                            "type": "string"
                        },
                        "inputSchema": {
                            "type": "object"
                        }
                    }
                },
                "ToolResult": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string"
                                    },
                                    "text": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return schema

# Logo endpoint
@app.get("/logo.svg", include_in_schema=False)
async def logo() -> PlainTextResponse:
    """Logo für MCP Manifest"""
    logo_svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="100" height="100" rx="20" fill="#2563eb"/>
<text x="50" y="35" font-family="Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle" fill="white">MINIMA</text>
<text x="50" y="55" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="white">LOCAL</text>
<text x="50" y="70" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="white">SEARCH</text>
<circle cx="50" cy="50" r="30" stroke="white" stroke-width="2" fill="none"/>
<path d="M35 50 L45 40 L55 50 L65 40" stroke="white" stroke-width="2" fill="none"/>
</svg>"""
    return PlainTextResponse(logo_svg, media_type="image/svg+xml")

# Legal info endpoint
@app.get("/legal", include_in_schema=False)
async def legal_info() -> JSONResponse:
    """Legal information für MCP Manifest"""
    legal_content = {
        "service_name": "Minima Local Search",
        "provider": "ILS Geomonitoring",
        "description": "This service provides local document search capabilities using Model Context Protocol.",
        "data_usage": "Your search queries are processed locally and not stored permanently.",
        "privacy": "No personal data is transmitted to external services.",
        "contact": "support@ils-geomonitoring.de"
    }
    return JSONResponse(content=legal_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
