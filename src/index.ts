#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";

// Service URLs
const AI_AGENTS_URL = process.env.AI_AGENTS_URL || "http://localhost:8082";
const INFERENCE_SERVICE_URL = process.env.INFERENCE_SERVICE_URL || "http://localhost:8085";
const LLM_GATEWAY_URL = process.env.LLM_GATEWAY_URL || "http://localhost:8080";
const LLM_GATEWAY_DEFAULT_MODEL = process.env.LLM_GATEWAY_DEFAULT_MODEL || "gpt-4o";
const SEMANTIC_SEARCH_URL = process.env.SEMANTIC_SEARCH_URL || "http://localhost:8081";
const CODE_ORCHESTRATOR_URL = process.env.CODE_ORCHESTRATOR_URL || "http://localhost:8083";
const NEO4J_HTTP_URL = process.env.NEO4J_HTTP_URL || "http://localhost:7474";
const NEO4J_USER = process.env.NEO4J_USER || "neo4j";
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || "devpassword";

// HTTP client helper
async function apiCall<T>(
  path: string,
  method: "GET" | "POST" = "GET",
  body?: unknown,
  baseUrl: string = AI_AGENTS_URL,
  timeoutMs: number = 30000
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${baseUrl}${path}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API error (${response.status}): ${error}`);
    }

    return response.json() as Promise<T>;
  } finally {
    clearTimeout(timeout);
  }
}

// Types for AI Agents API responses
interface AgentFunction {
  name: string;
  description?: string;
  input_schema?: Record<string, unknown>;
}

interface Protocol {
  id: string;
  name?: string;
  description?: string;
  inputs?: Record<string, unknown>;
}

interface HealthStatus {
  status: string;
  services?: Record<string, unknown>;
}

// Cache for dynamic tool discovery
let cachedFunctions: AgentFunction[] = [];
let cachedProtocols: Protocol[] = [];
let lastCacheTime = 0;
const CACHE_TTL_MS = 30000; // 30 seconds

async function refreshCache(): Promise<void> {
  const now = Date.now();
  if (now - lastCacheTime < CACHE_TTL_MS) return;

  try {
    const [functionsRes, protocolsRes] = await Promise.all([
      apiCall<{ functions: AgentFunction[] } | AgentFunction[]>("/v1/functions"),
      apiCall<{ protocols: Protocol[] } | Protocol[]>("/v1/protocols"),
    ]);

    cachedFunctions = Array.isArray(functionsRes) 
      ? functionsRes 
      : functionsRes.functions || [];
    cachedProtocols = Array.isArray(protocolsRes) 
      ? protocolsRes 
      : protocolsRes.protocols || [];
    lastCacheTime = now;
    console.error(`Cache refreshed: ${cachedFunctions.length} functions, ${cachedProtocols.length} protocols`);
  } catch (error) {
    console.error("Failed to refresh cache:", error);
    // Keep stale cache on error
  }
}

// Build dynamic tools list
async function buildToolsList(): Promise<Tool[]> {
  await refreshCache();

  const tools: Tool[] = [
    // Core management tools
    {
      name: "ai_agents_health",
      description: "Check health status of AI Agents service and all Kitchen Brigade dependencies",
      inputSchema: {
        type: "object",
        properties: {},
        required: [],
      },
    },
    {
      name: "ai_agents_list_functions",
      description: "List all available agent functions that can be executed",
      inputSchema: {
        type: "object",
        properties: {},
        required: [],
      },
    },
    {
      name: "ai_agents_list_protocols",
      description: "List all available Kitchen Brigade protocols for multi-agent collaboration",
      inputSchema: {
        type: "object",
        properties: {},
        required: [],
      },
    },
    // Generic execution tools
    {
      name: "ai_agents_run_function",
      description:
        "Execute an agent function by name. Use ai_agents_list_functions to see available functions like summarize-content, generate-code, analyze-artifact, etc.",
      inputSchema: {
        type: "object",
        properties: {
          function_name: {
            type: "string",
            description: "Name of the function to execute (e.g., 'summarize-content', 'generate-code', 'extract-structure')",
          },
          input: {
            type: "object",
            description: "Input parameters for the function",
          },
          preset: {
            type: "string",
            description: "Optional preset configuration (e.g., 'S1', 'D4')",
          },
        },
        required: ["function_name", "input"],
      },
    },
    {
      name: "ai_agents_run_protocol",
      description:
        "Execute a Kitchen Brigade protocol for multi-agent collaboration. Protocols include ROUNDTABLE_DISCUSSION, DEBATE_PROTOCOL, WBS_GENERATION, etc.",
      inputSchema: {
        type: "object",
        properties: {
          protocol_id: {
            type: "string",
            description:
              "Protocol ID (e.g., 'ROUNDTABLE_DISCUSSION', 'DEBATE_PROTOCOL', 'WBS_GENERATION', 'ARCHITECTURE_RECONCILIATION')",
          },
          inputs: {
            type: "object",
            description: "Protocol inputs",
            properties: {
              topic: { type: "string", description: "Main discussion topic" },
              context: { type: "string", description: "Background context" },
              documents: {
                type: "array",
                items: { type: "string" },
                description: "Document paths to include",
              },
              constraints: {
                type: "array",
                items: { type: "string" },
                description: "Constraints or guidelines",
              },
            },
          },
          config: {
            type: "object",
            description: "Optional execution configuration",
            properties: {
              max_feedback_loops: { type: "number" },
              allow_feedback: { type: "boolean" },
              run_cross_reference: { type: "boolean" },
            },
          },
          brigade_override: {
            type: "object",
            description: "Optional model overrides for brigade roles (analyst, critic, synthesizer, validator)",
          },
        },
        required: ["protocol_id", "inputs"],
      },
    },
    // LLM Complete with tiered fallback
    {
      name: "llm_complete",
      description:
        "Generate LLM completion with tiered fallback. Tier 1: Local inference-service. Tier 2: Cloud LLM via llm-gateway. Tier 3: Returns work package if all tiers unavailable.",
      inputSchema: {
        type: "object",
        properties: {
          prompt: {
            type: "string",
            description: "The prompt to complete",
          },
          model_preference: {
            type: "string",
            enum: ["auto", "local", "cloud"],
            description: "Model preference - 'auto' tries local first then cloud, 'local' only tries local inference, 'cloud' only tries cloud LLM",
          },
          max_tokens: {
            type: "number",
            description: "Maximum tokens to generate (default: 4096)",
          },
          temperature: {
            type: "number",
            description: "Sampling temperature 0-2 (default: 0.7)",
          },
          system_prompt: {
            type: "string",
            description: "Optional system prompt to set context",
          },
        },
        required: ["prompt"],
      },
    },
    // Direct service tools
    {
      name: "semantic_search",
      description:
        "Search across code, documentation, and textbooks using semantic similarity. Fast RAG queries without going through AI Agents orchestration.",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The search query - can be natural language or code snippet",
          },
          collection: {
            type: "string",
            description: "Collection to search: 'code', 'docs', 'textbooks', or 'all' (default: 'all')",
          },
          top_k: {
            type: "number",
            description: "Number of results to return (default: 10)",
          },
          threshold: {
            type: "number",
            description: "Minimum similarity score 0-1 (default: 0.7)",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "hybrid_search",
      description:
        "Combines semantic search with keyword matching for better precision. Use when you need both conceptual similarity and exact term matches.",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The search query",
          },
          collection: {
            type: "string",
            description: "Collection to search: 'code', 'docs', 'textbooks', or 'all'",
          },
          top_k: {
            type: "number",
            description: "Number of results to return (default: 10)",
          },
          semantic_weight: {
            type: "number",
            description: "Weight for semantic results 0-1 (default: 0.7)",
          },
          keyword_weight: {
            type: "number",
            description: "Weight for keyword results 0-1 (default: 0.3)",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "code_analyze",
      description:
        "Analyze code for patterns, complexity, dependencies, and quality metrics. Direct access to Code Orchestrator service.",
      inputSchema: {
        type: "object",
        properties: {
          code: {
            type: "string",
            description: "The code to analyze (can be a file path or code content)",
          },
          analysis_type: {
            type: "string",
            enum: ["complexity", "dependencies", "patterns", "quality", "security", "all"],
            description: "Type of analysis to perform (default: 'all')",
          },
          language: {
            type: "string",
            description: "Programming language (auto-detected if not specified)",
          },
          context: {
            type: "string",
            description: "Additional context about the codebase",
          },
        },
        required: ["code"],
      },
    },
    {
      name: "graph_query",
      description:
        "Query the Neo4j knowledge graph directly using Cypher. Access relationships between code entities, concepts, and documentation.",
      inputSchema: {
        type: "object",
        properties: {
          cypher: {
            type: "string",
            description: "Cypher query to execute (e.g., 'MATCH (n:Function) RETURN n LIMIT 10')",
          },
          parameters: {
            type: "object",
            description: "Query parameters as key-value pairs",
          },
        },
        required: ["cypher"],
      },
    },
    {
      name: "graph_get_neighbors",
      description:
        "Get all nodes connected to a specific node in the knowledge graph. Useful for exploring relationships.",
      inputSchema: {
        type: "object",
        properties: {
          node_id: {
            type: "string",
            description: "The ID or name of the node to explore",
          },
          node_type: {
            type: "string",
            description: "Type of node: 'Function', 'Class', 'Module', 'Concept', 'Document'",
          },
          relationship_type: {
            type: "string",
            description: "Filter by relationship type: 'CALLS', 'IMPORTS', 'INHERITS', 'REFERENCES', etc.",
          },
          depth: {
            type: "number",
            description: "How many hops to traverse (default: 1)",
          },
        },
        required: ["node_id"],
      },
    },
  ];

  // Add dynamic function-specific tools
  for (const fn of cachedFunctions) {
    const inputSchema = fn.input_schema && typeof fn.input_schema === 'object' && 'type' in fn.input_schema
      ? fn.input_schema as { type: "object"; properties?: Record<string, object>; required?: string[] }
      : {
          type: "object" as const,
          properties: {
            input: {
              type: "object",
              description: "Input parameters for the function",
            },
            preset: {
              type: "string",
              description: "Optional preset configuration",
            },
          },
          required: ["input"],
        };
    tools.push({
      name: `ai_fn_${fn.name.replaceAll("-", "_")}`,
      description: fn.description || `Execute the ${fn.name} agent function`,
      inputSchema,
    });
  }

  // Add dynamic protocol-specific tools
  for (const protocol of cachedProtocols) {
    tools.push({
      name: `ai_protocol_${protocol.id.toLowerCase().replaceAll("-", "_")}`,
      description:
        protocol.description || `Execute the ${protocol.name || protocol.id} Kitchen Brigade protocol`,
      inputSchema: {
        type: "object",
        properties: {
          inputs: {
            type: "object",
            description: "Protocol inputs",
            properties: {
              topic: { type: "string", description: "Main topic for discussion" },
              context: { type: "string", description: "Background context" },
              documents: { type: "array", items: { type: "string" }, description: "Document paths" },
              constraints: { type: "array", items: { type: "string" }, description: "Constraints" },
            },
            required: ["topic"],
          },
          config: { 
            type: "object",
            description: "Execution configuration"
          },
          brigade_override: { 
            type: "object",
            description: "Override brigade role models"
          },
        },
        required: ["inputs"],
      },
    });
  }

  return tools;
}

// Tool handler type
type ToolHandler = (args: Record<string, unknown>) => Promise<unknown>;

// Core tool handlers
async function handleHealth(): Promise<unknown> {
  return apiCall<HealthStatus>("/health");
}

async function handleListFunctions(): Promise<unknown> {
  await refreshCache();
  return {
    functions: cachedFunctions.map((f) => ({
      name: f.name,
      description: f.description,
    })),
    count: cachedFunctions.length,
  };
}

async function handleListProtocols(): Promise<unknown> {
  await refreshCache();
  return {
    protocols: cachedProtocols.map((p) => ({
      id: p.id,
      name: p.name,
      description: p.description,
    })),
    count: cachedProtocols.length,
  };
}

async function handleRunFunction(args: Record<string, unknown>): Promise<unknown> {
  const { function_name, input, preset } = args as {
    function_name: string;
    input: Record<string, unknown>;
    preset?: string;
  };
  return apiCall(`/v1/functions/${function_name}/run`, "POST", { input, preset });
}

async function handleRunProtocol(args: Record<string, unknown>): Promise<unknown> {
  const { protocol_id, inputs, config, brigade_override } = args as {
    protocol_id: string;
    inputs: Record<string, unknown>;
    config?: Record<string, unknown>;
    brigade_override?: Record<string, unknown>;
  };
  return apiCall(`/v1/protocols/${protocol_id}/run`, "POST", { inputs, config, brigade_override });
}

async function handleSemanticSearch(args: Record<string, unknown>): Promise<unknown> {
  const { query, collection = "all", top_k = 10, threshold = 0.7 } = args as {
    query: string;
    collection?: string;
    top_k?: number;
    threshold?: number;
  };
  return apiCall("/v1/search", "POST", { query, collection, top_k, threshold }, SEMANTIC_SEARCH_URL);
}

async function handleHybridSearch(args: Record<string, unknown>): Promise<unknown> {
  const { query, collection = "all", top_k = 10, semantic_weight = 0.7, keyword_weight = 0.3 } = args as {
    query: string;
    collection?: string;
    top_k?: number;
    semantic_weight?: number;
    keyword_weight?: number;
  };
  return apiCall("/v1/hybrid-search", "POST", { query, collection, top_k, semantic_weight, keyword_weight }, SEMANTIC_SEARCH_URL);
}

async function handleCodeAnalyze(args: Record<string, unknown>): Promise<unknown> {
  const { code, analysis_type = "all", language, context } = args as {
    code: string;
    analysis_type?: string;
    language?: string;
    context?: string;
  };
  return apiCall("/v1/analyze", "POST", { code, analysis_type, language, context }, CODE_ORCHESTRATOR_URL);
}

async function handleGraphQuery(args: Record<string, unknown>): Promise<unknown> {
  const { cypher, parameters = {} } = args as {
    cypher: string;
    parameters?: Record<string, unknown>;
  };
  
  const auth = Buffer.from(`${NEO4J_USER}:${NEO4J_PASSWORD}`).toString("base64");
  const response = await fetch(`${NEO4J_HTTP_URL}/db/neo4j/tx/commit`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Basic ${auth}`,
    },
    body: JSON.stringify({ statements: [{ statement: cypher, parameters }] }),
  });

  if (!response.ok) {
    throw new Error(`Neo4j error (${response.status}): ${await response.text()}`);
  }

  const data = await response.json() as {
    results: Array<{ columns: string[]; data: Array<{ row: unknown[] }> }>;
    errors: Array<{ message: string }>;
  };
  
  if (data.errors?.length > 0) {
    throw new Error(`Cypher error: ${data.errors[0].message}`);
  }

  const result = data.results[0];
  if (!result) return { rows: [], columns: [] };
  
  return {
    columns: result.columns,
    rows: result.data.map((d) => {
      const row: Record<string, unknown> = {};
      result.columns.forEach((col, i) => { row[col] = d.row[i]; });
      return row;
    }),
    count: result.data.length,
  };
}

// Static tool handlers map
const TOOL_HANDLERS: Record<string, ToolHandler> = {
  ai_agents_health: handleHealth,
  ai_agents_list_functions: handleListFunctions,
  ai_agents_list_protocols: handleListProtocols,
  ai_agents_run_function: handleRunFunction,
  ai_agents_run_protocol: handleRunProtocol,
  semantic_search: handleSemanticSearch,
  hybrid_search: handleHybridSearch,
  code_analyze: handleCodeAnalyze,
  graph_query: handleGraphQuery,
};

// Dynamic tool dispatch
async function handleDynamicFunction(name: string, args: Record<string, unknown>): Promise<unknown> {
  const fnName = name.replace("ai_fn_", "").replaceAll("_", "-");
  const { input, preset, ...rest } = args as {
    input?: Record<string, unknown>;
    preset?: string;
  };
  return apiCall(`/v1/functions/${fnName}/run`, "POST", { input: input || rest, preset });
}

async function handleDynamicProtocol(name: string, args: Record<string, unknown>): Promise<unknown> {
  const protocolId = name.replace("ai_protocol_", "").toUpperCase().replaceAll("_", "-");
  const { inputs, config, brigade_override } = args as {
    inputs: Record<string, unknown>;
    config?: Record<string, unknown>;
    brigade_override?: Record<string, unknown>;
  };
  return apiCall(`/v1/protocols/${protocolId}/run`, "POST", { inputs, config, brigade_override });
}

// Tool execution handler
async function executeTool(
  name: string,
  args: Record<string, unknown>
): Promise<unknown> {
  // Check static handlers first
  const handler = TOOL_HANDLERS[name];
  if (handler) {
    return handler(args);
  }

  // Dynamic function tools (ai_fn_*)
  if (name.startsWith("ai_fn_")) {
    return handleDynamicFunction(name, args);
  }

  // Dynamic protocol tools (ai_protocol_*)
  if (name.startsWith("ai_protocol_")) {
    return handleDynamicProtocol(name, args);
  }

  // LLM Complete with tiered fallback
  if (name === "llm_complete") {
    return handleLlmComplete(args);
  }

  // Graph Get Neighbors
  if (name === "graph_get_neighbors") {
    return handleGraphGetNeighbors(args);
  }

  throw new Error(`Unknown tool: ${name}`);
}

// LLM Complete handler (separate due to complexity)
async function handleLlmComplete(args: Record<string, unknown>): Promise<unknown> {
  const {
    prompt,
    model_preference = "auto",
    max_tokens = 4096,
    temperature = 0.7,
    system_prompt,
  } = args as {
    prompt: string;
    model_preference?: "auto" | "local" | "cloud";
    max_tokens?: number;
    temperature?: number;
    system_prompt?: string;
  };

  const messages: Array<{ role: string; content: string }> = [];
  if (system_prompt) {
    messages.push({ role: "system", content: system_prompt });
  }
  messages.push({ role: "user", content: prompt });

  // Tier 1: Try local inference-service
  if (model_preference === "auto" || model_preference === "local") {
    const localResult = await tryLocalInference(messages, max_tokens, temperature);
    if (localResult) return localResult;
  }

  // Tier 2: Try cloud via llm-gateway
  if (model_preference === "auto" || model_preference === "cloud") {
    const cloudResult = await tryCloudLlm(messages, max_tokens, temperature);
    if (cloudResult) return cloudResult;
  }

  // Tier 3: Return work package for client to handle
  return {
    tier: "deferred",
    model: null,
    content: null,
    work_package: {
      type: "llm_completion",
      prompt,
      system_prompt,
      max_tokens,
      temperature,
      reason: "All LLM tiers unavailable",
    },
  };
}

async function tryLocalInference(
  messages: Array<{ role: string; content: string }>,
  max_tokens: number,
  temperature: number
): Promise<unknown> {
  try {
    console.error("Trying Tier 1: Local inference-service...");
    const response = await apiCall<{
      model?: string;
      choices: Array<{ message: { content: string } }>;
      usage?: Record<string, number>;
    }>("/v1/chat/completions", "POST", { model: "auto", messages, max_tokens, temperature }, INFERENCE_SERVICE_URL, 30000);
    return {
      tier: "local",
      model: response.model || "local",
      content: response.choices[0].message.content,
      usage: response.usage || {},
    };
  } catch (error) {
    console.error(`Tier 1 (local) failed: ${error}`);
    return null;
  }
}

async function tryCloudLlm(
  messages: Array<{ role: string; content: string }>,
  max_tokens: number,
  temperature: number
): Promise<unknown> {
  try {
    console.error("Trying Tier 2: Cloud LLM via llm-gateway...");
    const response = await apiCall<{
      model?: string;
      choices: Array<{ message: { content: string } }>;
      usage?: Record<string, number>;
    }>("/v1/chat/completions", "POST", { model: LLM_GATEWAY_DEFAULT_MODEL, messages, max_tokens, temperature }, LLM_GATEWAY_URL, 60000);
    return {
      tier: "cloud",
      model: response.model || LLM_GATEWAY_DEFAULT_MODEL,
      content: response.choices[0].message.content,
      usage: response.usage || {},
    };
  } catch (error) {
    console.error(`Tier 2 (cloud) failed: ${error}`);
    return null;
  }
}

// Graph Get Neighbors handler
async function handleGraphGetNeighbors(args: Record<string, unknown>): Promise<unknown> {
  const { node_id, node_type, relationship_type, depth = 1 } = args as {
    node_id: string;
    node_type?: string;
    relationship_type?: string;
    depth?: number;
  };

  const nodeMatch = node_type ? `(n:${node_type} {name: $node_id})` : `(n {name: $node_id})`;
  const relMatch = relationship_type ? `-[r:${relationship_type}*1..${depth}]-` : `-[r*1..${depth}]-`;
  const cypher = `MATCH ${nodeMatch}${relMatch}(m) RETURN DISTINCT n, type(r[0]) as relationship, m LIMIT 50`;

  const auth = Buffer.from(`${NEO4J_USER}:${NEO4J_PASSWORD}`).toString("base64");
  const response = await fetch(`${NEO4J_HTTP_URL}/db/neo4j/tx/commit`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Basic ${auth}`,
    },
    body: JSON.stringify({ statements: [{ statement: cypher, parameters: { node_id } }] }),
  });

  if (!response.ok) {
    throw new Error(`Neo4j error (${response.status}): ${await response.text()}`);
  }

  const data = await response.json() as {
    results: Array<{ columns: string[]; data: Array<{ row: unknown[] }> }>;
    errors: Array<{ message: string }>;
  };
  
  if (data.errors?.length > 0) {
    throw new Error(`Cypher error: ${data.errors[0].message}`);
  }

  const result = data.results[0];
  if (!result) return { source: node_id, neighbors: [] };

  return {
    source: node_id,
    neighbors: result.data.map((d) => ({
      relationship: d.row[1],
      node: d.row[2],
    })),
    count: result.data.length,
  };
}

// Main server setup
const server = new Server(
  {
    name: "ai-agents-mcp-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Register handlers
server.setRequestHandler(ListToolsRequestSchema, async () => {
  const tools = await buildToolsList();
  return { tools };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    const result = await executeTool(name, args || {});
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return {
      content: [
        {
          type: "text",
          text: `Error: ${errorMessage}`,
        },
      ],
      isError: true,
    };
  }
});

// Start server using top-level await
const transport = new StdioServerTransport();
await server.connect(transport);
console.error("AI Agents MCP Server running on stdio");
console.error(`Connecting to AI Agents at: ${AI_AGENTS_URL}`);
