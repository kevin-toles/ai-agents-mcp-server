#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
// Service URLs
const AI_AGENTS_URL = process.env.AI_AGENTS_URL || "http://localhost:8082";
const INFERENCE_SERVICE_URL = process.env.INFERENCE_SERVICE_URL || "http://localhost:8085";
const LLM_GATEWAY_URL = process.env.LLM_GATEWAY_URL || "http://localhost:8080";
const LLM_GATEWAY_DEFAULT_MODEL = process.env.LLM_GATEWAY_DEFAULT_MODEL || "gpt-4o";
const SEMANTIC_SEARCH_URL = process.env.SEMANTIC_SEARCH_URL || "http://localhost:8081";
const CODE_ORCHESTRATOR_URL = process.env.CODE_ORCHESTRATOR_URL || "http://localhost:8083";
const AUDIT_SERVICE_URL = process.env.AUDIT_SERVICE_URL || "http://localhost:8084";
const NEO4J_HTTP_URL = process.env.NEO4J_HTTP_URL || "http://localhost:7474";
const NEO4J_USER = process.env.NEO4J_USER || "neo4j";
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || "devpassword";
// HTTP client helper
async function apiCall(path, method = "GET", body, baseUrl = AI_AGENTS_URL, timeoutMs = 30000) {
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
        return response.json();
    }
    finally {
        clearTimeout(timeout);
    }
}
// Cache for dynamic tool discovery
let cachedFunctions = [];
let cachedProtocols = [];
let lastCacheTime = 0;
const CACHE_TTL_MS = 30000; // 30 seconds
async function refreshCache() {
    const now = Date.now();
    if (now - lastCacheTime < CACHE_TTL_MS)
        return;
    try {
        const [functionsRes, protocolsRes] = await Promise.all([
            apiCall("/v1/functions"),
            apiCall("/v1/protocols"),
        ]);
        cachedFunctions = Array.isArray(functionsRes)
            ? functionsRes
            : functionsRes.functions || [];
        cachedProtocols = Array.isArray(protocolsRes)
            ? protocolsRes
            : protocolsRes.protocols || [];
        lastCacheTime = now;
        console.error(`Cache refreshed: ${cachedFunctions.length} functions, ${cachedProtocols.length} protocols`);
    }
    catch (error) {
        console.error("Failed to refresh cache:", error);
        // Keep stale cache on error
    }
}
// Build dynamic tools list
async function buildToolsList() {
    await refreshCache();
    const tools = [
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
            name: "run_agent_function",
            description: "Execute a single-purpose agent function. Use ai_agents_list_functions to see available functions (summarize-content, cross-reference, decompose-task, etc.).",
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
            name: "run_discussion",
            description: "Run multi-LLM discussion using a protocol. Use ai_agents_list_protocols to see available protocols (ROUNDTABLE_DISCUSSION, DEBATE_PROTOCOL, WBS_GENERATION, etc.). Supports LLM selection via brigade_override parameter.",
            inputSchema: {
                type: "object",
                properties: {
                    protocol_id: {
                        type: "string",
                        description: "Protocol ID (e.g., 'ROUNDTABLE_DISCUSSION', 'DEBATE_PROTOCOL', 'WBS_GENERATION', 'ARCHITECTURE_RECONCILIATION', 'RELEVANCE_VALIDATION')",
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
                    relevance_validation: {
                        type: "object",
                        description: "Relevance validation configuration (Stage 0.x). Controls parallel LLM rating and conditional discussion for cross-reference results.",
                        properties: {
                            enabled: {
                                type: "boolean",
                                default: true,
                                description: "Enable relevance validation pipeline",
                            },
                            variance_threshold: {
                                type: "number",
                                default: 2.0,
                                description: "Score variance that triggers discussion (Stage 0.2)",
                            },
                            max_discussion_rounds: {
                                type: "number",
                                default: 3,
                                description: "Maximum rounds for relevance discussion",
                            },
                            pattern_bundle_enabled: {
                                type: "boolean",
                                default: true,
                                description: "Inject engineering patterns into prompts",
                            },
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
            description: "Generate LLM completion with tiered fallback. Tier 1: Local inference-service. Tier 2: Cloud LLM via llm-gateway. Tier 3: Returns work package if all tiers unavailable.",
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
            description: "Search across code, documentation, and textbooks using semantic similarity. Fast RAG queries without going through AI Agents orchestration.",
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
            description: "Combines semantic search with keyword matching for better precision. Use when you need both conceptual similarity and exact term matches.",
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
            description: "Analyze code for patterns, complexity, dependencies, and quality metrics. Direct access to Code Orchestrator service.",
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
            description: "Query the Neo4j knowledge graph directly using Cypher. For advanced users - use run_agent_function with cross-reference for normal searches.",
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
        // =============================================================================
        // WBS-AUD: Audit Tools (audit-service:8084)
        // =============================================================================
        {
            name: "test_compliance_audit",
            description: "Audit tests for anti-patterns (AP1-AP4, U2) per TEST_AUDIT_GUIDELINES.md. Uses CodeBERT cross-reference to validate test quality against coding standards.",
            inputSchema: {
                type: "object",
                properties: {
                    repo_path: {
                        type: "string",
                        description: "Path to repository to audit",
                    },
                    threshold: {
                        type: "number",
                        description: "Similarity threshold for pattern matching (default: 0.7)",
                    },
                    cross_reference: {
                        type: "boolean",
                        description: "Enable CodeBERT cross-reference validation (default: true)",
                    },
                },
                required: ["repo_path"],
            },
        },
        {
            name: "code_pattern_audit",
            description: "Detect code anti-patterns using dual-net detection (regex + semantic). 4-layer pipeline: Detection → Enrichment → Scoring → Reporting. Returns violations with confidence scores and remediation examples.",
            inputSchema: {
                type: "object",
                properties: {
                    code: {
                        type: "string",
                        description: "Source code to analyze",
                    },
                    file_path: {
                        type: "string",
                        description: "Optional file path for language detection",
                    },
                    language: {
                        type: "string",
                        description: "Explicit language override (python, javascript, etc.)",
                    },
                    use_semantic: {
                        type: "boolean",
                        description: "Enable semantic search enrichment from Qdrant (default: true)",
                    },
                    include_remediation: {
                        type: "boolean",
                        description: "Include fix examples from good patterns collection (default: true)",
                    },
                    confidence_threshold: {
                        type: "number",
                        description: "Minimum confidence to report (0-1, default: 0.3)",
                    },
                },
                required: ["code"],
            },
        },
        // NOTE: graph_get_neighbors removed - cross-reference now includes Neo4j via UnifiedRetriever
    ];
    // NOTE: Dynamic ai_fn_* and ai_protocol_* tools removed to reduce clutter.
    // Use run_agent_function and run_discussion instead with function_name/protocol_id parameters.
    return tools;
}
// Core tool handlers
async function handleHealth() {
    return apiCall("/health");
}
async function handleListFunctions() {
    await refreshCache();
    return {
        functions: cachedFunctions.map((f) => ({
            name: f.name,
            description: f.description,
        })),
        count: cachedFunctions.length,
    };
}
async function handleListProtocols() {
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
async function handleRunFunction(args) {
    const { function_name, input, preset } = args;
    return apiCall(`/v1/functions/${function_name}/run`, "POST", { input, preset });
}
async function handleRunProtocol(args) {
    const { protocol_id, inputs, config, brigade_override } = args;
    return apiCall(`/v1/protocols/${protocol_id}/run`, "POST", { inputs, config, brigade_override });
}
async function handleSemanticSearch(args) {
    const { query, collection = "all", top_k = 10, threshold = 0.7 } = args;
    return apiCall("/v1/search", "POST", { query, collection, top_k, threshold }, SEMANTIC_SEARCH_URL);
}
async function handleHybridSearch(args) {
    const { query, collection = "all", top_k = 10, semantic_weight = 0.7, keyword_weight = 0.3 } = args;
    return apiCall("/v1/hybrid-search", "POST", { query, collection, top_k, semantic_weight, keyword_weight }, SEMANTIC_SEARCH_URL);
}
async function handleCodeAnalyze(args) {
    const { code, analysis_type = "all", language, context } = args;
    return apiCall("/v1/analyze", "POST", { code, analysis_type, language, context }, CODE_ORCHESTRATOR_URL);
}
async function handleGraphQuery(args) {
    const { cypher, parameters = {} } = args;
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
    const data = await response.json();
    if (data.errors?.length > 0) {
        throw new Error(`Cypher error: ${data.errors[0].message}`);
    }
    const result = data.results[0];
    if (!result)
        return { rows: [], columns: [] };
    return {
        columns: result.columns,
        rows: result.data.map((d) => {
            const row = {};
            result.columns.forEach((col, i) => { row[col] = d.row[i]; });
            return row;
        }),
        count: result.data.length,
    };
}
// =============================================================================
// WBS-AUD: Audit Tool Handlers (audit-service:8084)
// =============================================================================
async function handleTestComplianceAudit(args) {
    const { repo_path, threshold = 0.7, cross_reference = true } = args;
    console.error(`Calling audit-service cross-reference for: ${repo_path}`);
    return apiCall("/v1/audit/cross-reference", "POST", {
        repo_path,
        threshold,
        cross_reference,
    }, AUDIT_SERVICE_URL, 120000); // 2 minute timeout for large repos
}
async function handleCodePatternAudit(args) {
    const { code, file_path, language, use_semantic = true, include_remediation = true, confidence_threshold = 0.3, } = args;
    console.error(`Calling audit-service pattern detection (semantic=${use_semantic})`);
    return apiCall("/v1/patterns/detect", "POST", {
        code,
        file_path,
        language,
        use_semantic,
        include_remediation,
        confidence_threshold,
        max_results: 100,
    }, AUDIT_SERVICE_URL, 60000); // 1 minute timeout
}
// Static tool handlers map
const TOOL_HANDLERS = {
    ai_agents_health: handleHealth,
    ai_agents_list_functions: handleListFunctions,
    ai_agents_list_protocols: handleListProtocols,
    run_agent_function: handleRunFunction,
    run_discussion: handleRunProtocol,
    semantic_search: handleSemanticSearch,
    hybrid_search: handleHybridSearch,
    code_analyze: handleCodeAnalyze,
    graph_query: handleGraphQuery,
    // WBS-AUD: Audit tools
    test_compliance_audit: handleTestComplianceAudit,
    code_pattern_audit: handleCodePatternAudit,
};
// Tool execution handler
async function executeTool(name, args) {
    // Check static handlers first
    const handler = TOOL_HANDLERS[name];
    if (handler) {
        return handler(args);
    }
    // LLM Complete with tiered fallback
    if (name === "llm_complete") {
        return handleLlmComplete(args);
    }
    throw new Error(`Unknown tool: ${name}`);
}
// LLM Complete handler (separate due to complexity)
async function handleLlmComplete(args) {
    const { prompt, model_preference = "auto", max_tokens = 4096, temperature = 0.7, system_prompt, } = args;
    const messages = [];
    if (system_prompt) {
        messages.push({ role: "system", content: system_prompt });
    }
    messages.push({ role: "user", content: prompt });
    // Tier 1: Try local inference-service
    if (model_preference === "auto" || model_preference === "local") {
        const localResult = await tryLocalInference(messages, max_tokens, temperature);
        if (localResult)
            return localResult;
    }
    // Tier 2: Try cloud via llm-gateway
    if (model_preference === "auto" || model_preference === "cloud") {
        const cloudResult = await tryCloudLlm(messages, max_tokens, temperature);
        if (cloudResult)
            return cloudResult;
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
async function tryLocalInference(messages, max_tokens, temperature) {
    try {
        console.error("Trying Tier 1: Local inference-service...");
        const response = await apiCall("/v1/chat/completions", "POST", { model: "auto", messages, max_tokens, temperature }, INFERENCE_SERVICE_URL, 30000);
        return {
            tier: "local",
            model: response.model || "local",
            content: response.choices[0].message.content,
            usage: response.usage || {},
        };
    }
    catch (error) {
        console.error(`Tier 1 (local) failed: ${error}`);
        return null;
    }
}
async function tryCloudLlm(messages, max_tokens, temperature) {
    try {
        console.error("Trying Tier 2: Cloud LLM via llm-gateway...");
        const response = await apiCall("/v1/chat/completions", "POST", { model: LLM_GATEWAY_DEFAULT_MODEL, messages, max_tokens, temperature }, LLM_GATEWAY_URL, 60000);
        return {
            tier: "cloud",
            model: response.model || LLM_GATEWAY_DEFAULT_MODEL,
            content: response.choices[0].message.content,
            usage: response.usage || {},
        };
    }
    catch (error) {
        console.error(`Tier 2 (cloud) failed: ${error}`);
        return null;
    }
}
// NOTE: handleGraphGetNeighbors removed - cross-reference now includes Neo4j via UnifiedRetriever
// Main server setup
const server = new Server({
    name: "ai-agents-mcp-server",
    version: "1.0.0",
}, {
    capabilities: {
        tools: {},
    },
});
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
    }
    catch (error) {
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
