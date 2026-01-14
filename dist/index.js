#!/usr/bin/env node
import * as fs from "fs/promises";
// Server is deprecated in favor of McpServer, but McpServer requires significant refactoring
// NOSONAR: Suppressing deprecation warning until migration to McpServer is complete
// @ts-ignore Server class deprecated but McpServer migration pending
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
// Service URLs
const AI_AGENTS_URL = process.env.AI_AGENTS_URL || "http://localhost:8082";
const INFERENCE_SERVICE_URL = process.env.INFERENCE_SERVICE_URL || "http://localhost:8085";
const LLM_GATEWAY_URL = process.env.LLM_GATEWAY_URL || "http://localhost:8080";
// IMPORTANT: Must match llm-gateway/.env LLM_GATEWAY_DEFAULT_MODEL
// Registered models: gpt-5.2, gpt-5.2-pro, claude-opus-4-5-20250514, claude-sonnet-4-5-20250514, gemini-2.0-flash, deepseek-reasoner
const LLM_GATEWAY_DEFAULT_MODEL = process.env.LLM_GATEWAY_DEFAULT_MODEL || "gpt-5.2";
const SEMANTIC_SEARCH_URL = process.env.SEMANTIC_SEARCH_URL || "http://localhost:8081";
const CODE_ORCHESTRATOR_URL = process.env.CODE_ORCHESTRATOR_URL || "http://localhost:8083";
const AUDIT_SERVICE_URL = process.env.AUDIT_SERVICE_URL || "http://localhost:8084";
const NEO4J_HTTP_URL = process.env.NEO4J_HTTP_URL || "http://localhost:7474";
const NEO4J_USER = process.env.NEO4J_USER || "neo4j";
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || "devpassword";
// =============================================================================
// Logging Configuration
// LOG_LEVEL: "error" (default) = only errors, state changes, tool execution
//            "info" = include routine health checks
//            "debug" = verbose
// =============================================================================
const LOG_LEVEL = process.env.LOG_LEVEL || "error";
function logInfo(message) {
    if (LOG_LEVEL === "info" || LOG_LEVEL === "debug") {
        console.error(message);
    }
}
function logDebug(message) {
    if (LOG_LEVEL === "debug") {
        console.error(message);
    }
}
function logError(message) {
    console.error(message);
}
function logTool(message) {
    // Tool execution always logged
    console.error(message);
}
// WBS-PS3: Health cache configuration
const HEALTH_CACHE_TTL_MS = 10000; // Cache health status for 10 seconds
let healthCacheTime = 0; // Last time health was cached
let cachedHealthResponse = null;
const serviceHealth = {
    "ai-agents": {
        url: AI_AGENTS_URL,
        healthy: false,
        lastCheck: 0,
        consecutiveFailures: 0,
        lastSuccessTime: 0,
        totalRequests: 0,
        totalErrors: 0,
        errorRate: 0,
        avgResponseTimeMs: 0,
    },
    "inference-service": {
        url: INFERENCE_SERVICE_URL,
        healthy: false,
        lastCheck: 0,
        consecutiveFailures: 0,
        lastSuccessTime: 0,
        totalRequests: 0,
        totalErrors: 0,
        errorRate: 0,
        avgResponseTimeMs: 0,
    },
    "llm-gateway": {
        url: LLM_GATEWAY_URL,
        healthy: false,
        lastCheck: 0,
        consecutiveFailures: 0,
        lastSuccessTime: 0,
        totalRequests: 0,
        totalErrors: 0,
        errorRate: 0,
        avgResponseTimeMs: 0,
    },
    "semantic-search": {
        url: SEMANTIC_SEARCH_URL,
        healthy: false,
        lastCheck: 0,
        consecutiveFailures: 0,
        lastSuccessTime: 0,
        totalRequests: 0,
        totalErrors: 0,
        errorRate: 0,
        avgResponseTimeMs: 0,
    },
    "code-orchestrator": {
        url: CODE_ORCHESTRATOR_URL,
        healthy: false,
        lastCheck: 0,
        consecutiveFailures: 0,
        lastSuccessTime: 0,
        totalRequests: 0,
        totalErrors: 0,
        errorRate: 0,
        avgResponseTimeMs: 0,
    },
    "audit-service": {
        url: AUDIT_SERVICE_URL,
        healthy: false,
        lastCheck: 0,
        consecutiveFailures: 0,
        lastSuccessTime: 0,
        totalRequests: 0,
        totalErrors: 0,
        errorRate: 0,
        avgResponseTimeMs: 0,
    },
};
const HEALTH_CHECK_INTERVAL_MS = 5000; // Check every 5 seconds when unhealthy
const HEALTH_CHECK_INTERVAL_HEALTHY_MS = 30000; // Check every 30 seconds when healthy
let healthCheckTimer = null;
// WBS-PS3: Rolling average window for response times
const ROLLING_AVERAGE_ALPHA = 0.2; // Exponential moving average factor
async function checkServiceHealth(serviceName) {
    const service = serviceHealth[serviceName];
    if (!service)
        return false;
    const startTime = Date.now();
    service.totalRequests++;
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000);
        const response = await fetch(`${service.url}/health`, {
            method: "GET",
            signal: controller.signal,
        });
        clearTimeout(timeout);
        const responseTime = Date.now() - startTime;
        const wasHealthy = service.healthy;
        service.healthy = response.ok;
        service.lastCheck = Date.now();
        service.consecutiveFailures = 0;
        delete service.lastError;
        // WBS-PS3: Track success metrics
        if (response.ok) {
            service.lastSuccessTime = Date.now();
            service.responseTimeMs = responseTime;
            // Exponential moving average for response time
            service.avgResponseTimeMs = service.avgResponseTimeMs === 0
                ? responseTime
                : service.avgResponseTimeMs * (1 - ROLLING_AVERAGE_ALPHA) + responseTime * ROLLING_AVERAGE_ALPHA;
            // Try to parse health details from response
            try {
                const healthData = await response.json();
                service.healthDetails = healthData;
            }
            catch {
                // Response might not be JSON, that's okay
                service.healthDetails = { raw: "ok" };
            }
        }
        // Update error rate (rolling)
        service.errorRate = service.totalErrors / service.totalRequests;
        if (!wasHealthy && service.healthy) {
            logError(`✅ ${serviceName} is now ONLINE at ${service.url} (${responseTime}ms)`);
        }
        else {
            logDebug(`[health] ${serviceName} healthy (${responseTime}ms)`);
        }
        return service.healthy;
    }
    catch (error) {
        const responseTime = Date.now() - startTime;
        service.healthy = false;
        service.lastCheck = Date.now();
        service.consecutiveFailures++;
        service.totalErrors++;
        service.errorRate = service.totalErrors / service.totalRequests;
        service.responseTimeMs = responseTime;
        service.lastError = error instanceof Error ? error.message : String(error);
        // Log first failure, then every 10th to reduce noise
        if (service.consecutiveFailures === 1) {
            logError(`❌ ${serviceName} OFFLINE: ${service.lastError}`);
        }
        else if (service.consecutiveFailures % 10 === 0) {
            logInfo(`⏳ ${serviceName} still offline (attempt ${service.consecutiveFailures})`);
        }
        return false;
    }
}
async function checkAllServices() {
    const results = {};
    await Promise.all(Object.keys(serviceHealth).map(async (name) => {
        results[name] = await checkServiceHealth(name);
    }));
    return results;
}
function startHealthMonitor() {
    if (healthCheckTimer)
        return;
    logInfo("🔄 Starting health monitor - will auto-connect when services come online...");
    const runHealthCheck = async () => {
        const results = await checkAllServices();
        const anyHealthy = Object.values(results).some(Boolean);
        const allHealthy = Object.values(results).every(Boolean);
        // Adjust check interval based on health state
        const interval = allHealthy ? HEALTH_CHECK_INTERVAL_HEALTHY_MS : HEALTH_CHECK_INTERVAL_MS;
        healthCheckTimer = setTimeout(runHealthCheck, interval);
        // Refresh cache when services come online
        if (anyHealthy && lastCacheTime === 0) {
            logInfo("🔄 Services detected - refreshing tool cache...");
            await refreshCache();
        }
    };
    // Initial check immediately
    runHealthCheck();
}
function getServiceStatus() {
    const status = {};
    for (const [name, health] of Object.entries(serviceHealth)) {
        status[name] = {
            healthy: health.healthy,
            url: health.url,
            lastError: health.lastError,
        };
    }
    return status;
}
// WBS-PS3: Build structured health response with all metrics
function buildStructuredHealthResponse(fromCache = false) {
    const now = Date.now();
    const services = {};
    let healthyCount = 0;
    const totalCount = Object.keys(serviceHealth).length;
    for (const [name, health] of Object.entries(serviceHealth)) {
        if (health.healthy)
            healthyCount++;
        services[name] = {
            healthy: health.healthy,
            url: health.url,
            lastCheck: health.lastCheck > 0 ? new Date(health.lastCheck).toISOString() : "never",
            lastSuccess: health.lastSuccessTime > 0 ? new Date(health.lastSuccessTime).toISOString() : null,
            lastError: health.lastError,
            consecutiveFailures: health.consecutiveFailures,
            totalRequests: health.totalRequests,
            totalErrors: health.totalErrors,
            errorRate: `${(health.errorRate * 100).toFixed(2)}%`,
            responseTimeMs: health.responseTimeMs,
            avgResponseTimeMs: Math.round(health.avgResponseTimeMs),
            healthDetails: health.healthDetails,
        };
    }
    // Determine overall status
    let status;
    let message;
    if (healthyCount === totalCount) {
        status = "healthy";
        message = "All services operational";
    }
    else if (healthyCount > 0) {
        status = "degraded";
        message = `${healthyCount}/${totalCount} services operational`;
    }
    else if (Object.values(serviceHealth).some(s => s.totalRequests > 0)) {
        status = "unhealthy";
        message = "All services unavailable";
    }
    else {
        status = "waiting";
        message = "Waiting for platform services to start...";
    }
    return {
        status,
        timestamp: new Date(now).toISOString(),
        cached: fromCache,
        cacheAgeMs: fromCache && healthCacheTime > 0 ? now - healthCacheTime : undefined,
        summary: {
            total: totalCount,
            healthy: healthyCount,
            unhealthy: totalCount - healthyCount,
            healthPercentage: `${((healthyCount / totalCount) * 100).toFixed(1)}%`,
        },
        services,
        message,
        hint: healthyCount === 0 ? "Start the platform with: docker-compose up -d" : undefined,
    };
}
// WBS-PS3: Get cached or fresh health response
async function getStructuredHealthResponse() {
    const now = Date.now();
    // Return cached response if within TTL
    if (cachedHealthResponse && (now - healthCacheTime) < HEALTH_CACHE_TTL_MS) {
        return buildStructuredHealthResponse(true);
    }
    // Refresh health checks
    await checkAllServices();
    // Build and cache response
    cachedHealthResponse = buildStructuredHealthResponse(false);
    healthCacheTime = now;
    return cachedHealthResponse;
}
function isServiceHealthy(serviceName) {
    return serviceHealth[serviceName]?.healthy ?? false;
}
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
        logInfo(`📦 Cache refreshed: ${cachedFunctions.length} functions, ${cachedProtocols.length} protocols`);
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
        // =============================================================================
        // Core Agent Functions (8 functions from Kitchen Brigade)
        // These are first-class tools that call the ai-agents HTTP API
        // =============================================================================
        {
            name: "extract_structure",
            annotations: { readOnlyHint: true, title: "Extract Structure" },
            description: "Extract structured data from unstructured content. Parses JSON/Markdown/Code into hierarchical structure with headings, sections, and code blocks. Reference: Agent Function 1.",
            inputSchema: {
                type: "object",
                properties: {
                    content: { type: "string", description: "The content to extract structure from" },
                    extraction_type: { type: "string", enum: ["outline", "entities", "keywords"], description: "Type of extraction (default: outline)" },
                },
                required: ["content"],
            },
        },
        {
            name: "summarize_content",
            description: "Compress content while preserving key information. Generates summaries with citation markers for traceability. Reference: Agent Function 2.",
            annotations: { readOnlyHint: true, title: "Summarize Content" },
            inputSchema: {
                type: "object",
                properties: {
                    content: { type: "string", description: "The content to summarize" },
                    detail_level: { type: "string", enum: ["brief", "standard", "detailed"], description: "Level of detail (default: standard)" },
                    max_length: { type: "number", description: "Maximum length of summary in words (default: 500)" },
                },
                required: ["content"],
            },
        },
        {
            name: "generate_code",
            description: "Generate code from natural language specification. Reference: Agent Function 3.",
            annotations: { readOnlyHint: true, title: "Generate Code" },
            inputSchema: {
                type: "object",
                properties: {
                    specification: { type: "string", description: "Natural language description of what to generate" },
                    target_language: { type: "string", description: "Programming language for output (default: python)" },
                    include_tests: { type: "boolean", description: "Whether to include test stubs (default: false)" },
                    context: { type: "string", description: "Additional context about the codebase" },
                },
                required: ["specification"],
            },
        },
        {
            name: "analyze_artifact",
            description: "Analyze code or document for patterns, issues, and quality. Reference: Agent Function 4.",
            annotations: { readOnlyHint: true, title: "Analyze Artifact" },
            inputSchema: {
                type: "object",
                properties: {
                    artifact: { type: "string", description: "The code or document to analyze" },
                    analysis_type: { type: "string", enum: ["quality", "security", "patterns"], description: "Type of analysis (default: quality)" },
                    context: { type: "string", description: "Additional context about the artifact" },
                },
                required: ["artifact"],
            },
        },
        {
            name: "validate_against_spec",
            description: "Validate an artifact against its specification. Compares code/content against requirements and acceptance criteria. Reference: Agent Function 5.",
            annotations: { readOnlyHint: true, title: "Validate Against Spec" },
            inputSchema: {
                type: "object",
                properties: {
                    artifact: { type: "string", description: "The artifact to validate" },
                    specification: { type: "string", description: "The specification to validate against" },
                    acceptance_criteria: { type: "array", items: { type: "string" }, description: "List of specific criteria to check" },
                },
                required: ["artifact", "specification"],
            },
        },
        {
            name: "decompose_task",
            description: "Decompose a complex task into subtasks. Breaks down high-level objectives into executable subtasks with dependencies, forming a valid DAG for pipeline execution. Reference: Agent Function 6.",
            annotations: { readOnlyHint: true, title: "Decompose Task" },
            inputSchema: {
                type: "object",
                properties: {
                    task: { type: "string", description: "The task to decompose" },
                    constraints: { type: "array", items: { type: "string" }, description: "Constraints to consider during decomposition" },
                    available_agents: { type: "array", items: { type: "string" }, description: "List of available agent functions" },
                    context: { type: "string", description: "Additional context about the task" },
                },
                required: ["task"],
            },
        },
        {
            name: "synthesize_outputs",
            description: "Combine multiple outputs into a coherent result. Merges outputs from multiple agents while tracking provenance. Reference: Agent Function 7.",
            annotations: { readOnlyHint: true, title: "Synthesize Outputs" },
            inputSchema: {
                type: "object",
                properties: {
                    outputs: { type: "array", items: { type: "string" }, description: "List of outputs to synthesize" },
                    synthesis_strategy: { type: "string", enum: ["merge", "chain", "vote"], description: "Strategy (default: merge)" },
                    conflict_resolution: { type: "string", enum: ["latest", "vote", "manual"], description: "How to resolve conflicts (default: latest)" },
                },
                required: ["outputs"],
            },
        },
        {
            name: "cross_reference",
            description: "Find related content across knowledge sources. Queries semantic search to find related content across code, documentation, and textbooks. Reference: Agent Function 8.",
            annotations: { readOnlyHint: true, title: "Cross Reference" },
            inputSchema: {
                type: "object",
                properties: {
                    query: { type: "string", description: "The query to search for" },
                    sources: { type: "array", items: { type: "string" }, description: "Specific sources to search (default: all)" },
                    top_k: { type: "number", description: "Number of results to return (default: 5)" },
                    include_code: { type: "boolean", description: "Whether to search code repositories (default: true)" },
                    include_books: { type: "boolean", description: "Whether to search textbooks (default: true)" },
                },
                required: ["query"],
            },
        },
        // =============================================================================
        // Platform Management Tools
        // =============================================================================
        {
            name: "ai_agents_health",
            description: "Check health status of AI Agents service and all Kitchen Brigade dependencies. Returns structured JSON with per-service details including: healthy/unhealthy status, last successful request timestamp, error counts/rates, response times, and cached health details. Health status is cached for 10 seconds to avoid hammering services.",
            annotations: { readOnlyHint: true, title: "Ai Agents Health" },
            inputSchema: {
                type: "object",
                properties: {},
                required: [],
            },
        },
        {
            name: "ai_agents_list_functions",
            description: "List all available agent functions that can be executed",
            annotations: { readOnlyHint: true, title: "Ai Agents List Functions" },
            inputSchema: {
                type: "object",
                properties: {},
                required: [],
            },
        },
        {
            name: "ai_agents_list_protocols",
            description: "List all available Kitchen Brigade protocols for multi-agent collaboration",
            annotations: { readOnlyHint: true, title: "Ai Agents List Protocols" },
            inputSchema: {
                type: "object",
                properties: {},
                required: [],
            },
        },
        // Generic execution tools
        {
            name: "ai_agents_run_function",
            description: "Execute an agent function by name. Use ai_agents_list_functions to see available functions like summarize-content, generate-code, analyze-artifact, etc.",
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
            description: "Execute a Kitchen Brigade protocol for multi-agent collaboration. Protocols include ROUNDTABLE_DISCUSSION, DEBATE_PROTOCOL, WBS_GENERATION, etc.",
            inputSchema: {
                type: "object",
                properties: {
                    protocol_id: {
                        type: "string",
                        description: "Protocol ID (e.g., 'ROUNDTABLE_DISCUSSION', 'DEBATE_PROTOCOL', 'WBS_GENERATION', 'ARCHITECTURE_RECONCILIATION')",
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
            annotations: { readOnlyHint: true, title: "Semantic Search" },
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
            annotations: { readOnlyHint: true, title: "Hybrid Search" },
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
            annotations: { readOnlyHint: true, title: "Code Analyze" },
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
            description: "Query the Neo4j knowledge graph directly using Cypher. Access relationships between code entities, concepts, and documentation.",
            annotations: { readOnlyHint: true, title: "Graph Query" },
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
            description: "Get all nodes connected to a specific node in the knowledge graph. Useful for exploring relationships.",
            annotations: { readOnlyHint: true, title: "Graph Get Neighbors" },
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
        // Textbook Search - Direct access to textbook JSON files
        {
            name: "textbook_search",
            description: "Search through textbook JSON files for relevant passages. This searches your 256 indexed textbooks including AI Agents in Action, Building Microservices, etc.",
            annotations: { readOnlyHint: true, title: "Textbook Search" },
            inputSchema: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "Natural language query to search textbooks",
                    },
                    books: {
                        type: "array",
                        items: { type: "string" },
                        description: "Specific book titles to search (optional, searches all if empty)",
                    },
                    top_k: {
                        type: "number",
                        description: "Number of passages to return (default: 5)",
                    },
                    include_chapters: {
                        type: "boolean",
                        description: "Include chapter summaries in addition to passages (default: true)",
                    },
                },
                required: ["query"],
            },
        },
        // Audit Service - Generate citations and footnotes
        {
            name: "audit_generate_footnotes",
            description: "Generate Chicago-style footnotes from citation markers. Use after cross-reference to format citations properly.",
            inputSchema: {
                type: "object",
                properties: {
                    citations: {
                        type: "array",
                        items: {
                            type: "object",
                            properties: {
                                marker: { type: "string", description: "Citation marker (e.g., '[^1]')" },
                                source_id: { type: "string", description: "Source identifier" },
                                source_type: { type: "string", enum: ["book", "code", "textbook", "web"] },
                            },
                        },
                        description: "List of citations to format",
                    },
                    task_id: {
                        type: "string",
                        description: "Optional task ID for audit trail",
                    },
                },
                required: ["citations"],
            },
        },
        // Audit Service - Validate citations exist
        {
            name: "audit_validate_citations",
            description: "Validate that cited sources actually exist and contain the claimed content. Use for hallucination detection.",
            inputSchema: {
                type: "object",
                properties: {
                    content: {
                        type: "string",
                        description: "Content with citation markers to validate",
                    },
                    citations: {
                        type: "array",
                        items: {
                            type: "object",
                            properties: {
                                marker: { type: "string" },
                                source_path: { type: "string" },
                                claimed_content: { type: "string" },
                            },
                        },
                        description: "Citations to validate against sources",
                    },
                },
                required: ["content", "citations"],
            },
        },
        // Full Cross-Reference Pipeline (Stage 2 complete)
        {
            name: "cross_reference_full",
            description: "Execute complete Stage 2 cross-reference: parallel search across Qdrant (vectors), Neo4j (graph), Textbooks (JSON), Code-Reference-Engine, and Code Chunks (actual GitHub code). Returns unified results with relevance scores.",
            annotations: { readOnlyHint: true, title: "Cross Reference Full" },
            inputSchema: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "The query to cross-reference (e.g., 'Gateway API routing patterns')",
                    },
                    sources: {
                        type: "array",
                        items: { type: "string", enum: ["qdrant", "neo4j", "textbooks", "code", "code_chunks"] },
                        description: "Which sources to search (default: all including code_chunks)",
                    },
                    top_k: {
                        type: "number",
                        description: "Results per source (default: 5)",
                    },
                    merge_strategy: {
                        type: "string",
                        enum: ["interleave", "by_source", "by_relevance"],
                        description: "How to combine results (default: by_relevance)",
                    },
                },
                required: ["query"],
            },
        },
        // =============================================================================
        // Test Compliance Audit Tool
        // =============================================================================
        {
            name: "test_compliance_audit",
            description: "Audit a repository for test compliance per TEST_AUDIT_GUIDELINES.md. Scans test files for anti-patterns (AP1-AP4), calculates mock-to-test ratios (U2), and scores compliance. Returns detailed violations, scores (0-100), and remediation guidance. Use for continuous test quality monitoring. Set use_semantic=true to use vector search against indexed test examples for dynamic compliance detection.",
            annotations: { readOnlyHint: true, title: "Test Compliance Audit" },
            inputSchema: {
                type: "object",
                properties: {
                    repo_path: {
                        type: "string",
                        description: "Absolute path to the repository to audit (e.g., '/Users/kevintoles/POC/ai-agents')",
                    },
                    repo_name: {
                        type: "string",
                        description: "Repository name for reporting (e.g., 'ai-agents'). If not provided, derived from repo_path.",
                    },
                    rules: {
                        type: "array",
                        items: { type: "string", enum: ["AP1", "AP2", "AP3", "AP4", "U2", "all"] },
                        description: "Which rules to check (default: all). AP1=assertions in integration, AP2=mocks in integration, AP3=excessive mocking, AP4=tests without assertions, U2=mock ratio",
                    },
                    include_load_tests: {
                        type: "boolean",
                        description: "Whether to include load/benchmark tests in analysis (default: false, they are exempt from AP1)",
                    },
                    output_format: {
                        type: "string",
                        enum: ["summary", "detailed", "json"],
                        description: "Output format (default: detailed)",
                    },
                    use_semantic: {
                        type: "boolean",
                        description: "Use semantic search against indexed test examples for dynamic compliance detection (default: false). Requires test_good_patterns and test_bad_patterns collections in Qdrant.",
                    },
                    cross_reference: {
                        type: "boolean",
                        description: "Cross-reference violations against test_good_patterns to provide remediation examples (default: true). Shows similar compliant tests for each violation.",
                    },
                },
                required: ["repo_path"],
            },
        },
        // =============================================================================
        // Code Pattern Audit Tool
        // =============================================================================
        {
            name: "code_pattern_audit",
            description: "Audit a repository for code pattern compliance per AUDIT_CODE_PATTERNS_TOOL_SPEC.md. Detects anti-patterns from CODING_PATTERNS_ANALYSIS.md (252 patterns) and SonarQube rules. Set use_semantic=true to use dual-net vector search against code_good_patterns and code_bad_patterns collections for dynamic semantic detection. Returns violations with remediation examples from the pattern library.",
            annotations: { readOnlyHint: true, title: "Code Pattern Audit" },
            inputSchema: {
                type: "object",
                properties: {
                    repo_path: {
                        type: "string",
                        description: "Absolute path to the repository to audit (e.g., '/Users/kevintoles/POC/llm-gateway')",
                    },
                    repo_name: {
                        type: "string",
                        description: "Repository name for reporting. If not provided, derived from repo_path.",
                    },
                    pattern_categories: {
                        type: "array",
                        items: { type: "string", enum: ["complexity", "security", "maintainability", "reliability", "performance", "all"] },
                        description: "Which pattern categories to check (default: all)",
                    },
                    severity_threshold: {
                        type: "string",
                        enum: ["blocker", "critical", "major", "minor", "info"],
                        description: "Minimum severity to report (default: major)",
                    },
                    include_remediation: {
                        type: "boolean",
                        description: "Include before/after code examples from pattern library (default: true)",
                    },
                    use_semantic: {
                        type: "boolean",
                        description: "Use semantic search against indexed code patterns for dynamic compliance detection (default: false). Requires code_good_patterns and code_bad_patterns collections in Qdrant.",
                    },
                    cross_reference: {
                        type: "boolean",
                        description: "Cross-reference violations against code_good_patterns to provide remediation examples (default: true). Shows similar compliant code for each violation.",
                    },
                },
                required: ["repo_path"],
            },
        },
    ];
    // Add dynamic function-specific tools
    for (const fn of cachedFunctions) {
        const inputSchema = fn.input_schema && typeof fn.input_schema === 'object' && 'type' in fn.input_schema
            ? fn.input_schema
            : {
                type: "object",
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
        if (!protocol?.protocol_id)
            continue; // Skip protocols without valid protocol_id
        tools.push({
            name: `ai_protocol_${protocol.protocol_id.toLowerCase().replaceAll("-", "_")}`,
            description: protocol.description || `Execute the ${protocol.name || protocol.protocol_id} Kitchen Brigade protocol`,
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
// =============================================================================
// Tool Handlers - Each handler is a focused function to reduce complexity
// =============================================================================
async function handleHealthTool() {
    return await getStructuredHealthResponse();
}
async function handleListFunctions() {
    if (!isServiceHealthy("ai-agents")) {
        return {
            error: "ai-agents service not available",
            status: "waiting",
            message: "The ai-agents service is not running. Start the platform to use this tool.",
            cached_functions: cachedFunctions.length > 0 ? cachedFunctions.map((f) => f.name) : undefined,
        };
    }
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
    if (!isServiceHealthy("ai-agents")) {
        return {
            error: "ai-agents service not available",
            status: "waiting",
            message: "The ai-agents service is not running. Start the platform to use this tool.",
            cached_protocols: cachedProtocols.length > 0 ? cachedProtocols.map((p) => p.protocol_id) : undefined,
        };
    }
    await refreshCache();
    return {
        protocols: cachedProtocols.map((p) => ({
            protocol_id: p.protocol_id,
            name: p.name,
            description: p.description,
            brigade_roles: p.brigade_roles,
        })),
        count: cachedProtocols.length,
    };
}
async function handleRunFunction(args) {
    if (!isServiceHealthy("ai-agents")) {
        return {
            error: "ai-agents service not available",
            status: "waiting",
            message: "Cannot execute functions - ai-agents service is not running.",
            requested_function: args.function_name,
            hint: "Start the platform with: docker-compose up -d",
        };
    }
    const { function_name, input, preset } = args;
    return apiCall(`/v1/functions/${function_name}/run`, "POST", { input, preset });
}
// Helper to get preflight hint message
function getPreflightHint(blockingIssues) {
    if (blockingIssues.some((i) => i.includes("inference-service"))) {
        return "Start inference-service: cd inference-service && source .venv/bin/activate && python -m uvicorn src.main:app --port 8085";
    }
    if (blockingIssues.some((i) => i.includes("not found"))) {
        return "Required model not available in inference-service";
    }
    return "Check service health with ai_agents_health tool";
}
// Helper to run preflight check
async function runPreflightCheck(protocolId, brigadeOverride, enableCrossReference) {
    try {
        const preflightResult = await apiCall(`/v1/protocols/${protocolId}/preflight`, "POST", { brigade_override: brigadeOverride, enable_cross_reference: enableCrossReference }, AI_AGENTS_URL, 5000);
        logTool(`✓ [PREFLIGHT] Complete in ${preflightResult.check_time_ms}ms - ready: ${preflightResult.ready}`);
        if (!preflightResult.ready) {
            logError(`🚫 [PREFLIGHT] BLOCKED: ${preflightResult.blocking_issues.join(", ")}`);
            return {
                ready: false,
                result: {
                    error: "preflight_failed",
                    status: "blocked",
                    message: `Protocol cannot execute: ${preflightResult.blocking_issues.join("; ")}`,
                    protocol_id: protocolId,
                    preflight: preflightResult,
                    hint: getPreflightHint(preflightResult.blocking_issues),
                },
            };
        }
        if (preflightResult.warnings.length > 0) {
            logTool(`⚠️ [PREFLIGHT] Warnings: ${preflightResult.warnings.join(", ")}`);
        }
        return { ready: true };
    }
    catch (preflightError) {
        logError(`⚠️ [PREFLIGHT] Check failed: ${preflightError}. Proceeding with caution...`);
        return { ready: true }; // Proceed anyway if preflight fails
    }
}
async function handleRunProtocol(args) {
    if (!isServiceHealthy("ai-agents")) {
        return {
            error: "ai-agents service not available",
            status: "waiting",
            message: "Cannot execute protocols - ai-agents service is not running.",
            requested_protocol: args.protocol_id,
            hint: "Start the platform with: docker-compose up -d",
        };
    }
    const { protocol_id, inputs, config, brigade_override } = args;
    logTool(`🔍 [PREFLIGHT] Checking prerequisites for protocol ${protocol_id}...`);
    const preflight = await runPreflightCheck(protocol_id, brigade_override, config?.run_cross_reference !== false);
    if (!preflight.ready) {
        return preflight.result;
    }
    logTool(`🚀 [PROTOCOL] Executing ${protocol_id}...`);
    return apiCall(`/v1/protocols/${protocol_id}/run`, "POST", { inputs, config, brigade_override }, AI_AGENTS_URL, 300000);
}
async function handleDynamicFunction(name, args) {
    const fnName = name.replace("ai_fn_", "").replaceAll("_", "-");
    const { input, preset, ...rest } = args;
    return apiCall(`/v1/functions/${fnName}/run`, "POST", { input: input || rest, preset });
}
async function handleDynamicProtocol(name, args) {
    const protocolId = name.replace("ai_protocol_", "").toUpperCase().replaceAll("_", "-");
    const { inputs, config, brigade_override } = args;
    console.error(`[PREFLIGHT] Checking prerequisites for protocol ${protocolId}...`);
    const preflight = await runPreflightCheck(protocolId, brigade_override, config?.run_cross_reference !== false);
    if (!preflight.ready) {
        return preflight.result;
    }
    return apiCall(`/v1/protocols/${protocolId}/run`, "POST", { inputs, config, brigade_override }, AI_AGENTS_URL, 300000);
}
// Helper to try local inference
async function tryLocalInference(messages, maxTokens, temperature) {
    try {
        console.error("Trying Tier 1: Local inference-service...");
        let localModel = "qwen3-8b";
        try {
            const modelsResponse = await apiCall("/v1/models", "GET", undefined, INFERENCE_SERVICE_URL, 5000);
            const loadedModels = modelsResponse.data.filter((m) => m.status === "loaded");
            if (loadedModels.length > 0) {
                localModel = loadedModels[0].id;
                console.error(`Using loaded model: ${localModel}`);
            }
        }
        catch {
            console.error("Could not query models, using default");
        }
        const response = await apiCall("/v1/chat/completions", "POST", { model: localModel, messages, max_tokens: maxTokens, temperature }, INFERENCE_SERVICE_URL, 120000);
        return {
            success: true,
            result: {
                tier: "local",
                model: response.model || localModel,
                content: response.choices[0].message.content,
                usage: response.usage || {},
            },
        };
    }
    catch (error) {
        console.error(`Tier 1 (local) failed: ${error}`);
        return { success: false };
    }
}
// Helper to try cloud inference
async function tryCloudInference(messages, maxTokens, temperature) {
    try {
        console.error("Trying Tier 2: Cloud LLM via llm-gateway...");
        const response = await apiCall("/v1/chat/completions", "POST", { model: LLM_GATEWAY_DEFAULT_MODEL, messages, max_tokens: maxTokens, temperature }, LLM_GATEWAY_URL, 60000);
        return {
            success: true,
            result: {
                tier: "cloud",
                model: response.model || LLM_GATEWAY_DEFAULT_MODEL,
                content: response.choices[0].message.content,
                usage: response.usage || {},
            },
        };
    }
    catch (error) {
        console.error(`Tier 2 (cloud) failed: ${error}`);
        return { success: false };
    }
}
async function handleLlmComplete(args) {
    const { prompt, model_preference = "auto", max_tokens = 4096, temperature = 0.7, system_prompt, } = args;
    const messages = [];
    if (system_prompt) {
        messages.push({ role: "system", content: system_prompt });
    }
    messages.push({ role: "user", content: prompt });
    if (model_preference === "auto" || model_preference === "local") {
        const local = await tryLocalInference(messages, max_tokens, temperature);
        if (local.success)
            return local.result;
    }
    if (model_preference === "auto" || model_preference === "cloud") {
        const cloud = await tryCloudInference(messages, max_tokens, temperature);
        if (cloud.success)
            return cloud.result;
    }
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
// Helper to execute Neo4j query
async function executeNeo4jQuery(cypher, parameters) {
    const auth = Buffer.from(`${NEO4J_USER}:${NEO4J_PASSWORD}`).toString("base64");
    const response = await fetch(`${NEO4J_HTTP_URL}/db/neo4j/tx/commit`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Basic ${auth}` },
        body: JSON.stringify({ statements: [{ statement: cypher, parameters }] }),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Neo4j error (${response.status}): ${error}`);
    }
    return response.json();
}
async function handleGraphQuery(args) {
    const { cypher, parameters = {} } = args;
    const data = await executeNeo4jQuery(cypher, parameters);
    if (data.errors && data.errors.length > 0) {
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
async function handleGraphGetNeighbors(args) {
    const { node_id, node_type, relationship_type, depth = 1 } = args;
    const nodeMatch = node_type ? `(n:${node_type} {name: $node_id})` : `(n {name: $node_id})`;
    const relMatch = relationship_type ? `-[r:${relationship_type}*1..${depth}]-` : `-[r*1..${depth}]-`;
    const cypher = `MATCH ${nodeMatch}${relMatch}(m) RETURN DISTINCT n, type(r[0]) as relationship, m LIMIT 50`;
    const data = await executeNeo4jQuery(cypher, { node_id });
    if (data.errors && data.errors.length > 0) {
        throw new Error(`Cypher error: ${data.errors[0].message}`);
    }
    const result = data.results[0];
    if (!result)
        return { source: node_id, neighbors: [] };
    return {
        source: node_id,
        neighbors: result.data.map((d) => ({ relationship: d.row[1], node: d.row[2] })),
        count: result.data.length,
    };
}
async function handleTextbookSearch(args) {
    const { query, top_k = 5 } = args;
    return apiCall("/v1/search", "POST", { query, collection: "chapters", limit: top_k }, SEMANTIC_SEARCH_URL);
}
async function handleAuditGenerateFootnotes(args) {
    const { citations, task_id } = args;
    return apiCall("/v1/footnotes", "POST", { citations, task_id }, AUDIT_SERVICE_URL);
}
async function handleAuditValidateCitations(args) {
    const { content, citations } = args;
    return apiCall("/v1/validate", "POST", { content, citations }, AUDIT_SERVICE_URL);
}
// Helper to build cross-reference search promises
function buildCrossRefSearchPromises(query, sources, topK) {
    const searchPromises = [];
    if (sources.includes("qdrant")) {
        searchPromises.push(apiCall("/v1/search", "POST", { query, collection: "all", limit: topK }, SEMANTIC_SEARCH_URL)
            .then(results => ({ source: "qdrant", results }))
            .catch(err => ({ source: "qdrant", results: { error: String(err) } })));
    }
    if (sources.includes("neo4j")) {
        const cypher = `CALL db.index.fulltext.queryNodes("concept_search", $query) YIELD node, score RETURN node, score ORDER BY score DESC LIMIT $top_k`;
        const auth = Buffer.from(`${NEO4J_USER}:${NEO4J_PASSWORD}`).toString("base64");
        searchPromises.push(fetch(`${NEO4J_HTTP_URL}/db/neo4j/tx/commit`, {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: `Basic ${auth}` },
            body: JSON.stringify({ statements: [{ statement: cypher, parameters: { query, top_k: topK } }] }),
        })
            .then(r => r.json())
            .then(data => ({ source: "neo4j", results: data }))
            .catch(err => ({ source: "neo4j", results: { error: String(err) } })));
    }
    if (sources.includes("textbooks")) {
        searchPromises.push(apiCall("/v1/search", "POST", { query, collection: "chapters", limit: topK }, SEMANTIC_SEARCH_URL)
            .then(results => ({ source: "textbooks", results }))
            .catch(err => ({ source: "textbooks", results: { error: String(err) } })));
    }
    if (sources.includes("code")) {
        searchPromises.push(apiCall("/v1/search", "POST", { query, analysis_type: "semantic" }, CODE_ORCHESTRATOR_URL)
            .then(results => ({ source: "code", results }))
            .catch(err => ({ source: "code", results: { error: String(err) } })));
    }
    if (sources.includes("code_chunks")) {
        searchPromises.push(apiCall("/v1/search", "POST", { query, collection: "code_chunks", limit: topK }, SEMANTIC_SEARCH_URL)
            .then(results => ({ source: "code_chunks", results }))
            .catch(err => ({ source: "code_chunks", results: { error: String(err) } })));
    }
    return searchPromises;
}
async function handleCrossReferenceFull(args) {
    const { query, sources = ["qdrant", "neo4j", "textbooks", "code", "code_chunks"], top_k = 5, merge_strategy = "by_relevance", } = args;
    const searchPromises = buildCrossRefSearchPromises(query, sources, top_k);
    const allResults = await Promise.all(searchPromises);
    return {
        query,
        sources_searched: sources,
        merge_strategy,
        results: allResults,
        total_sources: allResults.length,
    };
}
// Helper to read directory recursively
async function readDirRecursive(dir, pattern) {
    const { readdir, stat } = await import("fs/promises");
    const { join } = await import("path");
    const files = [];
    async function walk(currentDir) {
        try {
            const entries = await readdir(currentDir, { withFileTypes: true });
            for (const entry of entries) {
                const fullPath = join(currentDir, entry.name);
                if (entry.isDirectory()) {
                    // Skip common non-test directories
                    if (!["node_modules", ".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".mypy_cache", "dist", "build"].includes(entry.name)) {
                        await walk(fullPath);
                    }
                }
                else if (pattern.test(entry.name)) {
                    files.push(fullPath);
                }
            }
        }
        catch {
            // Directory might not be readable
        }
    }
    await walk(dir);
    return files;
}
// Helper to read file content
async function readFileContent(filePath) {
    const { readFile } = await import("fs/promises");
    try {
        return await readFile(filePath, "utf-8");
    }
    catch {
        return "";
    }
}
// Patterns for test compliance detection
const ASSERTION_PATTERNS = [
    /assert\s+/,
    /self\.assert/,
    /pytest\.raises/,
    /\.should\(/,
    /expect\(/,
    /\.to_equal/,
    /\.to_be/,
    /\.toEqual/,
    /\.toBe/,
    /\.validate\(/,
    /\.verify\(/,
    /with\s+pytest\.raises/,
];
const MOCK_PATTERNS = [
    /@patch/,
    /mock\./i,
    /Mock\(/,
    /MagicMock/,
    /AsyncMock/,
    /patch\(/,
    /mocker\./,
    /monkeypatch\./,
    /unittest\.mock/,
];
const INTEGRATION_REAL_PATTERNS = [
    /httpx\.AsyncClient/,
    /aiohttp\./,
    /requests\./,
    /TestClient/,
    /AsyncClient/,
    /psycopg/,
    /asyncpg/,
    /redis\./,
    /qdrant_client/,
    /neo4j\./,
    /docker/i,
    /testcontainers/i,
    // Service client patterns (real HTTP clients wrapped in classes)
    /Client\(\)/,
    /client\.health_check/,
    /\.health_check\(\)/,
    // Service availability fixture patterns (skip when service unavailable)
    /cms_available/,
    /service_available/,
    /ai_agents_available/,
    /mcp_server_available/,
    /pytest\.skip.*not available/i,
    // Pipeline patterns (real processing)
    /pipeline\.process/,
    /\.process_book/,
    /\.process_chapter/,
    /\.process_all/,
];
const LOAD_BENCHMARK_PATTERNS = [
    /test_load/i,
    /test_benchmark/i,
    /test_performance/i,
    /test_stress/i,
    /@pytest\.mark\.benchmark/,
    /benchmark/i,
    /locust/i,
];
// File-level patterns that indicate integration tests use real services
// These fixtures/functions skip tests when services are unavailable
const FILE_SERVICE_AVAILABILITY_PATTERNS = [
    // Fixture patterns that check service availability
    /@pytest\.fixture[\s\S]*?(?:_available|_healthy|_running)\s*\(/,
    /def\s+\w*(?:available|healthy|running)\s*\([^)]*\):/,
    /async def\s+\w*(?:available|healthy|running)\s*\([^)]*\):/,
    // pytest.skip patterns for service availability
    /pytest\.skip\s*\([^)]*(?:not available|unavailable|not running|offline)/i,
    // Service health check in fixture
    /\.health_check\(\)[\s\S]*?pytest\.skip/,
    // Common service client patterns in fixtures
    /@pytest\.fixture[\s\S]*?Client\(\)/,
];
const CONFIG_MOCK_PATTERNS = [
    /mock_settings/i,
    /mock_config/i,
    /mock_env/i,
    /monkeypatch\.setenv/,
    /os\.environ/,
];
async function handleTestComplianceAudit(args) {
    const { repo_path, repo_name, rules = ["all"], include_load_tests = false, output_format = "detailed", use_semantic = false, cross_reference = true, } = args;
    const { basename } = await import("path");
    const repoName = repo_name || basename(repo_path);
    const checkRules = rules.includes("all") ? ["AP1", "AP2", "AP3", "AP4", "U2"] : rules;
    const mode = use_semantic ? "SEMANTIC" : "REGEX";
    logTool(`🔍 [TEST_AUDIT] Starting ${mode} audit of ${repoName} at ${repo_path}`);
    // Find all test files
    const testFiles = await readDirRecursive(repo_path, /^test_.*\.py$|_test\.py$/);
    logTool(`📁 [TEST_AUDIT] Found ${testFiles.length} test files`);
    const violations = [];
    const manualReviewItems = [];
    const semanticStats = {
        compliant: 0,
        violation: 0,
        manual_review: 0,
        uncertain: 0,
    };
    let totalTestFunctions = 0;
    let totalMocks = 0;
    let integrationTestsWithMocks = 0;
    let testsWithoutAssertions = 0;
    // Analyze each test file
    for (const filePath of testFiles) {
        const content = await readFileContent(filePath);
        const relativePath = filePath.replace(repo_path, "").replace(/^\//, "");
        const isIntegrationTest = relativePath.includes("/integration/") || relativePath.includes("integration_");
        const isLoadTest = LOAD_BENCHMARK_PATTERNS.some(p => p.test(content) || p.test(relativePath));
        // Skip load tests if not included
        if (isLoadTest && !include_load_tests) {
            continue;
        }
        // Extract test functions using regex (simpler than full AST for TypeScript)
        const testFunctionMatches = content.matchAll(/(?:def|async def)\s+(test_\w+)\s*\([^)]*\):/g);
        const testFunctions = [];
        for (const match of testFunctionMatches) {
            testFunctions.push({ name: match[1], startIndex: match.index || 0 });
        }
        totalTestFunctions += testFunctions.length;
        // Analyze each test function
        for (let i = 0; i < testFunctions.length; i++) {
            const fn = testFunctions[i];
            const nextFn = testFunctions[i + 1];
            const fnEndIndex = nextFn ? nextFn.startIndex : content.length;
            const fnContent = content.slice(fn.startIndex, fnEndIndex);
            const fnLine = content.slice(0, fn.startIndex).split("\n").length;
            // Count mocks in this function
            const mockCount = MOCK_PATTERNS.filter(p => p.test(fnContent)).length;
            totalMocks += mockCount;
            // Check for config mocks (exempt from U2)
            const hasConfigMock = CONFIG_MOCK_PATTERNS.some(p => p.test(fnContent));
            // SEMANTIC MODE: Dual-net vector search (GOOD patterns vs BAD patterns)
            if (use_semantic) {
                const testType = isIntegrationTest ? "integration" : "unit";
                const dualNetResult = await checkDualNetCompliance(fnContent, testType);
                // Track dual-net verdicts
                semanticStats[dualNetResult.verdict]++;
                if (dualNetResult.verdict === "violation" && dualNetResult.confidence > 0.4) {
                    // Clear violation from dual-net analysis
                    for (const rule of dualNetResult.inferred_rules_violated) {
                        if (checkRules.includes(rule)) {
                            violations.push({
                                rule,
                                severity: rule === "AP4" ? "critical" : rule === "AP3" ? "major" : "minor",
                                file: relativePath,
                                line: fnLine,
                                function: fn.name,
                                message: `[SEMANTIC] ${dualNetResult.reasoning}`,
                                code_snippet: fnContent.slice(0, 200),
                            });
                        }
                    }
                    // Track specific metrics
                    if (dualNetResult.inferred_rules_violated.includes("AP1")) {
                        integrationTestsWithMocks++;
                    }
                    if (dualNetResult.inferred_rules_violated.includes("AP4")) {
                        testsWithoutAssertions++;
                    }
                }
                else if (dualNetResult.verdict === "manual_review") {
                    // Flag for human review - conflicting signals from both nets
                    manualReviewItems.push({
                        file: relativePath,
                        line: fnLine,
                        function: fn.name,
                        reasoning: dualNetResult.reasoning,
                        good_score: dualNetResult.good_score,
                        bad_score: dualNetResult.bad_score,
                        top_good_match: dualNetResult.good_matches[0]?.test_id || "none",
                        top_bad_match: dualNetResult.bad_matches[0]?.test_id || "none",
                    });
                }
                else if (dualNetResult.verdict === "uncertain") {
                    // Fall back to regex for this function (don't continue)
                    // Let it fall through to regex checks below
                }
                else {
                    // Compliant - skip regex checks
                    continue;
                }
                // If not uncertain, skip regex checks
                if (dualNetResult.verdict !== "uncertain") {
                    continue;
                }
            }
            // REGEX MODE: Traditional pattern matching
            // AP1: Integration tests should NOT use mocks (unless load test)
            if (checkRules.includes("AP1") && isIntegrationTest && !isLoadTest) {
                const hasMock = MOCK_PATTERNS.some(p => p.test(fnContent));
                const hasRealDep = INTEGRATION_REAL_PATTERNS.some(p => p.test(fnContent));
                if (hasMock && !hasConfigMock) {
                    integrationTestsWithMocks++;
                    violations.push({
                        rule: "AP1",
                        severity: "major",
                        file: relativePath,
                        line: fnLine,
                        function: fn.name,
                        message: "Integration test uses mocks - should use real dependencies",
                        code_snippet: fnContent.slice(0, 200),
                    });
                }
            }
            // AP2: Integration tests must hit real endpoints
            if (checkRules.includes("AP2") && isIntegrationTest) {
                const hasRealDep = INTEGRATION_REAL_PATTERNS.some(p => p.test(fnContent));
                // Also check if file has service availability fixtures that this test might use
                const fileHasServiceFixture = FILE_SERVICE_AVAILABILITY_PATTERNS.some(p => p.test(content));
                if (!hasRealDep && !fileHasServiceFixture && !isLoadTest) {
                    violations.push({
                        rule: "AP2",
                        severity: "minor",
                        file: relativePath,
                        line: fnLine,
                        function: fn.name,
                        message: "Integration test doesn't appear to use real dependencies",
                    });
                }
            }
            // AP3: Excessive mocking per function (>5 mocks in single function)
            if (checkRules.includes("AP3") && mockCount > 5 && !hasConfigMock) {
                violations.push({
                    rule: "AP3",
                    severity: "major",
                    file: relativePath,
                    line: fnLine,
                    function: fn.name,
                    message: `Excessive mocking: ${mockCount} mocks in single function (threshold: 5)`,
                });
            }
            // AP4: Test without assertions
            if (checkRules.includes("AP4")) {
                const hasAssertion = ASSERTION_PATTERNS.some(p => p.test(fnContent));
                if (!hasAssertion) {
                    testsWithoutAssertions++;
                    violations.push({
                        rule: "AP4",
                        severity: "critical",
                        file: relativePath,
                        line: fnLine,
                        function: fn.name,
                        message: "Test function has no assertions",
                    });
                }
            }
        }
    }
    // U2: Mock-to-test ratio check (file level, excluding config mocks)
    const mockRatio = totalTestFunctions > 0 ? totalMocks / totalTestFunctions : 0;
    if (checkRules.includes("U2") && mockRatio > 3.0) {
        violations.push({
            rule: "U2",
            severity: "major",
            file: "repository-wide",
            message: `Mock-to-test ratio too high: ${mockRatio.toFixed(2)} (threshold: 3.0)`,
        });
    }
    // Cross-reference: Find similar compliant tests for each violation
    let crossRefStats = { violations_with_examples: 0, total_examples_found: 0 };
    if (cross_reference && violations.length > 0) {
        logTool(`🔗 [TEST_AUDIT] Cross-referencing ${violations.length} violations against good patterns`);
        // Get unique violation code snippets to search for remediation examples
        const violationsWithSnippets = violations.filter(v => v.code_snippet && v.code_snippet.length > 20);
        // Limit to first 10 violations to avoid too many API calls
        for (const violation of violationsWithSnippets.slice(0, 10)) {
            try {
                const examples = await searchCollection(GOOD_PATTERNS_COLLECTION, violation.code_snippet || "", 3);
                if (examples.length > 0) {
                    violation.remediation_examples = examples.map(e => ({
                        test_id: e.test_id,
                        repo: e.repo,
                        file_path: e.file_path,
                        function_name: e.function_name,
                        similarity: e.similarity,
                    }));
                    crossRefStats.violations_with_examples++;
                    crossRefStats.total_examples_found += examples.length;
                }
            }
            catch (err) {
                // Cross-reference failed for this violation, continue
            }
        }
        logTool(`🔗 [TEST_AUDIT] Found ${crossRefStats.total_examples_found} remediation examples for ${crossRefStats.violations_with_examples} violations`);
    }
    // Calculate score
    const violationsByRule = {};
    for (const v of violations) {
        violationsByRule[v.rule] = (violationsByRule[v.rule] || 0) + 1;
    }
    // Weighted scoring: critical=-15, major=-5, minor=-2
    const criticalPenalty = violations.filter(v => v.severity === "critical").length * 15;
    const majorPenalty = violations.filter(v => v.severity === "major").length * 5;
    const minorPenalty = violations.filter(v => v.severity === "minor").length * 2;
    const totalPenalty = Math.min(criticalPenalty + majorPenalty + minorPenalty, 100);
    const score = Math.max(0, 100 - totalPenalty);
    // Determine grade
    let grade;
    if (score >= 95)
        grade = "EXEMPLARY";
    else if (score >= 85)
        grade = "COMPLIANT";
    else if (score >= 70)
        grade = "ACCEPTABLE";
    else if (score >= 50)
        grade = "NON-COMPLIANT";
    else
        grade = "CRITICAL";
    logTool(`✅ [TEST_AUDIT] Complete: ${repoName} scored ${score}/100 (${grade})`);
    const result = {
        repo: repoName,
        timestamp: new Date().toISOString(),
        score,
        grade,
        summary: {
            files_scanned: testFiles.length,
            test_functions: totalTestFunctions,
            violations: violations.length,
            by_rule: violationsByRule,
        },
        violations: output_format === "summary" ? [] : violations,
        metrics: {
            mock_to_test_ratio: parseFloat(mockRatio.toFixed(2)),
            integration_tests_with_mocks: integrationTestsWithMocks,
            tests_without_assertions: testsWithoutAssertions,
        },
        remediation_priority: Object.entries(violationsByRule)
            .map(([rule, count]) => ({
            rule,
            count,
            impact: rule === "AP4" ? "critical" : rule === "AP1" || rule === "AP3" ? "high" : "medium",
        }))
            .sort((a, b) => {
            const impactOrder = { critical: 0, high: 1, medium: 2 };
            return (impactOrder[a.impact] || 3) - (impactOrder[b.impact] || 3);
        }),
        // Include cross-reference stats if used
        ...(cross_reference && crossRefStats.violations_with_examples > 0 && {
            cross_references: crossRefStats,
        }),
        // Include semantic mode data if used
        ...(use_semantic && {
            semantic_stats: semanticStats,
            manual_review: manualReviewItems.length > 0 ? manualReviewItems : undefined,
        }),
    };
    // Write audit report to Platform Documentation
    try {
        const reportDir = "/Users/kevintoles/POC/Platform Documentation/audits/Test Compliance";
        const dateStr = new Date().toISOString().split("T")[0];
        const reportFile = `${reportDir}/${repoName}_${dateStr}.json`;
        await fs.mkdir(reportDir, { recursive: true });
        await fs.writeFile(reportFile, JSON.stringify(result, null, 2));
        logTool(`📝 [TEST_AUDIT] Report written to ${reportFile}`);
        // Add report path to result
        result.report_path = reportFile;
    }
    catch (err) {
        logError(`[TEST_AUDIT] Failed to write report: ${err}`);
    }
    return result;
}
// =============================================================================
// Semantic Test Compliance - Dual-Net approach using vector search
// Net 1: Compare to GOOD patterns (known compliant tests)
// Net 2: Compare to BAD patterns (known anti-patterns)
// =============================================================================
const QDRANT_URL = process.env.QDRANT_URL || "http://localhost:6333";
const GOOD_PATTERNS_COLLECTION = "test_good_patterns";
const BAD_PATTERNS_COLLECTION = "test_bad_patterns";
// Code Pattern Collections (WBS-CDP7: Dual-Net Code Compliance)
const CODE_GOOD_PATTERNS_COLLECTION = "code_good_patterns";
const CODE_BAD_PATTERNS_COLLECTION = "code_bad_patterns";
async function searchCollection(collection, testContent, topK = 3) {
    try {
        // Get embedding from semantic-search-service (uses sentence-transformers, not LLM)
        const embedResponse = await apiCall("/v1/embed", "POST", { text: testContent.slice(0, 2000) }, SEMANTIC_SEARCH_URL, 10000);
        if (!embedResponse?.embedding) {
            logError(`[SEMANTIC_TEST] Failed to get embedding from semantic-search-service`);
            return [];
        }
        // Query Qdrant directly with the embedding
        const qdrantResponse = await apiCall(`/collections/${collection}/points/search`, "POST", {
            vector: embedResponse.embedding,
            limit: topK,
            with_payload: true,
        }, QDRANT_URL, 10000);
        if (!qdrantResponse?.result) {
            return [];
        }
        return qdrantResponse.result.map((r) => ({
            test_id: r.payload?.test_id || "",
            repo: r.payload?.repo || "",
            file_path: r.payload?.file_path || "",
            function_name: r.payload?.function_name || "",
            compliance_status: r.payload?.compliance_status || "unknown",
            rules_violated: r.payload?.rules_violated || [],
            patterns_detected: r.payload?.patterns_detected || [],
            similarity: r.score || 0,
        }));
    }
    catch (error) {
        logError(`[SEMANTIC_TEST] Search ${collection} failed: ${error}`);
        return [];
    }
}
async function checkDualNetCompliance(testContent, testType) {
    // Cast both nets in parallel
    const [goodMatches, badMatches] = await Promise.all([
        searchCollection(GOOD_PATTERNS_COLLECTION, testContent, 3),
        searchCollection(BAD_PATTERNS_COLLECTION, testContent, 3),
    ]);
    // Calculate weighted scores (top match weighted more heavily)
    const weights = [0.5, 0.3, 0.2]; // Weight decreases for lower-ranked matches
    const goodScore = goodMatches.reduce((sum, m, i) => sum + m.similarity * (weights[i] || 0.1), 0);
    const badScore = badMatches.reduce((sum, m, i) => sum + m.similarity * (weights[i] || 0.1), 0);
    // Thresholds for decision
    const HIGH_THRESHOLD = 0.7;
    const LOW_THRESHOLD = 0.4;
    // Collect rules from bad matches
    const rulesViolated = new Set();
    for (const match of badMatches) {
        if (match.similarity > 0.5) {
            match.rules_violated.forEach(r => rulesViolated.add(r));
        }
    }
    // Dual-net decision logic
    let verdict;
    let reasoning;
    let confidence;
    const topGood = goodMatches[0]?.similarity || 0;
    const topBad = badMatches[0]?.similarity || 0;
    if (topGood >= HIGH_THRESHOLD && topBad < LOW_THRESHOLD) {
        // Strong good match, weak bad match → Compliant
        verdict = "compliant";
        confidence = topGood;
        reasoning = `Strong match to compliant pattern: ${goodMatches[0]?.test_id} (${(topGood * 100).toFixed(1)}%)`;
    }
    else if (topBad >= HIGH_THRESHOLD && topGood < LOW_THRESHOLD) {
        // Strong bad match, weak good match → Violation
        verdict = "violation";
        confidence = topBad;
        reasoning = `Strong match to anti-pattern: ${badMatches[0]?.test_id} (${(topBad * 100).toFixed(1)}%)`;
    }
    else if (topGood >= HIGH_THRESHOLD && topBad >= HIGH_THRESHOLD) {
        // Strong match to BOTH → Manual review needed (conflict)
        verdict = "manual_review";
        confidence = Math.min(topGood, topBad);
        reasoning = `Conflict: matches both good (${(topGood * 100).toFixed(1)}%) and bad (${(topBad * 100).toFixed(1)}%) patterns`;
    }
    else if (topGood < LOW_THRESHOLD && topBad < LOW_THRESHOLD) {
        // No strong matches → Uncertain, fall back to regex
        verdict = "uncertain";
        confidence = Math.max(topGood, topBad);
        reasoning = `No strong matches (good: ${(topGood * 100).toFixed(1)}%, bad: ${(topBad * 100).toFixed(1)}%) - using regex fallback`;
    }
    else {
        // Mixed signals - use score comparison
        if (goodScore > badScore * 1.2) {
            verdict = "compliant";
            confidence = goodScore / (goodScore + badScore);
            reasoning = `Weighted toward good patterns (score: ${goodScore.toFixed(2)} vs ${badScore.toFixed(2)})`;
        }
        else if (badScore > goodScore * 1.2) {
            verdict = "violation";
            confidence = badScore / (goodScore + badScore);
            reasoning = `Weighted toward bad patterns (score: ${badScore.toFixed(2)} vs ${goodScore.toFixed(2)})`;
        }
        else {
            verdict = "manual_review";
            confidence = 0.5;
            reasoning = `Ambiguous: similar scores (good: ${goodScore.toFixed(2)}, bad: ${badScore.toFixed(2)})`;
        }
    }
    return {
        verdict,
        confidence,
        good_matches: goodMatches,
        bad_matches: badMatches,
        good_score: goodScore,
        bad_score: badScore,
        inferred_rules_violated: Array.from(rulesViolated),
        reasoning,
    };
}
// =============================================================================
// WBS-CDP7: Semantic Code Pattern Detection - Dual-Net approach
// Net 1: Compare to GOOD patterns (code_good_patterns - after_code examples)
// Net 2: Compare to BAD patterns (code_bad_patterns - before_code examples)
// =============================================================================
async function searchCodePatternCollection(collection, codeContent, topK = 3) {
    try {
        // Get embedding from semantic-search-service (uses sentence-transformers, not LLM)
        const embedResponse = await apiCall("/v1/embed", "POST", { text: codeContent.slice(0, 2000) }, SEMANTIC_SEARCH_URL, 10000);
        if (!embedResponse?.embedding) {
            logError(`[SEMANTIC_CODE] Failed to get embedding from semantic-search-service`);
            return [];
        }
        // Query Qdrant directly with the embedding
        const qdrantResponse = await apiCall(`/collections/${collection}/points/search`, "POST", {
            vector: embedResponse.embedding,
            limit: topK,
            with_payload: true,
        }, QDRANT_URL, 10000);
        if (!qdrantResponse?.result) {
            return [];
        }
        return qdrantResponse.result.map((r) => ({
            instance_id: r.payload?.instance_id || "",
            repo: r.payload?.repo || "",
            file: r.payload?.file || "",
            function_name: r.payload?.function || undefined,
            rule_id: r.payload?.rule_id || "",
            code_snippet: r.payload?.before_code || r.payload?.after_code || "",
            similarity: r.score || 0,
        }));
    }
    catch (error) {
        logError(`[SEMANTIC_CODE] Search ${collection} failed: ${error}`);
        return [];
    }
}
async function checkCodeDualNet(codeContent, fileContext) {
    // Cast both nets in parallel
    const [goodMatches, badMatches] = await Promise.all([
        searchCodePatternCollection(CODE_GOOD_PATTERNS_COLLECTION, codeContent, 3),
        searchCodePatternCollection(CODE_BAD_PATTERNS_COLLECTION, codeContent, 3),
    ]);
    // Calculate weighted scores (top match weighted more heavily)
    const weights = [0.5, 0.3, 0.2]; // Weight decreases for lower-ranked matches
    const goodScore = goodMatches.reduce((sum, m, i) => sum + m.similarity * (weights[i] || 0.1), 0);
    const badScore = badMatches.reduce((sum, m, i) => sum + m.similarity * (weights[i] || 0.1), 0);
    // Thresholds for decision
    const HIGH_THRESHOLD = 0.7;
    const LOW_THRESHOLD = 0.4;
    // Collect pattern IDs from bad matches
    const patternsDetected = new Set();
    for (const match of badMatches) {
        if (match.similarity > 0.5 && match.rule_id) {
            patternsDetected.add(match.rule_id);
        }
    }
    // Dual-net decision logic
    let verdict;
    let reasoning;
    let confidence;
    const topGood = goodMatches[0]?.similarity || 0;
    const topBad = badMatches[0]?.similarity || 0;
    if (topGood >= HIGH_THRESHOLD && topBad < LOW_THRESHOLD) {
        // Strong good match, weak bad match → Compliant
        verdict = "compliant";
        confidence = topGood;
        reasoning = `Strong match to compliant pattern: ${goodMatches[0]?.instance_id} (${(topGood * 100).toFixed(1)}%)`;
    }
    else if (topBad >= HIGH_THRESHOLD && topGood < LOW_THRESHOLD) {
        // Strong bad match, weak good match → Violation
        verdict = "violation";
        confidence = topBad;
        reasoning = `Strong match to anti-pattern: ${badMatches[0]?.instance_id} (${(topBad * 100).toFixed(1)}%)`;
    }
    else if (topGood >= HIGH_THRESHOLD && topBad >= HIGH_THRESHOLD) {
        // Strong match to BOTH → Manual review needed (conflict)
        verdict = "manual_review";
        confidence = Math.min(topGood, topBad);
        reasoning = `Conflict: matches both good (${(topGood * 100).toFixed(1)}%) and bad (${(topBad * 100).toFixed(1)}%) patterns`;
    }
    else if (topGood < LOW_THRESHOLD && topBad < LOW_THRESHOLD) {
        // No strong matches → Uncertain, fall back to regex
        verdict = "uncertain";
        confidence = Math.max(topGood, topBad);
        reasoning = `No strong matches (good: ${(topGood * 100).toFixed(1)}%, bad: ${(topBad * 100).toFixed(1)}%) - using regex fallback`;
    }
    else {
        // Mixed signals - use score comparison
        if (goodScore > badScore * 1.2) {
            verdict = "compliant";
            confidence = goodScore / (goodScore + badScore);
            reasoning = `Weighted toward good patterns (score: ${goodScore.toFixed(2)} vs ${badScore.toFixed(2)})`;
        }
        else if (badScore > goodScore * 1.2) {
            verdict = "violation";
            confidence = badScore / (goodScore + badScore);
            reasoning = `Weighted toward bad patterns (score: ${badScore.toFixed(2)} vs ${goodScore.toFixed(2)})`;
        }
        else {
            verdict = "manual_review";
            confidence = 0.5;
            reasoning = `Ambiguous: similar scores (good: ${goodScore.toFixed(2)}, bad: ${badScore.toFixed(2)})`;
        }
    }
    return {
        verdict,
        confidence,
        good_matches: goodMatches,
        bad_matches: badMatches,
        good_score: goodScore,
        bad_score: badScore,
        inferred_patterns: Array.from(patternsDetected),
        reasoning,
    };
}
// Common code anti-patterns to detect
const CODE_PATTERNS = [
    {
        id: "S1172",
        name: "Unused parameter",
        category: "maintainability",
        severity: "minor",
        regex: /def\s+\w+\([^)]*\b(\w+)\b[^)]*\):[^}]+(?!.*\b\1\b)/s,
        message: "Parameter declared but not used in function body",
    },
    {
        id: "S3776",
        name: "Cognitive complexity",
        category: "complexity",
        severity: "major",
        // Detect deeply nested if/for/while (simplified check)
        regex: /(?:if|for|while)[^:]+:\s*\n(?:\s{4,})+(?:if|for|while)[^:]+:\s*\n(?:\s{8,})+(?:if|for|while)/,
        message: "High cognitive complexity - deeply nested control structures",
    },
    {
        id: "S1134",
        name: "TODO/FIXME comments",
        category: "maintainability",
        severity: "info",
        regex: /#\s*(?:TODO|FIXME|HACK|XXX):/i,
        message: "Unresolved TODO/FIXME comment",
    },
    {
        id: "S5727",
        name: "Comparison to None",
        category: "reliability",
        severity: "minor",
        regex: /(?:==|!=)\s*None/,
        message: "Use 'is None' or 'is not None' instead of == or !=",
    },
    {
        id: "S1481",
        name: "Unused local variable",
        category: "maintainability",
        severity: "minor",
        regex: /(\w+)\s*=\s*[^=\n]+(?!\n.*\b\1\b)/,
        message: "Local variable assigned but not used",
    },
    {
        id: "S125",
        name: "Commented-out code",
        category: "maintainability",
        severity: "minor",
        regex: /#\s*(?:def|class|if|for|while|return|import)\s+\w+/,
        message: "Commented-out code should be removed",
    },
    {
        id: "HARDCODED_SECRET",
        name: "Hardcoded secret",
        category: "security",
        severity: "blocker",
        regex: /(?:password|secret|api_key|token)\s*=\s*["'][^"']{8,}["']/i,
        message: "Potential hardcoded secret detected",
    },
    {
        id: "BROAD_EXCEPT",
        name: "Broad exception",
        category: "reliability",
        severity: "major",
        regex: /except\s*:\s*$|except\s+Exception\s*:/m,
        message: "Catching broad exceptions hides specific errors",
    },
    {
        id: "EMPTY_EXCEPT",
        name: "Empty except block",
        category: "reliability",
        severity: "critical",
        regex: /except[^:]*:\s*\n\s*pass/,
        message: "Empty except block silently swallows errors",
    },
    {
        id: "PRINT_DEBUG",
        name: "Print statement for debugging",
        category: "maintainability",
        severity: "minor",
        regex: /print\s*\([^)]*(?:debug|test|TODO)/i,
        message: "Debug print statement should use proper logging",
    },
];
async function handleCodePatternAudit(args) {
    const { repo_path, repo_name, pattern_categories = ["all"], severity_threshold = "major", include_remediation = true, use_semantic = false, cross_reference = true, } = args;
    const { basename } = await import("path");
    const repoName = repo_name || basename(repo_path);
    const checkCategories = pattern_categories.includes("all")
        ? ["complexity", "security", "maintainability", "reliability", "performance"]
        : pattern_categories;
    const severityOrder = ["blocker", "critical", "major", "minor", "info"];
    const thresholdIndex = severityOrder.indexOf(severity_threshold);
    const mode = use_semantic ? "SEMANTIC" : "REGEX";
    logTool(`🔍 [CODE_AUDIT] Starting ${mode} audit of ${repoName} at ${repo_path}`);
    // Find all Python source files (not tests)
    const sourceFiles = await readDirRecursive(repo_path, /\.py$/);
    const nonTestFiles = sourceFiles.filter(f => !f.includes("/tests/") && !f.includes("test_"));
    logTool(`📁 [CODE_AUDIT] Found ${nonTestFiles.length} source files`);
    const violations = [];
    const manualReviewItems = [];
    const semanticStats = {
        compliant: 0,
        violation: 0,
        manual_review: 0,
        uncertain: 0,
    };
    // Scan each file for patterns
    for (const filePath of nonTestFiles) {
        const content = await readFileContent(filePath);
        const relativePath = filePath.replace(repo_path, "").replace(/^\//, "");
        const lines = content.split("\n");
        // Extract functions for semantic analysis
        const functionMatches = content.matchAll(/(?:def|async def)\s+(\w+)\s*\([^)]*\):/g);
        const functions = [];
        for (const match of functionMatches) {
            functions.push({ name: match[1], startIndex: match.index || 0 });
        }
        // SEMANTIC MODE: Dual-net analysis for each function
        if (use_semantic && functions.length > 0) {
            for (let i = 0; i < functions.length; i++) {
                const fn = functions[i];
                const nextFn = functions[i + 1];
                const fnEndIndex = nextFn ? nextFn.startIndex : content.length;
                const fnContent = content.slice(fn.startIndex, fnEndIndex);
                const fnLine = content.slice(0, fn.startIndex).split("\n").length;
                // Run dual-net analysis on this function
                const dualNetResult = await checkCodeDualNet(fnContent, relativePath);
                // Track dual-net verdicts
                semanticStats[dualNetResult.verdict]++;
                if (dualNetResult.verdict === "violation" && dualNetResult.confidence > 0.4) {
                    // Clear violation from dual-net analysis
                    for (const patternId of dualNetResult.inferred_patterns) {
                        const pattern = CODE_PATTERNS.find(p => p.id === patternId);
                        if (pattern && checkCategories.includes(pattern.category)) {
                            const violation = {
                                pattern_id: patternId,
                                category: pattern?.category || "maintainability",
                                severity: pattern?.severity || "major",
                                file: relativePath,
                                line: fnLine,
                                message: `[SEMANTIC] ${dualNetResult.reasoning}`,
                                code_snippet: fnContent.slice(0, 200),
                            };
                            // AC-7.4: Add remediation examples from good matches
                            if (dualNetResult.good_matches.length > 0) {
                                violation.remediation_examples = dualNetResult.good_matches.map(m => ({
                                    instance_id: m.instance_id,
                                    repo: m.repo,
                                    file: m.file,
                                    after_code: m.code_snippet,
                                    similarity: m.similarity,
                                }));
                            }
                            violations.push(violation);
                        }
                    }
                    // If no specific patterns inferred, create a general violation
                    if (dualNetResult.inferred_patterns.length === 0) {
                        const violation = {
                            pattern_id: dualNetResult.bad_matches[0]?.rule_id || "SEMANTIC_MATCH",
                            category: "maintainability",
                            severity: "major",
                            file: relativePath,
                            line: fnLine,
                            message: `[SEMANTIC] ${dualNetResult.reasoning}`,
                            code_snippet: fnContent.slice(0, 200),
                        };
                        if (dualNetResult.good_matches.length > 0) {
                            violation.remediation_examples = dualNetResult.good_matches.map(m => ({
                                instance_id: m.instance_id,
                                repo: m.repo,
                                file: m.file,
                                after_code: m.code_snippet,
                                similarity: m.similarity,
                            }));
                        }
                        violations.push(violation);
                    }
                }
                else if (dualNetResult.verdict === "manual_review") {
                    // AC-7.5: Flag for human review - conflicting signals from both nets
                    manualReviewItems.push({
                        file: relativePath,
                        line: fnLine,
                        function: fn.name,
                        reasoning: dualNetResult.reasoning,
                        good_score: dualNetResult.good_score,
                        bad_score: dualNetResult.bad_score,
                        top_good_match: dualNetResult.good_matches[0]?.instance_id || "none",
                        top_bad_match: dualNetResult.bad_matches[0]?.instance_id || "none",
                    });
                }
                else if (dualNetResult.verdict === "uncertain") {
                    // Fall through to regex checks below for this function
                }
                else {
                    // Compliant - skip regex checks for this function
                    continue;
                }
                // If verdict is uncertain, fall through to regex mode
                if (dualNetResult.verdict !== "uncertain") {
                    continue;
                }
            }
        }
        // REGEX MODE: Traditional pattern matching (also used as fallback for uncertain)
        for (const pattern of CODE_PATTERNS) {
            // Filter by category
            if (!checkCategories.includes(pattern.category))
                continue;
            // Filter by severity threshold
            if (severityOrder.indexOf(pattern.severity) > thresholdIndex)
                continue;
            // Check for pattern matches
            const matches = content.matchAll(new RegExp(pattern.regex, "gm"));
            for (const match of matches) {
                const lineNumber = content.slice(0, match.index || 0).split("\n").length;
                const codeLine = lines[lineNumber - 1] || "";
                // Skip if already found by semantic mode at same location
                const alreadyFound = violations.some(v => v.file === relativePath && v.line === lineNumber && v.pattern_id === pattern.id);
                if (alreadyFound)
                    continue;
                violations.push({
                    pattern_id: pattern.id,
                    category: pattern.category,
                    severity: pattern.severity,
                    file: relativePath,
                    line: lineNumber,
                    message: pattern.message,
                    code_snippet: codeLine.trim().slice(0, 100),
                });
            }
        }
    }
    // Cross-reference with knowledge base if enabled
    let crossReferences;
    let crossRefStats = { violations_with_examples: 0, total_examples_found: 0 };
    if (cross_reference && violations.length > 0) {
        logTool(`🔗 [CODE_AUDIT] Cross-referencing ${violations.length} violations against good patterns`);
        // Get violations that need cross-referencing (those without remediation_examples)
        const violationsNeedingRef = violations.filter(v => !v.remediation_examples && v.code_snippet);
        for (const violation of violationsNeedingRef.slice(0, 10)) { // Limit to 10
            try {
                const examples = await searchCodePatternCollection(CODE_GOOD_PATTERNS_COLLECTION, violation.code_snippet || "", 3);
                if (examples.length > 0) {
                    violation.remediation_examples = examples.map(e => ({
                        instance_id: e.instance_id,
                        repo: e.repo,
                        file: e.file,
                        after_code: e.code_snippet,
                        similarity: e.similarity,
                    }));
                    crossRefStats.violations_with_examples++;
                    crossRefStats.total_examples_found += examples.length;
                }
            }
            catch {
                // Cross-reference failed for this violation
            }
        }
        logTool(`🔗 [CODE_AUDIT] Found ${crossRefStats.total_examples_found} remediation examples for ${crossRefStats.violations_with_examples} violations`);
    }
    // Calculate score
    const byCategory = {};
    const bySeverity = {};
    for (const v of violations) {
        byCategory[v.category] = (byCategory[v.category] || 0) + 1;
        bySeverity[v.severity] = (bySeverity[v.severity] || 0) + 1;
    }
    // Weighted penalties
    const blockerPenalty = (bySeverity["blocker"] || 0) * 25;
    const criticalPenalty = (bySeverity["critical"] || 0) * 15;
    const majorPenalty = (bySeverity["major"] || 0) * 5;
    const minorPenalty = (bySeverity["minor"] || 0) * 2;
    const totalPenalty = Math.min(blockerPenalty + criticalPenalty + majorPenalty + minorPenalty, 100);
    const score = Math.max(0, 100 - totalPenalty);
    // Grade
    let grade;
    if (score >= 95)
        grade = "EXEMPLARY";
    else if (score >= 85)
        grade = "COMPLIANT";
    else if (score >= 70)
        grade = "ACCEPTABLE";
    else if (score >= 50)
        grade = "NON-COMPLIANT";
    else
        grade = "CRITICAL";
    logTool(`✅ [CODE_AUDIT] Complete: ${repoName} scored ${score}/100 (${grade})`);
    const result = {
        repo: repoName,
        timestamp: new Date().toISOString(),
        score,
        grade,
        summary: {
            files_scanned: nonTestFiles.length,
            violations: violations.length,
            by_category: byCategory,
            by_severity: bySeverity,
        },
        violations,
        cross_references: crossReferences,
        // Include cross-reference stats if used
        ...(cross_reference && crossRefStats.violations_with_examples > 0 && {
            cross_ref_stats: crossRefStats,
        }),
        // Include semantic mode data if used (AC-7.5)
        ...(use_semantic && {
            semantic_stats: semanticStats,
            manual_review: manualReviewItems.length > 0 ? manualReviewItems : undefined,
        }),
    };
    // Write audit report to Platform Documentation
    try {
        const reportDir = "/Users/kevintoles/POC/Platform Documentation/audits/Code Compliance";
        const dateStr = new Date().toISOString().split("T")[0];
        const reportFile = `${reportDir}/${repoName}_${dateStr}.json`;
        await fs.mkdir(reportDir, { recursive: true });
        await fs.writeFile(reportFile, JSON.stringify(result, null, 2));
        logTool(`📝 [CODE_AUDIT] Report written to ${reportFile}`);
        // Add report path to result
        result.report_path = reportFile;
    }
    catch (err) {
        logError(`[CODE_AUDIT] Failed to write report: ${err}`);
    }
    return result;
}
// =============================================================================
// Agent Function Handlers - Route to ai-agents HTTP API
// =============================================================================
// Map clean tool names to ai-agents API function names
const agentFunctionMap = {
    "extract_structure": "extract-structure",
    "summarize_content": "summarize-content",
    "generate_code": "generate-code",
    "analyze_artifact": "analyze-artifact",
    "validate_against_spec": "validate-against-spec",
    "decompose_task": "decompose-task",
    "synthesize_outputs": "synthesize-outputs",
    "cross_reference": "cross-reference",
};
async function handleAgentFunction(toolName, args) {
    const functionName = agentFunctionMap[toolName];
    if (!functionName) {
        throw new Error(`Unknown agent function: ${toolName}`);
    }
    if (!isServiceHealthy("ai-agents")) {
        return {
            error: "ai-agents service not available",
            status: "waiting",
            message: "The ai-agents service is not running. Start the platform to use this tool.",
            tool: toolName,
            function: functionName,
        };
    }
    try {
        const result = await apiCall(`/v1/functions/${functionName}/run`, "POST", { input: args });
        return result;
    }
    catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        return {
            error: errorMessage,
            tool: toolName,
            function: functionName,
            input: args,
        };
    }
}
const toolHandlers = {
    // Agent Functions (8 core functions)
    "extract_structure": (args) => handleAgentFunction("extract_structure", args),
    "summarize_content": (args) => handleAgentFunction("summarize_content", args),
    "generate_code": (args) => handleAgentFunction("generate_code", args),
    "analyze_artifact": (args) => handleAgentFunction("analyze_artifact", args),
    "validate_against_spec": (args) => handleAgentFunction("validate_against_spec", args),
    "decompose_task": (args) => handleAgentFunction("decompose_task", args),
    "synthesize_outputs": (args) => handleAgentFunction("synthesize_outputs", args),
    "cross_reference": (args) => handleAgentFunction("cross_reference", args),
    // Platform Management
    "ai_agents_health": handleHealthTool,
    "ai_agents_list_functions": handleListFunctions,
    "ai_agents_list_protocols": handleListProtocols,
    "ai_agents_run_function": handleRunFunction,
    "ai_agents_run_protocol": handleRunProtocol,
    "llm_complete": handleLlmComplete,
    "semantic_search": handleSemanticSearch,
    "hybrid_search": handleHybridSearch,
    "code_analyze": handleCodeAnalyze,
    "graph_query": handleGraphQuery,
    "graph_get_neighbors": handleGraphGetNeighbors,
    "textbook_search": handleTextbookSearch,
    "audit_generate_footnotes": handleAuditGenerateFootnotes,
    "audit_validate_citations": handleAuditValidateCitations,
    "cross_reference_full": handleCrossReferenceFull,
    "test_compliance_audit": handleTestComplianceAudit,
    "code_pattern_audit": handleCodePatternAudit,
};
// Tool execution handler - now simplified with dispatch pattern
async function executeTool(name, args) {
    // Check static tool handlers first
    const handler = toolHandlers[name];
    if (handler) {
        return handler(args);
    }
    // Check dynamic function tools (ai_fn_*)
    if (name.startsWith("ai_fn_")) {
        return handleDynamicFunction(name, args);
    }
    // Check dynamic protocol tools (ai_protocol_*)
    if (name.startsWith("ai_protocol_")) {
        return handleDynamicProtocol(name, args);
    }
    throw new Error(`Unknown tool: ${name}`);
}
// Main server setup
// @ts-ignore Server class deprecated but McpServer migration pending
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
        const result = await executeTool(name, args ?? {});
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
// Start server with top-level await
try {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("════════════════════════════════════════════════════════════════");
    console.error("  AI Agents MCP Server v1.0.0 - Kitchen Brigade Dynamic Server  ");
    console.error("════════════════════════════════════════════════════════════════");
    console.error(`  AI Agents:        ${AI_AGENTS_URL}`);
    console.error(`  Inference:        ${INFERENCE_SERVICE_URL}`);
    console.error(`  LLM Gateway:      ${LLM_GATEWAY_URL}`);
    console.error(`  Semantic Search:  ${SEMANTIC_SEARCH_URL}`);
    console.error("════════════════════════════════════════════════════════════════");
    // Start background health monitor - keeps checking until services come up
    startHealthMonitor();
}
catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
}
