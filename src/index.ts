#!/usr/bin/env node

// Server is deprecated in favor of McpServer, but McpServer requires significant refactoring
// NOSONAR: Suppressing deprecation warning until migration to McpServer is complete
// @ts-expect-error Server class deprecated but McpServer migration pending
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

function logInfo(message: string): void {
  if (LOG_LEVEL === "info" || LOG_LEVEL === "debug") {
    console.error(message);
  }
}

function logDebug(message: string): void {
  if (LOG_LEVEL === "debug") {
    console.error(message);
  }
}

function logError(message: string): void {
  console.error(message);
}

function logTool(message: string): void {
  // Tool execution always logged
  console.error(message);
}

// =============================================================================
// Service Health Tracking - Auto-retry until platform starts
// WBS-PS3: Enhanced health instrumentation with caching and metrics
// =============================================================================

interface ServiceHealth {
  url: string;
  healthy: boolean;
  lastCheck: number;
  lastError?: string;
  consecutiveFailures: number;
  // WBS-PS3: New fields for enhanced health tracking
  lastSuccessTime: number;           // Last successful health check timestamp
  totalRequests: number;             // Total requests made to this service
  totalErrors: number;               // Total errors across all requests
  errorRate: number;                 // Rolling error rate (errors / total)
  responseTimeMs?: number;           // Last response time in ms
  avgResponseTimeMs: number;         // Rolling average response time
  healthDetails?: Record<string, unknown>; // Response from /health endpoint
}

// WBS-PS3: Health cache configuration
const HEALTH_CACHE_TTL_MS = 10000;   // Cache health status for 10 seconds
let healthCacheTime = 0;             // Last time health was cached
let cachedHealthResponse: StructuredHealthResponse | null = null;

interface ServiceHealthStatus {
  healthy: boolean;
  url: string;
  lastCheck: string;
  lastSuccess: string | null;
  lastError?: string;
  consecutiveFailures: number;
  totalRequests: number;
  totalErrors: number;
  errorRate: string;
  responseTimeMs?: number;
  avgResponseTimeMs: number;
  healthDetails?: Record<string, unknown>;
}

interface StructuredHealthResponse {
  status: "healthy" | "degraded" | "unhealthy" | "waiting";
  timestamp: string;
  cached: boolean;
  cacheAgeMs?: number;
  summary: {
    total: number;
    healthy: number;
    unhealthy: number;
    healthPercentage: string;
  };
  services: Record<string, ServiceHealthStatus>;
  message?: string;
  hint?: string;
}

const serviceHealth: Record<string, ServiceHealth> = {
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
let healthCheckTimer: NodeJS.Timeout | null = null;

// WBS-PS3: Rolling average window for response times
const ROLLING_AVERAGE_ALPHA = 0.2; // Exponential moving average factor

async function checkServiceHealth(serviceName: string): Promise<boolean> {
  const service = serviceHealth[serviceName];
  if (!service) return false;

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
      } catch {
        // Response might not be JSON, that's okay
        service.healthDetails = { raw: "ok" };
      }
    }

    // Update error rate (rolling)
    service.errorRate = service.totalErrors / service.totalRequests;

    if (!wasHealthy && service.healthy) {
      logError(`✅ ${serviceName} is now ONLINE at ${service.url} (${responseTime}ms)`);
    } else {
      logDebug(`[health] ${serviceName} healthy (${responseTime}ms)`);
    }

    return service.healthy;
  } catch (error) {
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
    } else if (service.consecutiveFailures % 10 === 0) {
      logInfo(`⏳ ${serviceName} still offline (attempt ${service.consecutiveFailures})`);
    }

    return false;
  }
}

async function checkAllServices(): Promise<Record<string, boolean>> {
  const results: Record<string, boolean> = {};
  
  await Promise.all(
    Object.keys(serviceHealth).map(async (name) => {
      results[name] = await checkServiceHealth(name);
    })
  );

  return results;
}

function startHealthMonitor(): void {
  if (healthCheckTimer) return;

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

function getServiceStatus(): Record<string, { healthy: boolean; url: string; lastError?: string }> {
  const status: Record<string, { healthy: boolean; url: string; lastError?: string }> = {};
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
function buildStructuredHealthResponse(fromCache: boolean = false): StructuredHealthResponse {
  const now = Date.now();
  const services: Record<string, ServiceHealthStatus> = {};
  let healthyCount = 0;
  const totalCount = Object.keys(serviceHealth).length;

  for (const [name, health] of Object.entries(serviceHealth)) {
    if (health.healthy) healthyCount++;

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
  let status: StructuredHealthResponse["status"];
  let message: string;

  if (healthyCount === totalCount) {
    status = "healthy";
    message = "All services operational";
  } else if (healthyCount > 0) {
    status = "degraded";
    message = `${healthyCount}/${totalCount} services operational`;
  } else if (Object.values(serviceHealth).some(s => s.totalRequests > 0)) {
    status = "unhealthy";
    message = "All services unavailable";
  } else {
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
async function getStructuredHealthResponse(): Promise<StructuredHealthResponse> {
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

function isServiceHealthy(serviceName: string): boolean {
  return serviceHealth[serviceName]?.healthy ?? false;
}

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
    logInfo(`📦 Cache refreshed: ${cachedFunctions.length} functions, ${cachedProtocols.length} protocols`);
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
      description: "Check health status of AI Agents service and all Kitchen Brigade dependencies. Returns structured JSON with per-service details including: healthy/unhealthy status, last successful request timestamp, error counts/rates, response times, and cached health details. Health status is cached for 10 seconds to avoid hammering services.",
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
    // Textbook Search - Direct access to textbook JSON files
    {
      name: "textbook_search",
      description:
        "Search through textbook JSON files for relevant passages. This searches your 256 indexed textbooks including AI Agents in Action, Building Microservices, etc.",
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
      description:
        "Generate Chicago-style footnotes from citation markers. Use after cross-reference to format citations properly.",
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
      description:
        "Validate that cited sources actually exist and contain the claimed content. Use for hallucination detection.",
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
      description:
        "Execute complete Stage 2 cross-reference: parallel search across Qdrant (vectors), Neo4j (graph), Textbooks (JSON), Code-Reference-Engine, and Code Chunks (actual GitHub code). Returns unified results with relevance scores.",
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

// =============================================================================
// Tool Handlers - Each handler is a focused function to reduce complexity
// =============================================================================

async function handleHealthTool(): Promise<unknown> {
  return await getStructuredHealthResponse();
}

async function handleListFunctions(): Promise<unknown> {
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

async function handleListProtocols(): Promise<unknown> {
  if (!isServiceHealthy("ai-agents")) {
    return {
      error: "ai-agents service not available",
      status: "waiting",
      message: "The ai-agents service is not running. Start the platform to use this tool.",
      cached_protocols: cachedProtocols.length > 0 ? cachedProtocols.map((p) => p.id) : undefined,
    };
  }
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
  if (!isServiceHealthy("ai-agents")) {
    return {
      error: "ai-agents service not available",
      status: "waiting",
      message: "Cannot execute functions - ai-agents service is not running.",
      requested_function: args.function_name,
      hint: "Start the platform with: docker-compose up -d",
    };
  }
  const { function_name, input, preset } = args as {
    function_name: string;
    input: Record<string, unknown>;
    preset?: string;
  };
  return apiCall(`/v1/functions/${function_name}/run`, "POST", { input, preset });
}

// Helper to get preflight hint message
function getPreflightHint(blockingIssues: string[]): string {
  if (blockingIssues.some((i: string) => i.includes("inference-service"))) {
    return "Start inference-service: cd inference-service && source .venv/bin/activate && python -m uvicorn src.main:app --port 8085";
  }
  if (blockingIssues.some((i: string) => i.includes("not found"))) {
    return "Required model not available in inference-service";
  }
  return "Check service health with ai_agents_health tool";
}

// Helper to run preflight check
async function runPreflightCheck(
  protocolId: string,
  brigadeOverride: Record<string, unknown> | undefined,
  enableCrossReference: boolean
): Promise<{ ready: boolean; result?: unknown }> {
  try {
    const preflightResult = await apiCall<{
      ready: boolean;
      blocking_issues: string[];
      warnings: string[];
      services?: Array<{ name: string; healthy: boolean; error?: string }>;
      models?: Array<{ model_id: string; status: string; required_by: string[] }>;
      check_time_ms: number;
    }>(
      `/v1/protocols/${protocolId}/preflight`,
      "POST",
      { brigade_override: brigadeOverride, enable_cross_reference: enableCrossReference },
      AI_AGENTS_URL,
      5000
    );
    
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
  } catch (preflightError) {
    logError(`⚠️ [PREFLIGHT] Check failed: ${preflightError}. Proceeding with caution...`);
    return { ready: true }; // Proceed anyway if preflight fails
  }
}

async function handleRunProtocol(args: Record<string, unknown>): Promise<unknown> {
  if (!isServiceHealthy("ai-agents")) {
    return {
      error: "ai-agents service not available",
      status: "waiting",
      message: "Cannot execute protocols - ai-agents service is not running.",
      requested_protocol: args.protocol_id,
      hint: "Start the platform with: docker-compose up -d",
    };
  }
  
  const { protocol_id, inputs, config, brigade_override } = args as {
    protocol_id: string;
    inputs: Record<string, unknown>;
    config?: Record<string, unknown>;
    brigade_override?: Record<string, unknown>;
  };
  
  logTool(`🔍 [PREFLIGHT] Checking prerequisites for protocol ${protocol_id}...`);
  const preflight = await runPreflightCheck(protocol_id, brigade_override, config?.run_cross_reference !== false);
  if (!preflight.ready) {
    return preflight.result;
  }
  
  logTool(`🚀 [PROTOCOL] Executing ${protocol_id}...`);
  return apiCall(
    `/v1/protocols/${protocol_id}/run`,
    "POST",
    { inputs, config, brigade_override },
    AI_AGENTS_URL,
    300000
  );
}

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
  
  console.error(`[PREFLIGHT] Checking prerequisites for protocol ${protocolId}...`);
  const preflight = await runPreflightCheck(protocolId, brigade_override, config?.run_cross_reference !== false);
  if (!preflight.ready) {
    return preflight.result;
  }
  
  return apiCall(
    `/v1/protocols/${protocolId}/run`,
    "POST",
    { inputs, config, brigade_override },
    AI_AGENTS_URL,
    300000
  );
}

// Helper to try local inference
async function tryLocalInference(
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
  temperature: number
): Promise<{ success: boolean; result?: unknown }> {
  try {
    console.error("Trying Tier 1: Local inference-service...");
    let localModel = "qwen3-8b";
    
    try {
      const modelsResponse = await apiCall<{ data: Array<{ id: string; status: string }> }>(
        "/v1/models", "GET", undefined, INFERENCE_SERVICE_URL, 5000
      );
      const loadedModels = modelsResponse.data.filter((m) => m.status === "loaded");
      if (loadedModels.length > 0) {
        localModel = loadedModels[0].id;
        console.error(`Using loaded model: ${localModel}`);
      }
    } catch {
      console.error("Could not query models, using default");
    }
    
    const response = await apiCall<{
      model?: string;
      choices: Array<{ message: { content: string } }>;
      usage?: Record<string, number>;
    }>(
      "/v1/chat/completions",
      "POST",
      { model: localModel, messages, max_tokens: maxTokens, temperature },
      INFERENCE_SERVICE_URL,
      120000
    );
    
    return {
      success: true,
      result: {
        tier: "local",
        model: response.model || localModel,
        content: response.choices[0].message.content,
        usage: response.usage || {},
      },
    };
  } catch (error) {
    console.error(`Tier 1 (local) failed: ${error}`);
    return { success: false };
  }
}

// Helper to try cloud inference
async function tryCloudInference(
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
  temperature: number
): Promise<{ success: boolean; result?: unknown }> {
  try {
    console.error("Trying Tier 2: Cloud LLM via llm-gateway...");
    const response = await apiCall<{
      model?: string;
      choices: Array<{ message: { content: string } }>;
      usage?: Record<string, number>;
    }>(
      "/v1/chat/completions",
      "POST",
      { model: LLM_GATEWAY_DEFAULT_MODEL, messages, max_tokens: maxTokens, temperature },
      LLM_GATEWAY_URL,
      60000
    );
    
    return {
      success: true,
      result: {
        tier: "cloud",
        model: response.model || LLM_GATEWAY_DEFAULT_MODEL,
        content: response.choices[0].message.content,
        usage: response.usage || {},
      },
    };
  } catch (error) {
    console.error(`Tier 2 (cloud) failed: ${error}`);
    return { success: false };
  }
}

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

  if (model_preference === "auto" || model_preference === "local") {
    const local = await tryLocalInference(messages, max_tokens, temperature);
    if (local.success) return local.result;
  }

  if (model_preference === "auto" || model_preference === "cloud") {
    const cloud = await tryCloudInference(messages, max_tokens, temperature);
    if (cloud.success) return cloud.result;
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

// Helper to execute Neo4j query
async function executeNeo4jQuery(
  cypher: string,
  parameters: Record<string, unknown>
): Promise<{ results: Array<{ columns: string[]; data: Array<{ row: unknown[] }> }>; errors: Array<{ message: string }> }> {
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

  return response.json() as Promise<{ results: Array<{ columns: string[]; data: Array<{ row: unknown[] }> }>; errors: Array<{ message: string }> }>;
}

async function handleGraphQuery(args: Record<string, unknown>): Promise<unknown> {
  const { cypher, parameters = {} } = args as { cypher: string; parameters?: Record<string, unknown> };
  
  const data = await executeNeo4jQuery(cypher, parameters);
  
  if (data.errors && data.errors.length > 0) {
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

  const data = await executeNeo4jQuery(cypher, { node_id });
  
  if (data.errors && data.errors.length > 0) {
    throw new Error(`Cypher error: ${data.errors[0].message}`);
  }

  const result = data.results[0];
  if (!result) return { source: node_id, neighbors: [] };

  return {
    source: node_id,
    neighbors: result.data.map((d) => ({ relationship: d.row[1], node: d.row[2] })),
    count: result.data.length,
  };
}

async function handleTextbookSearch(args: Record<string, unknown>): Promise<unknown> {
  const { query, top_k = 5 } = args as { query: string; books?: string[]; top_k?: number };
  return apiCall("/v1/search", "POST", { query, collection: "chapters", limit: top_k }, SEMANTIC_SEARCH_URL);
}

async function handleAuditGenerateFootnotes(args: Record<string, unknown>): Promise<unknown> {
  const { citations, task_id } = args as {
    citations: Array<{ marker: string; source_id: string; source_type: string }>;
    task_id?: string;
  };
  return apiCall("/v1/footnotes", "POST", { citations, task_id }, AUDIT_SERVICE_URL);
}

async function handleAuditValidateCitations(args: Record<string, unknown>): Promise<unknown> {
  const { content, citations } = args as {
    content: string;
    citations: Array<{ marker: string; source_path: string; claimed_content: string }>;
  };
  return apiCall("/v1/validate", "POST", { content, citations }, AUDIT_SERVICE_URL);
}

// Helper to build cross-reference search promises
function buildCrossRefSearchPromises(
  query: string,
  sources: string[],
  topK: number
): Promise<{ source: string; results: unknown }>[] {
  const searchPromises: Promise<{ source: string; results: unknown }>[] = [];

  if (sources.includes("qdrant")) {
    searchPromises.push(
      apiCall<unknown>("/v1/search", "POST", { query, collection: "all", limit: topK }, SEMANTIC_SEARCH_URL)
        .then(results => ({ source: "qdrant", results }))
        .catch(err => ({ source: "qdrant", results: { error: String(err) } }))
    );
  }

  if (sources.includes("neo4j")) {
    const cypher = `CALL db.index.fulltext.queryNodes("concept_search", $query) YIELD node, score RETURN node, score ORDER BY score DESC LIMIT $top_k`;
    const auth = Buffer.from(`${NEO4J_USER}:${NEO4J_PASSWORD}`).toString("base64");
    searchPromises.push(
      fetch(`${NEO4J_HTTP_URL}/db/neo4j/tx/commit`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Basic ${auth}` },
        body: JSON.stringify({ statements: [{ statement: cypher, parameters: { query, top_k: topK } }] }),
      })
        .then(r => r.json())
        .then(data => ({ source: "neo4j", results: data }))
        .catch(err => ({ source: "neo4j", results: { error: String(err) } }))
    );
  }

  if (sources.includes("textbooks")) {
    searchPromises.push(
      apiCall<unknown>("/v1/search", "POST", { query, collection: "chapters", limit: topK }, SEMANTIC_SEARCH_URL)
        .then(results => ({ source: "textbooks", results }))
        .catch(err => ({ source: "textbooks", results: { error: String(err) } }))
    );
  }

  if (sources.includes("code")) {
    searchPromises.push(
      apiCall<unknown>("/v1/search", "POST", { query, analysis_type: "semantic" }, CODE_ORCHESTRATOR_URL)
        .then(results => ({ source: "code", results }))
        .catch(err => ({ source: "code", results: { error: String(err) } }))
    );
  }

  if (sources.includes("code_chunks")) {
    searchPromises.push(
      apiCall<unknown>("/v1/search", "POST", { query, collection: "code_chunks", limit: topK }, SEMANTIC_SEARCH_URL)
        .then(results => ({ source: "code_chunks", results }))
        .catch(err => ({ source: "code_chunks", results: { error: String(err) } }))
    );
  }

  return searchPromises;
}

async function handleCrossReferenceFull(args: Record<string, unknown>): Promise<unknown> {
  const {
    query,
    sources = ["qdrant", "neo4j", "textbooks", "code", "code_chunks"],
    top_k = 5,
    merge_strategy = "by_relevance",
  } = args as { query: string; sources?: string[]; top_k?: number; merge_strategy?: string };

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

// =============================================================================
// Tool Dispatch Map - Routes tool names to their handlers
// =============================================================================
type ToolHandler = (args: Record<string, unknown>) => Promise<unknown>;

const toolHandlers: Record<string, ToolHandler> = {
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
};

// Tool execution handler - now simplified with dispatch pattern
async function executeTool(
  name: string,
  args: Record<string, unknown>
): Promise<unknown> {
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
// @ts-expect-error Server class deprecated but McpServer migration pending
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
    const result = await executeTool(name, args ?? {});
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
} catch (error) {
  console.error("Fatal error:", error);
  process.exit(1);
}
