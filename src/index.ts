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
      console.error(`✅ ${serviceName} is now ONLINE at ${service.url} (${responseTime}ms)`);
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

    // Only log every 10th failure to reduce noise
    if (service.consecutiveFailures === 1 || service.consecutiveFailures % 10 === 0) {
      console.error(`⏳ ${serviceName} not available (attempt ${service.consecutiveFailures}): ${service.lastError}`);
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

  console.error("🔄 Starting health monitor - will auto-connect when services come online...");

  const runHealthCheck = async () => {
    const results = await checkAllServices();
    const anyHealthy = Object.values(results).some((h) => h);
    const allHealthy = Object.values(results).every((h) => h);

    // Adjust check interval based on health state
    const interval = allHealthy ? HEALTH_CHECK_INTERVAL_HEALTHY_MS : HEALTH_CHECK_INTERVAL_MS;
    healthCheckTimer = setTimeout(runHealthCheck, interval);

    // Refresh cache when services come online
    if (anyHealthy && lastCacheTime === 0) {
      console.error("🔄 Services detected - refreshing tool cache...");
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
      name: `ai_fn_${fn.name.replace(/-/g, "_")}`,
      description: fn.description || `Execute the ${fn.name} agent function`,
      inputSchema,
    });
  }

  // Add dynamic protocol-specific tools
  for (const protocol of cachedProtocols) {
    tools.push({
      name: `ai_protocol_${protocol.id.toLowerCase().replace(/-/g, "_")}`,
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

// Tool execution handler
async function executeTool(
  name: string,
  args: Record<string, unknown>
): Promise<unknown> {
  // Core tools - always available even when services are down
  if (name === "ai_agents_health") {
    // WBS-PS3: Return structured health response with caching
    return await getStructuredHealthResponse();
  }

  if (name === "ai_agents_list_functions") {
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

  if (name === "ai_agents_list_protocols") {
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

  if (name === "ai_agents_run_function") {
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
    return apiCall(`/v1/functions/${function_name}/run`, "POST", {
      input,
      preset,
    });
  }

  if (name === "ai_agents_run_protocol") {
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
    return apiCall(`/v1/protocols/${protocol_id}/run`, "POST", {
      inputs,
      config,
      brigade_override,
    });
  }

  // Dynamic function tools (ai_fn_*)
  if (name.startsWith("ai_fn_")) {
    const fnName = name.replace("ai_fn_", "").replace(/_/g, "-");
    const { input, preset, ...rest } = args as {
      input?: Record<string, unknown>;
      preset?: string;
    };
    return apiCall(`/v1/functions/${fnName}/run`, "POST", {
      input: input || rest,
      preset,
    });
  }

  // Dynamic protocol tools (ai_protocol_*)
  if (name.startsWith("ai_protocol_")) {
    const protocolId = name
      .replace("ai_protocol_", "")
      .toUpperCase()
      .replace(/_/g, "-");
    const { inputs, config, brigade_override } = args as {
      inputs: Record<string, unknown>;
      config?: Record<string, unknown>;
      brigade_override?: Record<string, unknown>;
    };
    return apiCall(`/v1/protocols/${protocolId}/run`, "POST", {
      inputs,
      config,
      brigade_override,
    });
  }

  // LLM Complete with tiered fallback
  if (name === "llm_complete") {
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
      try {
        console.error("Trying Tier 1: Local inference-service...");
        
        // First get the currently loaded model
        let localModel = "qwen3-8b"; // fallback
        try {
          const modelsResponse = await apiCall<{
            data: Array<{ id: string; status: string }>;
          }>("/v1/models", "GET", undefined, INFERENCE_SERVICE_URL, 5000);
          
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
          {
            model: localModel,
            messages,
            max_tokens,
            temperature,
          },
          INFERENCE_SERVICE_URL,
          120000 // Increased timeout for local inference
        );
        return {
          tier: "local",
          model: response.model || localModel,
          content: response.choices[0].message.content,
          usage: response.usage || {},
        };
      } catch (error) {
        console.error(`Tier 1 (local) failed: ${error}`);
      }
    }

    // Tier 2: Try cloud via llm-gateway
    if (model_preference === "auto" || model_preference === "cloud") {
      try {
        console.error("Trying Tier 2: Cloud LLM via llm-gateway...");
        const response = await apiCall<{
          model?: string;
          choices: Array<{ message: { content: string } }>;
          usage?: Record<string, number>;
        }>(
          "/v1/chat/completions",
          "POST",
          {
            model: LLM_GATEWAY_DEFAULT_MODEL,
            messages,
            max_tokens,
            temperature,
          },
          LLM_GATEWAY_URL,
          60000
        );
        return {
          tier: "cloud",
          model: response.model || LLM_GATEWAY_DEFAULT_MODEL,
          content: response.choices[0].message.content,
          usage: response.usage || {},
        };
      } catch (error) {
        console.error(`Tier 2 (cloud) failed: ${error}`);
      }
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

  // Semantic Search
  if (name === "semantic_search") {
    const {
      query,
      collection = "all",
      top_k = 10,
      threshold = 0.7,
    } = args as {
      query: string;
      collection?: string;
      top_k?: number;
      threshold?: number;
    };
    return apiCall(
      "/v1/search",
      "POST",
      { query, collection, top_k, threshold },
      SEMANTIC_SEARCH_URL
    );
  }

  // Hybrid Search
  if (name === "hybrid_search") {
    const {
      query,
      collection = "all",
      top_k = 10,
      semantic_weight = 0.7,
      keyword_weight = 0.3,
    } = args as {
      query: string;
      collection?: string;
      top_k?: number;
      semantic_weight?: number;
      keyword_weight?: number;
    };
    return apiCall(
      "/v1/hybrid-search",
      "POST",
      { query, collection, top_k, semantic_weight, keyword_weight },
      SEMANTIC_SEARCH_URL
    );
  }

  // Code Analyze
  if (name === "code_analyze") {
    const {
      code,
      analysis_type = "all",
      language,
      context,
    } = args as {
      code: string;
      analysis_type?: string;
      language?: string;
      context?: string;
    };
    return apiCall(
      "/v1/analyze",
      "POST",
      { code, analysis_type, language, context },
      CODE_ORCHESTRATOR_URL
    );
  }

  // Graph Query (Cypher)
  if (name === "graph_query") {
    const { cypher, parameters = {} } = args as {
      cypher: string;
      parameters?: Record<string, unknown>;
    };
    
    // Neo4j HTTP API
    const auth = Buffer.from(`${NEO4J_USER}:${NEO4J_PASSWORD}`).toString("base64");
    const response = await fetch(`${NEO4J_HTTP_URL}/db/neo4j/tx/commit`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Basic ${auth}`,
      },
      body: JSON.stringify({
        statements: [{ statement: cypher, parameters }],
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Neo4j error (${response.status}): ${error}`);
    }

    const data = await response.json() as {
      results: Array<{ columns: string[]; data: Array<{ row: unknown[] }> }>;
      errors: Array<{ message: string }>;
    };
    
    if (data.errors && data.errors.length > 0) {
      throw new Error(`Cypher error: ${data.errors[0].message}`);
    }

    // Transform to more readable format
    const result = data.results[0];
    if (!result) return { rows: [], columns: [] };
    
    return {
      columns: result.columns,
      rows: result.data.map((d) => {
        const row: Record<string, unknown> = {};
        result.columns.forEach((col, i) => {
          row[col] = d.row[i];
        });
        return row;
      }),
      count: result.data.length,
    };
  }

  // Graph Get Neighbors
  if (name === "graph_get_neighbors") {
    const {
      node_id,
      node_type,
      relationship_type,
      depth = 1,
    } = args as {
      node_id: string;
      node_type?: string;
      relationship_type?: string;
      depth?: number;
    };

    // Build Cypher query
    const nodeMatch = node_type 
      ? `(n:${node_type} {name: $node_id})`
      : `(n {name: $node_id})`;
    const relMatch = relationship_type 
      ? `-[r:${relationship_type}*1..${depth}]-`
      : `-[r*1..${depth}]-`;
    
    const cypher = `MATCH ${nodeMatch}${relMatch}(m) RETURN DISTINCT n, type(r[0]) as relationship, m LIMIT 50`;

    const auth = Buffer.from(`${NEO4J_USER}:${NEO4J_PASSWORD}`).toString("base64");
    const response = await fetch(`${NEO4J_HTTP_URL}/db/neo4j/tx/commit`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Basic ${auth}`,
      },
      body: JSON.stringify({
        statements: [{ statement: cypher, parameters: { node_id } }],
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Neo4j error (${response.status}): ${error}`);
    }

    const data = await response.json() as {
      results: Array<{ columns: string[]; data: Array<{ row: unknown[] }> }>;
      errors: Array<{ message: string }>;
    };
    
    if (data.errors && data.errors.length > 0) {
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

  // Textbook Search
  if (name === "textbook_search") {
    const {
      query,
      books,
      top_k = 5,
    } = args as {
      query: string;
      books?: string[];
      top_k?: number;
    };
    // Route through semantic-search which indexes textbooks in 'chapters' collection
    // Note: 'chapters' collection contains indexed book chapters from ai-platform-data
    return apiCall(
      "/v1/search",
      "POST",
      { 
        query, 
        collection: "chapters",  // Collection name in Qdrant
        limit: top_k,  // API uses 'limit' not 'top_k'
      },
      SEMANTIC_SEARCH_URL
    );
  }

  // Audit Generate Footnotes
  if (name === "audit_generate_footnotes") {
    const { citations, task_id } = args as {
      citations: Array<{ marker: string; source_id: string; source_type: string }>;
      task_id?: string;
    };
    return apiCall(
      "/v1/footnotes",
      "POST",
      { citations, task_id },
      AUDIT_SERVICE_URL
    );
  }

  // Audit Validate Citations
  if (name === "audit_validate_citations") {
    const { content, citations } = args as {
      content: string;
      citations: Array<{ marker: string; source_path: string; claimed_content: string }>;
    };
    return apiCall(
      "/v1/validate",
      "POST",
      { content, citations },
      AUDIT_SERVICE_URL
    );
  }

  // Full Cross-Reference Pipeline (Stage 2)
  if (name === "cross_reference_full") {
    const {
      query,
      sources = ["qdrant", "neo4j", "textbooks", "code", "code_chunks"],
      top_k = 5,
      merge_strategy = "by_relevance",
    } = args as {
      query: string;
      sources?: string[];
      top_k?: number;
      merge_strategy?: string;
    };

    // Execute parallel searches across all requested sources
    const searchPromises: Promise<{ source: string; results: unknown }>[] = [];

    if (sources.includes("qdrant")) {
      searchPromises.push(
        apiCall<unknown>("/v1/search", "POST", { query, collection: "all", limit: top_k }, SEMANTIC_SEARCH_URL)
          .then(results => ({ source: "qdrant", results }))
          .catch(err => ({ source: "qdrant", results: { error: String(err) } }))
      );
    }

    if (sources.includes("neo4j")) {
      const cypher = `
        CALL db.index.fulltext.queryNodes("concept_search", $query) 
        YIELD node, score 
        RETURN node, score 
        ORDER BY score DESC 
        LIMIT $top_k
      `;
      const auth = Buffer.from(`${NEO4J_USER}:${NEO4J_PASSWORD}`).toString("base64");
      searchPromises.push(
        fetch(`${NEO4J_HTTP_URL}/db/neo4j/tx/commit`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Basic ${auth}`,
          },
          body: JSON.stringify({
            statements: [{ statement: cypher, parameters: { query, top_k } }],
          }),
        })
          .then(r => r.json())
          .then(data => ({ source: "neo4j", results: data }))
          .catch(err => ({ source: "neo4j", results: { error: String(err) } }))
      );
    }

    if (sources.includes("textbooks")) {
      // Search 'chapters' collection which contains indexed book chapters from textbooks
      searchPromises.push(
        apiCall<unknown>("/v1/search", "POST", { query, collection: "chapters", limit: top_k }, SEMANTIC_SEARCH_URL)
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

    // NEW: Search code_chunks collection for actual GitHub code snippets
    if (sources.includes("code_chunks")) {
      searchPromises.push(
        apiCall<unknown>("/v1/search", "POST", { query, collection: "code_chunks", limit: top_k }, SEMANTIC_SEARCH_URL)
          .then(results => ({ source: "code_chunks", results }))
          .catch(err => ({ source: "code_chunks", results: { error: String(err) } }))
      );
    }

    // Wait for all searches in parallel
    const allResults = await Promise.all(searchPromises);

    // Merge results based on strategy
    return {
      query,
      sources_searched: sources,
      merge_strategy,
      results: allResults,
      total_sources: allResults.length,
    };
  }

  throw new Error(`Unknown tool: ${name}`);
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
    const result = await executeTool(name, (args || {}) as Record<string, unknown>);
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

// Start server
async function main() {
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

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
