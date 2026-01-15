/**
 * Configuration types for AI Agents MCP Server.
 *
 * WBS-RV8: MCP Server Config Updates
 *
 * @module types/config
 */

/**
 * Configuration for relevance validation pipeline (Stage 0.x).
 *
 * Controls the protocol-layer validation that occurs before LLM responses
 * are finalized. When enabled, cross-reference results are rated by all 4
 * brigade LLMs in parallel, with optional discussion rounds for resolving
 * disagreements.
 *
 * @see RELEVANCE_VALIDATION.json protocol definition
 */
export interface RelevanceValidationConfig {
  /**
   * Enable or disable the relevance validation pipeline.
   * When disabled, cross-reference results pass through without validation.
   * @default true
   */
  enabled: boolean;

  /**
   * Score variance threshold that triggers a mini-discussion.
   * When variance across the 4 LLM ratings exceeds this threshold,
   * Stage 0.2 (conditional_discussion) is executed.
   * @default 2.0
   */
  variance_threshold: number;

  /**
   * Maximum number of discussion rounds before forcing a decision.
   * Discussions continue until consensus is reached or max_rounds is hit.
   * @default 3
   */
  max_discussion_rounds: number;

  /**
   * Whether to inject engineering patterns (best practices, anti-patterns)
   * into LLM prompts during relevance rating and discussion.
   * @default true
   */
  pattern_bundle_enabled: boolean;
}

/**
 * Default configuration values for relevance validation.
 * These match the config_defaults in RELEVANCE_VALIDATION.json.
 */
export const DEFAULT_RELEVANCE_CONFIG: RelevanceValidationConfig = {
  enabled: true,
  variance_threshold: 2.0,
  max_discussion_rounds: 3,
  pattern_bundle_enabled: true,
};

/**
 * JSON Schema definition for RelevanceValidationConfig.
 * Used in MCP tool inputSchema definitions.
 */
export const RELEVANCE_VALIDATION_SCHEMA = {
  type: "object",
  description:
    "Relevance validation configuration (Stage 0.x). Controls parallel LLM rating and conditional discussion for cross-reference results.",
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
} as const;
