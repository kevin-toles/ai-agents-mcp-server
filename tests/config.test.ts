/**
 * Tests for relevance validation configuration.
 *
 * WBS-RV8: MCP Server Config Updates
 * RED tests for AC-RV8.1 through AC-RV8.4
 *
 * Acceptance Criteria:
 * - AC-RV8.1: run_protocol inputSchema includes relevance_validation config
 * - AC-RV8.2: Config has enabled, variance_threshold, max_discussion_rounds
 * - AC-RV8.3: Default values match protocol JSON config_defaults
 * - AC-RV8.4: TypeScript types generated for config
 */
import { describe, it, expect } from "vitest";
import {
  RelevanceValidationConfig,
  DEFAULT_RELEVANCE_CONFIG,
  RELEVANCE_VALIDATION_SCHEMA,
} from "../src/types/config.js";

// =============================================================================
// AC-RV8.2: Config has enabled, variance_threshold, max_discussion_rounds
// =============================================================================

describe("RelevanceValidationConfig interface", () => {
  it("should have enabled property", () => {
    const config: RelevanceValidationConfig = {
      enabled: true,
      variance_threshold: 2.0,
      max_discussion_rounds: 3,
      pattern_bundle_enabled: true,
    };
    expect(config.enabled).toBe(true);
  });

  it("should have variance_threshold property", () => {
    const config: RelevanceValidationConfig = {
      enabled: true,
      variance_threshold: 1.5,
      max_discussion_rounds: 3,
      pattern_bundle_enabled: true,
    };
    expect(config.variance_threshold).toBe(1.5);
  });

  it("should have max_discussion_rounds property", () => {
    const config: RelevanceValidationConfig = {
      enabled: true,
      variance_threshold: 2.0,
      max_discussion_rounds: 5,
      pattern_bundle_enabled: true,
    };
    expect(config.max_discussion_rounds).toBe(5);
  });

  it("should have pattern_bundle_enabled property", () => {
    const config: RelevanceValidationConfig = {
      enabled: true,
      variance_threshold: 2.0,
      max_discussion_rounds: 3,
      pattern_bundle_enabled: false,
    };
    expect(config.pattern_bundle_enabled).toBe(false);
  });
});

// =============================================================================
// AC-RV8.3: Default values match protocol JSON config_defaults
// =============================================================================

describe("DEFAULT_RELEVANCE_CONFIG", () => {
  it("should have enabled default to true", () => {
    expect(DEFAULT_RELEVANCE_CONFIG.enabled).toBe(true);
  });

  it("should have variance_threshold default to 2.0", () => {
    expect(DEFAULT_RELEVANCE_CONFIG.variance_threshold).toBe(2.0);
  });

  it("should have max_discussion_rounds default to 3", () => {
    expect(DEFAULT_RELEVANCE_CONFIG.max_discussion_rounds).toBe(3);
  });

  it("should have pattern_bundle_enabled default to true", () => {
    expect(DEFAULT_RELEVANCE_CONFIG.pattern_bundle_enabled).toBe(true);
  });

  it("should match protocol JSON config_defaults", () => {
    // Values from RELEVANCE_VALIDATION.json config_defaults
    const protocolDefaults = {
      variance_threshold: 2.0,
      max_discussion_rounds: 3,
      pattern_bundle_enabled: true,
    };

    expect(DEFAULT_RELEVANCE_CONFIG.variance_threshold).toBe(
      protocolDefaults.variance_threshold
    );
    expect(DEFAULT_RELEVANCE_CONFIG.max_discussion_rounds).toBe(
      protocolDefaults.max_discussion_rounds
    );
    expect(DEFAULT_RELEVANCE_CONFIG.pattern_bundle_enabled).toBe(
      protocolDefaults.pattern_bundle_enabled
    );
  });
});

// =============================================================================
// AC-RV8.1: run_protocol inputSchema includes relevance_validation config
// =============================================================================

describe("RELEVANCE_VALIDATION_SCHEMA", () => {
  it("should be an object type", () => {
    expect(RELEVANCE_VALIDATION_SCHEMA.type).toBe("object");
  });

  it("should have description", () => {
    expect(RELEVANCE_VALIDATION_SCHEMA.description).toBeDefined();
    expect(RELEVANCE_VALIDATION_SCHEMA.description.length).toBeGreaterThan(10);
  });

  it("should have enabled property in schema", () => {
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.enabled).toBeDefined();
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.enabled.type).toBe("boolean");
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.enabled.default).toBe(true);
  });

  it("should have variance_threshold property in schema", () => {
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.variance_threshold).toBeDefined();
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.variance_threshold.type).toBe("number");
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.variance_threshold.default).toBe(2.0);
  });

  it("should have max_discussion_rounds property in schema", () => {
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.max_discussion_rounds).toBeDefined();
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.max_discussion_rounds.type).toBe("number");
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.max_discussion_rounds.default).toBe(3);
  });

  it("should have pattern_bundle_enabled property in schema", () => {
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.pattern_bundle_enabled).toBeDefined();
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.pattern_bundle_enabled.type).toBe("boolean");
    expect(RELEVANCE_VALIDATION_SCHEMA.properties.pattern_bundle_enabled.default).toBe(true);
  });
});

// =============================================================================
// AC-RV8.4: TypeScript types generated for config
// =============================================================================

describe("TypeScript type safety", () => {
  it("should enforce all required properties", () => {
    // This test verifies TypeScript compilation
    const validConfig: RelevanceValidationConfig = {
      enabled: false,
      variance_threshold: 1.0,
      max_discussion_rounds: 2,
      pattern_bundle_enabled: false,
    };

    // All properties accessible and correctly typed
    expect(typeof validConfig.enabled).toBe("boolean");
    expect(typeof validConfig.variance_threshold).toBe("number");
    expect(typeof validConfig.max_discussion_rounds).toBe("number");
    expect(typeof validConfig.pattern_bundle_enabled).toBe("boolean");
  });
});
