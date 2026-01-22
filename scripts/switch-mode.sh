#!/bin/bash
# =============================================================================
# AI Platform MCP Server - Configuration Switcher
# =============================================================================
# Usage: ./switch-mode.sh <hybrid|docker|native>
#
# This script updates ~/.vscode/mcp.json to use the appropriate configuration
# for the selected deployment mode.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MCP_JSON="$HOME/.vscode/mcp.json"
BACKUP_FILE="$MCP_JSON.backup"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    echo "Usage: $0 <mode>"
    echo ""
    echo "Modes:"
    echo "  hybrid  - Infrastructure in Docker, services native (RECOMMENDED)"
    echo "  docker  - Everything in Docker containers"
    echo "  native  - Everything native, no Docker"
    echo ""
    echo "Example:"
    echo "  $0 hybrid"
}

if [ -z "$1" ]; then
    usage
    exit 1
fi

MODE=$1

case $MODE in
    hybrid|docker|native)
        ;;
    *)
        echo -e "${RED}Error: Invalid mode '$MODE'${NC}"
        usage
        exit 1
        ;;
esac

# Create backup
if [ -f "$MCP_JSON" ]; then
    cp "$MCP_JSON" "$BACKUP_FILE"
    echo -e "${YELLOW}Backed up existing config to $BACKUP_FILE${NC}"
fi

# Generate configuration based on mode
if [ "$MODE" == "hybrid" ] || [ "$MODE" == "native" ]; then
    # Hybrid and native use localhost
    cat > "$MCP_JSON" << 'EOF'
{
  "servers": {
    "ai-kitchen-br": {
      "type": "stdio",
      "command": "node",
      "args": ["dist/index.js"],
      "cwd": "/Users/kevintoles/POC/ai-agents-mcp-server",
      "env": {
        "PLATFORM_MODE": "MODE_PLACEHOLDER",
        "AI_AGENTS_URL": "http://localhost:8082",
        "INFERENCE_SERVICE_URL": "http://localhost:8085",
        "LLM_GATEWAY_URL": "http://localhost:8080",
        "LLM_GATEWAY_DEFAULT_MODEL": "gpt-4o",
        "SEMANTIC_SEARCH_URL": "http://localhost:8081",
        "CODE_ORCHESTRATOR_URL": "http://localhost:8083",
        "AUDIT_SERVICE_URL": "http://localhost:8084",
        "NEO4J_HTTP_URL": "http://localhost:7474",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "devpassword"
      }
    }
  }
}
EOF
    # Replace placeholder with actual mode
    sed -i '' "s/MODE_PLACEHOLDER/$MODE/" "$MCP_JSON"
    
elif [ "$MODE" == "docker" ]; then
    # Docker mode uses host.docker.internal for MCP server to reach containers
    cat > "$MCP_JSON" << 'EOF'
{
  "servers": {
    "ai-kitchen-br": {
      "type": "stdio",
      "command": "node",
      "args": ["dist/index.js"],
      "cwd": "/Users/kevintoles/POC/ai-agents-mcp-server",
      "env": {
        "PLATFORM_MODE": "docker",
        "AI_AGENTS_URL": "http://host.docker.internal:8082",
        "INFERENCE_SERVICE_URL": "http://host.docker.internal:8085",
        "LLM_GATEWAY_URL": "http://host.docker.internal:8080",
        "LLM_GATEWAY_DEFAULT_MODEL": "gpt-4o",
        "SEMANTIC_SEARCH_URL": "http://host.docker.internal:8081",
        "CODE_ORCHESTRATOR_URL": "http://host.docker.internal:8083",
        "AUDIT_SERVICE_URL": "http://host.docker.internal:8084",
        "NEO4J_HTTP_URL": "http://host.docker.internal:7474",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "devpassword"
      }
    }
  }
}
EOF
fi

echo -e "${GREEN}✓ MCP configuration updated to '$MODE' mode${NC}"
echo ""
echo "Configuration written to: $MCP_JSON"
echo ""

# Mode-specific instructions
case $MODE in
    hybrid)
        echo "Next steps for HYBRID mode:"
        echo "  1. Start infrastructure: cd /Users/kevintoles/POC/ai-platform-data/docker && docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d"
        echo "  2. Start services natively (see docs)"
        echo "  3. Reload VS Code window: Cmd+Shift+P → 'Developer: Reload Window'"
        ;;
    docker)
        echo "Next steps for DOCKER mode:"
        echo "  1. Start all containers: docker-compose up -d"
        echo "  2. Wait for all services to be healthy"
        echo "  3. Reload VS Code window: Cmd+Shift+P → 'Developer: Reload Window'"
        ;;
    native)
        echo "Next steps for NATIVE mode:"
        echo "  1. Ensure databases (Neo4j, Qdrant, Redis) are running"
        echo "  2. Start all services natively (see docs)"
        echo "  3. Reload VS Code window: Cmd+Shift+P → 'Developer: Reload Window'"
        ;;
esac

echo ""
echo "IMPORTANT: Reload VS Code to pick up the new MCP configuration!"
