#!/bin/bash
# Gollek - Unified Installation Script
# This script prepares the environment, ensures Java 25 is present, and installs the Gollek CLI.
# Usage: curl -fsSL https://raw.githubusercontent.com/bhangun/gollek/main/install.sh | bash

set -e

# --- Configuration ---
GOLLEK_REPO="bhangun/gollek"
GOLLEK_BRANCH="main" # Default branch
INSTALL_DIR="${HOME}/.gollek"
BIN_DIR="${HOME}/.local/bin"

# --- Visuals ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Gollek - Complete Environment Setup                          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo "This script will prepare your environment for Gollek."
echo ""

# --- Helper: Detect Interactivity ---
if [ -t 0 ]; then
    INTERACTIVE=true
else
    INTERACTIVE=false
fi

prompt_confirm() {
    if [ "$INTERACTIVE" = false ]; then
        return 0 # Auto-confirm in non-interactive mode
    fi
    local reply
    # Use /dev/tty so interactive prompt works even if script is piped via curl
    read -p "$1 (y/n) " -n 1 -r reply < /dev/tty
    echo
    if [[ $reply =~ ^[Yy]$ ]]; then
        return 0
    fi
    return 1
}

# --- 1. Java 25 Verification ---
echo -e "${BLUE}🔍 Checking Java environment...${NC}"
JAVA_OK=false
if command -v java >/dev/null 2>&1; then
    JAVA_VERSION=$(java -version 2>&1 | head -n 1)
    if [[ "$JAVA_VERSION" == *"\"25"* ]]; then
        JAVA_OK=true
        echo -e "${GREEN}✓ Found $JAVA_VERSION${NC}"
    else
        echo -e "${YELLOW}⚠ Found $JAVA_VERSION, but Java 25 is required.${NC}"
    fi
fi

if [ "$JAVA_OK" = false ]; then
    if prompt_confirm "Java 25 is required but not found. Install it now via SDKMAN?"; then
        echo -e "${BLUE}📦 Installing Java 25 via SDKMAN...${NC}"
        curl -fsSL "https://raw.githubusercontent.com/${GOLLEK_REPO}/${GOLLEK_BRANCH}/scripts/install-jdk.sh" | bash
        
        # Load SDKMAN if it was just installed
        export SDKMAN_DIR="$HOME/.sdkman"
        if [ -s "$SDKMAN_DIR/bin/sdkman-init.sh" ]; then
            source "$SDKMAN_DIR/bin/sdkman-init.sh"
        fi
    else
        echo -e "${RED}⏭️ Skipping Java installation. Gollek may not run correctly.${NC}"
    fi
fi

# --- 2. Gollek CLI Installation ---
echo ""
echo -e "${BLUE}🚀 Checking Gollek CLI...${NC}"
if command -v gollek >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Gollek CLI is already available: $(gollek --version 2>/dev/null || echo 'installed')${NC}"
    if prompt_confirm "Do you want to reinstall/update Gollek CLI?"; then
        echo -e "${BLUE}📦 Updating Gollek CLI...${NC}"
        curl -fsSL "https://raw.githubusercontent.com/${GOLLEK_REPO}/${GOLLEK_BRANCH}/scripts/install-cli.sh" | bash
    fi
else
    if prompt_confirm "Gollek CLI not found. Install it now?"; then
        echo -e "${BLUE}📦 Installing Gollek CLI...${NC}"
        curl -fsSL "https://raw.githubusercontent.com/${GOLLEK_REPO}/${GOLLEK_BRANCH}/scripts/install-cli.sh" | bash
    else
        echo -e "${YELLOW}⏭️ Skipping Gollek CLI installation.${NC}"
    fi
fi

# --- 3. JBang Support ---
echo ""
echo -e "${BLUE}📦 Checking jbang...${NC}"
if ! command -v jbang >/dev/null 2>&1; then
    if prompt_confirm "jbang is recommended for running Gollek examples. Install it now?"; then
        echo -e "${BLUE}📦 Installing jbang...${NC}"
        curl -fsSL https://sh.jbang.dev | bash -s - app setup
        # Ensure jbang is in path for this session
        export PATH="$PATH:$HOME/.jbang/bin"
    else
        echo -e "${YELLOW}⏭️ Skipping jbang installation.${NC}"
    fi
else
    echo -e "${GREEN}✓ jbang $(jbang --version 2>/dev/null | head -n 1) is already available.${NC}"
fi

# --- 4. Summary ---
echo ""
echo -e "${GREEN}✅ Gollek environment setup complete!${NC}"
echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo ""
echo -e "1. Try the CLI:"
echo -e "   ${YELLOW}gollek --version${NC}"
echo ""
echo -e "2. Run a Hello World example (if jbang installed):"
echo -e "   ${YELLOW}jbang https://raw.githubusercontent.com/${GOLLEK_REPO}/${GOLLEK_BRANCH}/examples/jbang/hello_gollek.java${NC}"
echo ""
echo -e "3. Open a ${YELLOW}NEW TERMINAL${NC} to ensure all paths (Java, Gollek, JBang) are updated."
echo ""
echo -e "📖 Documentation: https://github.com/${GOLLEK_REPO}"
echo ""
