#!/bin/bash
# One-command setup for Gollek SDK with Jupyter and jbang (using pre-built artifacts)

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Gollek SDK - Jupyter & jbang Setup (Pre-built)              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

STEP=1

# Step 1: Check prerequisites
echo "[$STEP/4] Checking prerequisites..."
STEP=$((STEP+1))

command -v java &> /dev/null || { echo "❌ Java not found. Install Java 11+"; exit 1; }
java_version=$(java -version 2>&1 | grep 'version' | sed 's/.*version "\([^"]*\)".*/\1/')
echo "✓ Java $java_version found"

# Step 2: Setup Jupyter
echo ""
echo "[$STEP/4] Setting up Jupyter kernel..."
STEP=$((STEP+1))

command -v jupyter &> /dev/null || {
    echo "⚠ Jupyter not found. To install:"
    echo "  pip install jupyter"
    echo "  Or: conda install jupyter"
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/jupyter-kernel" ]; then
    bash "$SCRIPT_DIR/jupyter-kernel/install.sh"
    echo "✓ Jupyter kernel installed"
else
    echo "❌ jupyter-kernel directory not found"
    exit 1
fi

# Step 3: Setup jbang
echo ""
echo "[$STEP/4] Checking jbang..."
STEP=$((STEP+1))

if ! command -v jbang &> /dev/null; then
    echo "⚠ jbang not found. To install:"
    echo "  curl -Ls https://sh.jbang.dev | bash -s - app setup"
    echo "  Then add to PATH: export PATH=\$PATH:~/.jbang/bin"
else
    jbang_version=$(jbang --version)
    echo "✓ jbang $jbang_version found"
fi

# Step 4: Summary
echo ""
echo "[$STEP/4] Setup complete!"
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Gollek SDK Ready for Jupyter & jbang                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "📚 Next Steps:"
echo ""
echo "1. Jupyter Notebooks:"
echo "   jupyter notebook"
echo "   Create new notebook with 'Gollek SDK (Java)' kernel"
echo ""
echo "2. jbang Scripts:"
echo "   jbang $SCRIPT_DIR/jbang-templates/gollek-template.java"
echo "   Or: jbang $SCRIPT_DIR/jbang-templates/examples/neural-network-example.java"
echo ""
echo "📖 Documentation:"
echo "   - Jupyter: $SCRIPT_DIR/jupyter-kernel/JUPYTER_SETUP.md"
echo "   - jbang: $SCRIPT_DIR/jbang-templates/JBANG_SETUP.md"
echo "   - Examples: $SCRIPT_DIR/examples/EXAMPLES.md"
echo ""
echo "🚀 Ready to build amazing things with Gollek SDK!"
echo ""
