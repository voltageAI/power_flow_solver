#!/usr/bin/env bash
set -euo pipefail

# Build release for current platform
VERSION="${1:-0.1.2}"
NIF_VERSION="2.15"

echo "Building release for version ${VERSION}..."

# Force build from source
export POWER_FLOW_FORCE_BUILD=true

# Clean and compile
mix clean
mix compile

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        TARGET="aarch64-apple-darwin"
    else
        TARGET="x86_64-apple-darwin"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    TARGET="x86_64-unknown-linux-gnu"
elif [[ "$OSTYPE" == "linux-musl"* ]]; then
    TARGET="x86_64-unknown-linux-musl"
else
    echo "Unknown platform: $OSTYPE"
    exit 1
fi

echo "Target platform: ${TARGET}"

# Find the compiled .so file
SO_FILE=$(find native/power_flow_solver/target -name "libpower_flow_solver.so" | head -1)

if [[ -z "$SO_FILE" ]]; then
    echo "ERROR: Could not find libpower_flow_solver.so"
    exit 1
fi

echo "Found NIF at: ${SO_FILE}"

# Create tarball
PKG_NAME="libpower_flow_solver-v${VERSION}-nif-${NIF_VERSION}-${TARGET}.so.tar.gz"
mkdir -p releases
tar -czf "releases/${PKG_NAME}" -C "$(dirname "$SO_FILE")" "libpower_flow_solver.so"

echo ""
echo "✅ Created release package: releases/${PKG_NAME}"
echo ""
echo "Next steps:"
echo "1. Create a GitHub release for v${VERSION}"
echo "2. Upload releases/${PKG_NAME} to that release"
echo "3. Repeat for other platforms (Linux, other macOS arch)"
echo ""
echo "Or push a tag 'v${VERSION}' and let GitHub Actions handle it:"
echo "   git tag v${VERSION}"
echo "   git push origin v${VERSION}"
