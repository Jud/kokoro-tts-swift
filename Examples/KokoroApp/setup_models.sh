#!/bin/bash
# Downloads Kokoro models into the app's Models/ directory for bundling.
# Run once before building: ./setup_models.sh

set -euo pipefail
cd "$(dirname "$0")"

MODELS_DIR="Models"
mkdir -p "$MODELS_DIR"

if [ -d "$MODELS_DIR/kokoro_frontend.mlmodelc" ] && [ -d "$MODELS_DIR/voices" ]; then
    echo "Models already present in $MODELS_DIR/"
    exit 0
fi

# Try to copy from CLI cache first
CLI_CACHE="$HOME/Library/Application Support/kokoro-coreml/models/kokoro"
if [ -d "$CLI_CACHE/kokoro_frontend.mlmodelc" ]; then
    echo "Copying models from CLI cache..."
    cp -R "$CLI_CACHE/kokoro_frontend.mlmodelc" "$MODELS_DIR/"
    cp -R "$CLI_CACHE/kokoro_backend.mlmodelc" "$MODELS_DIR/"
    cp -R "$CLI_CACHE/voices" "$MODELS_DIR/"
    echo "Done. Models ready in $MODELS_DIR/"
    exit 0
fi

# Otherwise download via CLI
echo "No cached models found. Downloading via kokoro CLI..."
if ! command -v kokoro &>/dev/null; then
    echo "Error: kokoro CLI not found. Install via: brew install jud/kokoro-coreml/kokoro"
    exit 1
fi

kokoro update
cp -R "$CLI_CACHE/kokoro_frontend.mlmodelc" "$MODELS_DIR/"
cp -R "$CLI_CACHE/kokoro_backend.mlmodelc" "$MODELS_DIR/"
cp -R "$CLI_CACHE/voices" "$MODELS_DIR/"
echo "Done. Models ready in $MODELS_DIR/"
