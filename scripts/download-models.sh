#!/bin/bash
set -euo pipefail

REPO="Jud/kokoro-tts-swift"
TAG="models-v1"
ASSET="kokoro-models.tar.gz"
DEFAULT_DIR="$HOME/Library/Application Support/kokoro-tts/models/kokoro"
DEST="${1:-$DEFAULT_DIR}"

URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

if [ -d "$DEST/voices" ]; then
    echo "Models already present at: $DEST"
    exit 0
fi

echo "Downloading KokoroTTS models..."
echo "  from: $URL"
echo "  to:   $DEST"

mkdir -p "$DEST"
curl -L --progress-bar "$URL" | tar xz -C "$DEST"

echo "Done. Models installed to: $DEST"
