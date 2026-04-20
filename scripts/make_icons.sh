#!/usr/bin/env bash
# Convert a PNG to .icns (macOS) and .ico (Windows)
# Usage: ./scripts/make_icons.sh <source.png>
# Outputs: same directory as source, same stem

set -e

SRC="${1:?Usage: $0 <source.png>}"
STEM="${SRC%.*}"
DIR="$(dirname "$SRC")"

# ── icns (macOS) ──────────────────────────────────────────────────────────────
ICONSET="${DIR}/$(basename "$STEM").iconset"
mkdir -p "$ICONSET"

sizes=(16 32 128 256 512)
for s in "${sizes[@]}"; do
  sips -z $s $s "$SRC" --out "${ICONSET}/icon_${s}x${s}.png"          >/dev/null
  sips -z $((s*2)) $((s*2)) "$SRC" --out "${ICONSET}/icon_${s}x${s}@2x.png" >/dev/null
done

iconutil -c icns "$ICONSET" -o "${STEM}.icns"
rm -rf "$ICONSET"
echo "Created ${STEM}.icns"

# ── ico (Windows) ─────────────────────────────────────────────────────────────
if command -v magick &>/dev/null; then
  CONVERTER="magick"
elif command -v convert &>/dev/null; then
  CONVERTER="convert"
else
  echo "Warning: ImageMagick not found — skipping .ico (install with: brew install imagemagick)"
  exit 0
fi

$CONVERTER "$SRC" \
  \( -clone 0 -resize 16x16   \) \
  \( -clone 0 -resize 32x32   \) \
  \( -clone 0 -resize 48x48   \) \
  \( -clone 0 -resize 64x64   \) \
  \( -clone 0 -resize 128x128 \) \
  \( -clone 0 -resize 256x256 \) \
  -delete 0 "${STEM}.ico"
echo "Created ${STEM}.ico"
