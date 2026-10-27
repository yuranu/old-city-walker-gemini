#!/usr/bin/env bash
# End-to-end reproducible build pipeline for Old City Walker (Jerusalem):
#   1. Process pinned OSM & DEM snapshot -> 2D city plan & parcelized building footprints
#   2. Generate & upgrade PBR textures + Poly Haven CC0 photogrammetry scans
#   3. Fetch Wikimedia Commons reference photographs for the 29 viewpoints
#   4. Build all 12 Old City 3D region meshes (.glb) in headless Blender + run mesh linter
#   5. Validate city-wide street connectivity, enclosure, and grounding
#
# Usage:
#   ./build.sh                        # Full build (skips texture/photo download if already built)
#   REGIONS="r01_jaffa_gate" ./build.sh   # Rebuild only specified region(s)
#   FORCE_TEXTURES=1 ./build.sh       # Force re-generation of all PBR textures
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

BLENDER="${BLENDER:-blender}"
REGIONS="${REGIONS:-all}"

echo "=== [1/5] Processing OpenStreetMap & Elevation into City Plan and Urban Block Parcels ==="
mkdir -p public/data
cp -f data/places.json public/data/places.json
"$PY" pipeline/fetch_osm_and_dem.py
"$PY" pipeline/build_city_plan.py
"$PY" pipeline/generate_complete_city_footprints.py

echo "=== [2/5] Generating & Upgrading PBR Textures (Poly Haven CC0 + Jerusalem Stone Grading) ==="
if [ ! -f public/textures/sign_atlas.json ] || [ "${FORCE_TEXTURES:-0}" = "1" ]; then
  "$PY" pipeline/gen_pbr_textures.py
  "$PY" pipeline/fetch_real_pbr_textures.py
  "$PY" pipeline/upgrade_realism_textures.py
else
  echo "  [SKIP] public/textures/ already populated (set FORCE_TEXTURES=1 to regenerate)"
fi

echo "=== [3/5] Fetching Wikimedia Commons Reference Photographs ==="
if [ ! -f public/data/references.json ] || [ ! -f public/references/west_plaza.jpg ] || [ "${FORCE_REFS:-0}" = "1" ]; then
  "$PY" pipeline/fetch_references.py
  "$PY" pipeline/fetch_full_city_references.py
else
  # Ensure public/data/places.json is synced from data/places.json
  mkdir -p public/data
  cp -f data/places.json public/data/places.json
  echo "  [SKIP] public/references/ already populated (set FORCE_REFS=1 to re-download)"
fi

echo "=== [4/5] Building 3D Region Models (.glb) via Headless Blender ==="
"$BLENDER" --background --factory-startup --python pipeline/build_city.py -- --region "$REGIONS"

echo "=== [5/5] Validating 3D City Completeness & Street Enclosure ==="
"$PY" pipeline/validate_city_completeness.py
echo "=== Build Complete! Run 'npm run dev' to explore locally. ==="
