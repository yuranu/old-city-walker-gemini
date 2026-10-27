# Old City Walker — Jerusalem (Procedural 3D Reconstruction Pipeline)

Live build: **[old-city-walker-gemini.yuranu.com](https://old-city-walker-gemini.yuranu.com)**

A research project and reproducible automated pipeline for constructing a walkable, street-level 3D historic city—the entire walled **Old City of Jerusalem** (`~1 km²`, all four quarters, the Temple Mount / Haram al-Sharif, the Ottoman city walls, 8 gates, and 29 landmarks)—from open geographic data, elevation models, procedural architectural grammar rules, and open-licensed reference photography.

This repository contains **only the source methodology and pinned raw geographic snapshots** (`~1.5 MB` of code and curated definitions + `5.0 MB` of pinned OpenStreetMap/elevation JSON). All 3D region meshes (`.glb`), PBR textures, street-carved building footprints, and reference photographs are generated deterministically from source via `./build.sh`.

---

## Methodology & Pipeline Architecture

```text
data/osm_old_city_raw.json & data/osm_jaffa_raw.json (Pinned OSM + DEM snapshots)
  │
  ├─► pipeline/fetch_osm_and_dem.py
  │     Projects raw OpenStreetMap nodes/ways/relations into local East-North-Up (ENU)
  │     metres anchored at the Jaffa Gate portal (31.77661°N, 35.22768°E, 766.0m ASL)
  │     and evaluates the continuous C1-smooth multi-hill / Tyropoeon Valley elevation z(x, y).
  │
  ├─► pipeline/build_city_plan.py
  │     Extracts the 16th-century Ottoman City Wall loop, Temple Mount retaining walls,
  │     Quarter polygons, and named historic street centerlines -> public/data/city_plan.json
  │
  ├─► pipeline/generate_complete_city_footprints.py
  │     Topological Insula parcelizer: subtracts street corridors, plazas, and landmark
  │     reserves from the walled area and recursively subdivides urban blocks along their
  │     oriented bounding boxes into ~4,000 street-hugging building parcels + over-street
  │     vaulted bridge passages (Qantara) -> data/osm_old_city.json & data/osm_jaffa.json
  │
  ├─► pipeline/gen_pbr_textures.py
  │   pipeline/fetch_real_pbr_textures.py
  │   pipeline/upgrade_realism_textures.py
  │     Synthesizes base PBR maps, downloads CC0 photogrammetry scans from Poly Haven,
  │     grades them to Jerusalem Meleke & Mizzi Hilu limestone palettes, and generates
  │     trilingual (Hebrew / Arabic / English) ceramic street-name plates & shutter atlases
  │     -> public/textures/
  │
  ├─► pipeline/fetch_references.py & pipeline/fetch_full_city_references.py
  │     Queries Wikimedia Commons API for outdoor architectural photographs of all 29
  │     curated viewpoints and records file/author/source attribution -> public/references/
  │
  ├─► pipeline/build_city.py & pipeline/regions_builder.py
  │     Headless Blender (bpy/bmesh) 3D generator using pipeline/arch_rules.py and
  │     pipeline/props_lib.py: builds 12 streaming region GLBs (r01..r09) with baked
  │     vertex AO/rising-damp colors and runs pipeline/qa_mesh_linter.py (UV stretch &
  │     coplanar Z-fight auto-repair) -> public/models/regions/*.glb
  │
  └─► pipeline/validate_city_completeness.py
        Verifies street clearance, 360° urban enclosure along all thoroughfares, and
        zero floating walls, roofs, or domes across all exported .glb models.
```

---

## Running Locally

### Option A — Quick Start (Pre-Built Release Snapshot, No Blender/Python Required)

If you just want to run the 3D city locally with Node.js without running the Python/Blender generation pipeline, download the pre-built snapshot from [GitHub Releases](https://github.com/yuranu/old-city-walker-gemini/releases/tag/v1.0.0) (which works either standalone or extracted directly into a clone of this repo):

```bash
curl -L https://github.com/yuranu/old-city-walker-gemini/releases/download/v1.0.0/old-city-walker-prebuilt-v1.0.0.tar.gz | tar -xz
npm install
npm run dev
```

### Option B — Full Reproducible Build from Source

#### Prerequisites

- **Linux / macOS** with bash
- **Python 3.10+** (for data processing, Shapely parcelization, and PBR texture synthesis)
- **Blender 4.0+** (`blender` on `PATH`, invoked in `--background` headless mode)
- **Node.js 18+** & **npm** (for the Vite + Three.js + `three-mesh-bvh` web walker)

#### 1. Install Dependencies

```bash
# Python virtual environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Frontend dependencies
npm install
```

#### 2. Run the Full City Build Pipeline

```bash
./build.sh
```

`./build.sh` executes all 5 pipeline stages in order:
1. Processes the pinned OSM & DEM snapshots into `public/data/city_plan.json`, `data/osm_old_city.json`, and `data/osm_jaffa.json`.
2. Generates and upgrades all 77 PBR texture maps and signage atlases in `public/textures/`.
3. Downloads the 29 open-licensed Wikimedia Commons reference photographs into `public/references/`.
4. Builds all 12 regional `.glb` 3D models (`~178 MB` total) in `public/models/regions/` using headless Blender and runs the automated UV/Z-fight mesh linter.
5. Runs the geometric completeness and street-enclosure validator (`pipeline/validate_city_completeness.py`).

#### Useful Partial Rebuild Options

```bash
# Rebuild only a single region (e.g. Jaffa Gate or Temple Mount) in ~15 seconds:
REGIONS="r01_jaffa_gate" ./build.sh
REGIONS="r07_temple_mount" ./build.sh

# Force regeneration of PBR textures or re-fetching of reference photos:
FORCE_TEXTURES=1 ./build.sh
FORCE_REFS=1 ./build.sh

# Re-query live Overpass API & Open-Meteo DEM instead of using the pinned snapshot:
.venv/bin/python pipeline/fetch_osm_and_dem.py --refresh
```

### 3. Run the Interactive 3D Walker

```bash
npm run dev
```

Open `http://localhost:5173` in your browser:
- **Move**: `W` `A` `S` `D` or Arrow Keys (`Shift` to sprint, `Space` to jump)
- **Look**: Click canvas for pointer-lock mouse look, or drag with mouse
- **Fly / Walk Mode**: Press `F` to toggle between ground-clamped BVH walking and free-flight camera
- **Teleport & Streets Plan**: Press `M` or click **Map / Teleport** to jump to any of the 29 landmarks or 8 historic gates
- **Reference Photo Overlay**: Press `R` to compare the 3D view with the Wikimedia Commons photograph for the nearest viewpoint

To build a static bundle for production hosting (`dist/`):

```bash
npm run build
```

---

## Automated Visual & Geometric QA Harness

In addition to the build-time Blender mesh linter (`pipeline/qa_mesh_linter.py`) and geometric validator (`pipeline/validate_city_completeness.py`), the repository includes headless Chrome WebGL inspection tools in `scripts/`:

- `node scripts/qa_capture.mjs`: Captures PBR renders, `+1 cm` camera-jitter frames (for localized `32×32` block Z-fighting detection), and skykey renders across 30 viewpoints.
- `node scripts/roam_capture.mjs`: Captures multi-angle street-level sweeps and generates labeled contact sheets in `build/qa/` via `scripts/make_roam_sheets.py`.

---

## Repository Layout

```text
pipeline/                       Python & headless Blender (bpy) procedural generation pipeline
  fetch_osm_and_dem.py          Overpass API + Open-Meteo DEM fetcher & local ENU elevation model
  build_city_plan.py            2D vector city plan & street network extractor
  generate_complete_city_footprints.py
                                Topological street-carved Insula parcelizer & tunnel extractor
  gen_pbr_textures.py           Base procedural PBR texture & trilingual sign atlas generator
  fetch_real_pbr_textures.py    Poly Haven CC0 photogrammetry PBR fetcher & Jerusalem stone grader
  upgrade_realism_textures.py   Micro-grain stone weathering, shutter graffiti & sign patina
  arch_rules.py                 Jerusalem architectural grammar & custom monument builders
  props_lib.py                  BMesh street furniture, arches, domes, awnings, trees & souq stalls
  build_city.py                 Main headless Blender entrypoint & r01 Jaffa Gate builder
  regions_builder.py            Modular builder for regions r02..r09 across the four quarters
  qa_mesh_linter.py             In-Blender UV stretch & coplanar Z-fight auto-repair linter
  validate_city_completeness.py Street clearance, 360° enclosure & floating-mesh validator
src/                            Three.js + three-mesh-bvh web application (main.js, player.js, minimap.js)
scripts/                        Headless Chrome WebGL QA capture & contact-sheet generators
data/
  osm_old_city_raw.json         Pinned OpenStreetMap + DEM snapshot for the full Old City
  osm_jaffa_raw.json            Pinned OpenStreetMap + DEM snapshot for Jaffa Gate & Mamilla
  places.json                   Curated camera poses and metadata for the 29 viewpoints
  old_city_pois.json            Categorized index of the 8 Gates, Temples, and Touristic Spots
  roam_regions_c1.json          Multi-region QA sweep camera definitions
build.sh                        End-to-end reproducible build script
```

**Not versioned (generated by `./build.sh`)**: `public/models/`, `public/textures/`, `public/references/`, `public/data/`, `data/pbr_cache/`, `data/osm_old_city.json`, `data/osm_jaffa.json`, `build/`, `dist/`.

---

## License & Third-Party Attributions

 All original source code, procedural architectural rules, and documentation in this repository are released under the **[MIT License](LICENSE)**.

Third-party datasets and assets downloaded or processed by the pipeline remain under their respective licenses:

1. **Map Data — OpenStreetMap (`data/osm_*_raw.json`)**:
   - © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), available under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/). Derived 3D building parcels and street networks carry the same attribution.
2. **Elevation Data — Open-Meteo / SRTM (`data/osm_*_raw.json`)**:
   - Spot elevation grid sourced via the [Open-Meteo Elevation API](https://open-meteo.com/en/docs/elevation-api) (Copernicus DEM GLO-90 / SRTM, CC BY 4.0).
3. **Photogrammetry PBR Textures — Poly Haven (`pipeline/fetch_real_pbr_textures.py`)**:
   - High-resolution stone, cobblestone, plaster, wood, bark, and metal scans downloaded at build time from [Poly Haven](https://polyhaven.com/), released under the [CC0 1.0 Universal Public Domain Dedication](https://polyhaven.com/license).
4. **Reference Photographs — Wikimedia Commons (`pipeline/fetch_references.py`, `pipeline/fetch_full_city_references.py`)**:
   - Street-level architectural reference photographs fetched at build time from [Wikimedia Commons](https://commons.wikimedia.org/) are licensed individually under Creative Commons (`CC BY`, `CC BY-SA`, `CC0`) or Public Domain terms. Exact file titles and source description URLs for all 29 photographs are recorded in `public/data/references.json`.
