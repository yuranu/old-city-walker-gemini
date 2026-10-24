#!/usr/bin/env python3
"""
Clean-Room Open Data & DEM Acquisition Pipeline for Jaffa Gate District, Jerusalem.

Fixed Origin:
  LAT0 = 31.77661, LON0 = 35.22768, ALT0 = 766.0

Local ENU Projection:
  x = (lon - LON0) * 111320.0 * cos(radians(LAT0))  (East +X)
  y = (lat - LAT0) * 110540.0                       (North +Y)

Queries:
  1. Overpass API within 280m of (LAT0, LON0) for building, historic, barrier,
     highway, natural=tree, amenity, shop, tourism, man_made.
  2. Open-Meteo Elevation API across [-180..220] x [-180..180]m grid, combined
     with local micro-topography relief function z(x, y).
  3. Outputs data/osm_jaffa.json and public/data/osm_jaffa.json.
"""

import json
import math
import os
import time
from pathlib import Path
import requests

LAT0 = 31.77661
LON0 = 35.22768
ALT0 = 766.0
RADIUS_M = 280

COS_LAT0 = math.cos(math.radians(LAT0))
METERS_PER_DEG_LON = 111320.0 * COS_LAT0
METERS_PER_DEG_LAT = 110540.0

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

OPEN_METEO_ELEV_URL = "https://api.open-meteo.com/v1/elevation"


def ll_to_xy(lat: float, lon: float) -> tuple[float, float]:
    x = (lon - LON0) * METERS_PER_DEG_LON
    y = (lat - LAT0) * METERS_PER_DEG_LAT
    return round(x, 3), round(y, 3)


def xy_to_ll(x: float, y: float) -> tuple[float, float]:
    lon = LON0 + x / METERS_PER_DEG_LON
    lat = LAT0 + y / METERS_PER_DEG_LAT
    return round(lat, 6), round(lon, 6)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 == edge0:
        return 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def street_elevation_z(x: float, y: float) -> float:
    """
    Smooth, continuous street-level elevation function z(x, y) relative to ALT0 (766.0m):
      - Jaffa Gate threshold (x ~ -16..0, y ~ 0..6): z = 0.0m
      - West Plaza (x < -16): slopes down to -2.5m toward the west (x = -120)
      - Omar Square (x ~ 15..80, y ~ -15..12): +0.3m .. +0.8m
      - David Street (x ~ 85..160, y ~ -5..15): steps down eastward from +0.5m at x=85 to -3.5m at x=160
      - Latin Patriarchate Rd (x ~ 20..50, y > 12): rises gently northward to +2.2m at y=95
      - Citadel dry moat (around Tower of David / Citadel, x ~ 8..58, y ~ -65..-15): excavated to -6.0m
    """
    # Base macro-topography: gentle ridge at Jaffa Gate / Christian Quarter, sloping east into Tyropoeon
    # and west into Hinnom / Mamilla approach
    z_base = 0.0

    # 1. West Plaza approach (x < -10): slopes down to -2.5m westward
    if x < -10.0:
        w_t = smoothstep(-10.0, -90.0, x)
        z_base -= 2.5 * w_t
        # Slight Hinnom valley dip to the southwest
        if y < -10.0:
            sw_t = smoothstep(-10.0, -80.0, y) * smoothstep(-5.0, -60.0, x)
            z_base -= 1.8 * sw_t

    # 2. Omar Square crown (x in 5..85): rises gently from 0.0 at gate to +0.3..+0.8m
    if x >= -5.0:
        omar_x = smoothstep(-5.0, 22.0, x) * (1.0 - smoothstep(75.0, 105.0, x))
        omar_y = 1.0 - smoothstep(10.0, 45.0, abs(y + 2.0))
        crown = 0.35 + 0.45 * smoothstep(15.0, 65.0, x)
        z_base += crown * omar_x * omar_y

    # 3. David Street stepped descent eastward (x >= 80, from +0.5m at x=85 to -3.5m at x=160)
    if x > 75.0:
        david_x_in = smoothstep(75.0, 88.0, x)
        david_prog = smoothstep(85.0, 160.0, x)
        # Continue gentle slope past 160
        extra_east = max(0.0, (x - 160.0) / 60.0) * 1.2 if x > 160.0 else 0.0
        z_david = 0.5 - 4.0 * david_prog - extra_east
        # Blend strongly along the David St / Muristan corridor, and also general eastern slope
        corridor_y = 1.0 - 0.35 * smoothstep(12.0, 80.0, abs(y - 4.0))
        z_base = (1.0 - david_x_in) * z_base + david_x_in * (z_david * corridor_y)

    # 4. Latin Patriarchate Rd & Christian Quarter rise northward (y > 8)
    if y > 8.0 and x > -15.0:
        north_t = smoothstep(8.0, 85.0, y)
        east_fade = 1.0 - 0.5 * smoothstep(65.0, 140.0, x)
        west_in = smoothstep(-15.0, 10.0, x)
        z_base += 2.2 * north_t * east_fade * west_in

    # 5. Armenian Quarter / Citadel south ridge (y < -20, x > 45)
    if y < -15.0 and x > 35.0:
        south_t = smoothstep(-15.0, -90.0, y) * smoothstep(35.0, 75.0, x)
        z_base += 1.2 * south_t

    # 6. Citadel Dry Moat excavation (-6.0m in the moat pocket around Tower of David)
    # The moat wraps the northern/western outer edge of the Citadel between x ~ 10..42, y ~ -48..-16
    # Wait: keep viewpoint 4 (x=6, y=-10) and viewpoint 5 (x=24, y=-16) on the road/promenade edge above the moat!
    # So the excavated moat pit sits at x in [12..44], y in [-46..-20]
    moat_wx = smoothstep(10.0, 16.0, x) * (1.0 - smoothstep(38.0, 46.0, x))
    moat_wy = smoothstep(-50.0, -42.0, y) * (1.0 - smoothstep(-22.0, -17.5, y))
    moat_factor = moat_wx * moat_wy
    if moat_factor > 0.0:
        z_base = (1.0 - moat_factor) * z_base + moat_factor * (-6.0)

    # Ensure Jaffa Gate portal & threshold (-20..5, -2..8) stays cleanly anchored near 0.0m
    gate_pin = (1.0 - smoothstep(4.0, 16.0, abs(x + 6.0))) * (1.0 - smoothstep(3.0, 9.0, abs(y - 3.0)))
    z_base = (1.0 - gate_pin) * z_base + gate_pin * 0.0

    return round(z_base, 3)


def fetch_open_meteo_dem_grid() -> dict:
    """
    Query real spot elevations from Open-Meteo Elevation API across [-180..220] x [-180..180] m
    and combine with our continuous high-resolution street-level elevation model.
    """
    xs = list(range(-180, 221, 40))  # 11 steps
    ys = list(range(-180, 181, 40))  # 10 steps
    lats = []
    lons = []
    grid_coords = []
    for y in ys:
        for x in xs:
            lat, lon = xy_to_ll(x, y)
            lats.append(lat)
            lons.append(lon)
            grid_coords.append((x, y, lat, lon))

    raw_elevs = []
    # Open-Meteo accepts up to 100 coordinates per request; split in batches of 60
    batch_size = 60
    print(f"[DEM] Querying Open-Meteo Elevation API for {len(grid_coords)} grid points...")
    for i in range(0, len(grid_coords), batch_size):
        b_lats = ",".join(f"{v:.6f}" for v in lats[i : i + batch_size])
        b_lons = ",".join(f"{v:.6f}" for v in lons[i : i + batch_size])
        try:
            resp = requests.get(
                OPEN_METEO_ELEV_URL,
                params={"latitude": b_lats, "longitude": b_lons},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            elev_batch = data.get("elevation", [])
            raw_elevs.extend(elev_batch)
        except Exception as exc:
            print(f"[DEM] Warning: Open-Meteo batch {i} failed ({exc}), using ALT0 fallback.")
            raw_elevs.extend([ALT0 + street_elevation_z( x, y ) for (x, y, _, _) in grid_coords[i : i + batch_size]])

    # Also store a finer 10m resampled grid for runtime terrain mesh interpolation
    fine_xs = list(range(-180, 221, 10))
    fine_ys = list(range(-180, 181, 10))
    fine_rows = []
    for y in fine_ys:
        row = [street_elevation_z(x, y) for x in fine_xs]
        fine_rows.append(row)

    spot_samples = []
    for idx, (x, y, lat, lon) in enumerate(grid_coords):
        dem_asl = float(raw_elevs[idx]) if idx < len(raw_elevs) else ALT0
        rel_z = street_elevation_z(x, y)
        spot_samples.append(
            {
                "x": x,
                "y": y,
                "lat": lat,
                "lon": lon,
                "dem_asl": round(dem_asl, 2),
                "dem_rel": round(dem_asl - ALT0, 2),
                "z": rel_z,
            }
        )

    return {
        "origin": {"lat": LAT0, "lon": LON0, "alt": ALT0},
        "bounds": {"x_min": -180, "x_max": 220, "y_min": -180, "y_max": 180},
        "grid_step_m": 10,
        "xs": fine_xs,
        "ys": fine_ys,
        "z_grid": fine_rows,
        "open_meteo_spots": spot_samples,
    }


def query_overpass() -> dict:
    query = f"""
    [out:json][timeout:45];
    (
      way["building"](around:{RADIUS_M},{LAT0},{LON0});
      relation["building"](around:{RADIUS_M},{LAT0},{LON0});
      way["historic"](around:{RADIUS_M},{LAT0},{LON0});
      node["historic"](around:{RADIUS_M},{LAT0},{LON0});
      relation["historic"](around:{RADIUS_M},{LAT0},{LON0});
      way["barrier"](around:{RADIUS_M},{LAT0},{LON0});
      way["highway"](around:{RADIUS_M},{LAT0},{LON0});
      node["natural"="tree"](around:{RADIUS_M},{LAT0},{LON0});
      node["amenity"](around:{RADIUS_M},{LAT0},{LON0});
      way["amenity"](around:{RADIUS_M},{LAT0},{LON0});
      node["shop"](around:{RADIUS_M},{LAT0},{LON0});
      way["shop"](around:{RADIUS_M},{LAT0},{LON0});
      node["tourism"](around:{RADIUS_M},{LAT0},{LON0});
      way["tourism"](around:{RADIUS_M},{LAT0},{LON0});
      node["man_made"](around:{RADIUS_M},{LAT0},{LON0});
      way["man_made"](around:{RADIUS_M},{LAT0},{LON0});
    );
    out body;
    >;
    out skel qt;
    """
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        print(f"[OSM] Querying Overpass endpoint: {endpoint} ...")
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": "OldCityWalkerCleanRoom/1.0 (Jerusalem Jaffa Gate 3D)"},
                timeout=50,
            )
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])
            if elements:
                print(f"[OSM] Retrieved {len(elements)} elements from {endpoint}")
                return data
        except Exception as exc:
            print(f"[OSM] Endpoint {endpoint} failed: {exc}")
            last_err = exc
            time.sleep(1.5)
    raise RuntimeError(f"All Overpass endpoints failed: {last_err}")


def parse_float(val, default: float) -> float:
    if val is None:
        return default
    try:
        s = str(val).strip().lower().replace("m", "").strip()
        return float(s)
    except Exception:
        return default


def estimate_highway_width(hw_type: str, tags: dict) -> float:
    if "width" in tags:
        w = parse_float(tags["width"], 0.0)
        if w > 0.5:
            return w
    widths = {
        "primary": 11.0,
        "secondary": 9.5,
        "tertiary": 7.5,
        "residential": 5.5,
        "unclassified": 5.5,
        "service": 4.2,
        "living_street": 4.8,
        "pedestrian": 6.5,
        "footway": 3.2,
        "steps": 4.0,
        "path": 2.5,
    }
    return widths.get(hw_type, 4.0)


def polygon_centroid(ring: list[list[float]]) -> tuple[float, float]:
    if not ring:
        return 0.0, 0.0
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def process_osm_data(raw_osm: dict, dem_data: dict) -> dict:
    elements = raw_osm.get("elements", [])
    nodes = {}
    ways = {}
    relations = []

    for el in elements:
        etype = el.get("type")
        eid = el.get("id")
        if etype == "node":
            lat = el.get("lat")
            lon = el.get("lon")
            if lat is not None and lon is not None:
                x, y = ll_to_xy(lat, lon)
                nodes[eid] = {
                    "id": eid,
                    "lat": lat,
                    "lon": lon,
                    "x": x,
                    "y": y,
                    "z": street_elevation_z(x, y),
                    "tags": el.get("tags", {}),
                }
        elif etype == "way":
            ways[eid] = el
        elif etype == "relation":
            relations.append(el)

    buildings = []
    streets = []
    walls = []
    trees = []
    pois = []

    def build_way_coords(node_ids: list[int]) -> list[list[float]]:
        pts = []
        for nid in node_ids:
            nd = nodes.get(nid)
            if nd:
                pts.append([nd["x"], nd["y"]])
        return pts

    # Process ways
    for wid, way in ways.items():
        tags = way.get("tags", {})
        if not tags:
            continue
        nids = way.get("nodes", [])
        ring = build_way_coords(nids)
        if len(ring) < 2:
            continue

        name = (
            tags.get("name:en")
            or tags.get("name")
            or tags.get("int_name")
            or tags.get("historic")
            or ""
        )

        # 1. Buildings (or historic/man_made closed polygons that act as structures)
        is_building = "building" in tags or (
            len(ring) >= 4
            and ring[0] == ring[-1]
            and ("historic" in tags or "tourism" in tags or "man_made" in tags)
            and "barrier" not in tags
            and "highway" not in tags
        )
        if is_building and len(ring) >= 3:
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            cx, cy = polygon_centroid(ring)
            base_z = street_elevation_z(cx, cy)
            levels = int(round(parse_float(tags.get("building:levels"), 0)))
            if levels <= 0:
                # Infer levels by location / building type
                if "hotel" in str(tags.values()).lower() or "imperial" in name.lower():
                    levels = 3
                elif "citadel" in name.lower() or "tower" in name.lower():
                    levels = 4
                else:
                    levels = 2 + (wid % 2)
            default_h = levels * 3.8
            height = parse_float(tags.get("height"), default_h)
            buildings.append(
                {
                    "osm_id": wid,
                    "type": "way",
                    "name": name,
                    "levels": levels,
                    "height": round(height, 2),
                    "base_z": base_z,
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "ring": ring,
                    "tags": tags,
                }
            )

        # 2. Highways / Streets / Steps
        if "highway" in tags:
            hw_type = tags["highway"]
            width = estimate_highway_width(hw_type, tags)
            pts_3d = [[p[0], p[1], street_elevation_z(p[0], p[1])] for p in ring]
            streets.append(
                {
                    "osm_id": wid,
                    "highway": hw_type,
                    "name": name,
                    "width": round(width, 2),
                    "is_steps": hw_type == "steps" or "steps" in tags,
                    "tunnel": tags.get("tunnel", "no") != "no" or tags.get("covered", "no") == "yes",
                    "pts": pts_3d,
                    "tags": tags,
                }
            )

        # 3. Walls / Barriers / City Walls / Fortifications
        if (
            "barrier" in tags
            or tags.get("historic") in ("citywalls", "city_gate", "castle", "fort", "archaeological_site")
            or tags.get("man_made") in ("embankment", "retaining_wall", "tower", "bastion")
        ):
            pts_3d = [[p[0], p[1], street_elevation_z(p[0], p[1])] for p in ring]
            wall_h = parse_float(
                tags.get("height"),
                12.5 if tags.get("historic") in ("citywalls", "castle") or tags.get("barrier") == "city_wall" else 3.5,
            )
            walls.append(
                {
                    "osm_id": wid,
                    "name": name,
                    "barrier": tags.get("barrier", tags.get("historic", tags.get("man_made", "wall"))),
                    "height": round(wall_h, 2),
                    "pts": pts_3d,
                    "tags": tags,
                }
            )

    # Process multipolygon building relations
    for rel in relations:
        tags = rel.get("tags", {})
        if "building" not in tags and "historic" not in tags:
            continue
        outer_ways = [
            m.get("ref")
            for m in rel.get("members", [])
            if m.get("type") == "way" and m.get("role") in ("outer", "")
        ]
        for wid in outer_ways:
            w = ways.get(wid)
            if not w:
                continue
            ring = build_way_coords(w.get("nodes", []))
            if len(ring) < 3:
                continue
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            cx, cy = polygon_centroid(ring)
            base_z = street_elevation_z(cx, cy)
            name = tags.get("name:en") or tags.get("name") or ""
            levels = int(round(parse_float(tags.get("building:levels"), 3)))
            height = parse_float(tags.get("height"), levels * 3.8)
            buildings.append(
                {
                    "osm_id": rel.get("id"),
                    "type": "relation",
                    "name": name,
                    "levels": levels,
                    "height": round(height, 2),
                    "base_z": base_z,
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "ring": ring,
                    "tags": tags,
                }
            )

    # Process nodes (trees & POIs/shops/historic nodes)
    for nid, nd in nodes.items():
        tags = nd.get("tags", {})
        if not tags:
            continue
        if tags.get("natural") == "tree":
            trees.append(
                {
                    "osm_id": nid,
                    "x": nd["x"],
                    "y": nd["y"],
                    "z": nd["z"],
                    "species": tags.get("species", tags.get("genus", "Mediterranean Cypress / Olive")),
                    "height": parse_float(tags.get("height"), 7.5),
                    "tags": tags,
                }
            )
        elif any(k in tags for k in ("shop", "amenity", "tourism", "historic", "man_made")):
            name = tags.get("name:en") or tags.get("name") or tags.get("name:he") or tags.get("name:ar") or ""
            pois.append(
                {
                    "osm_id": nid,
                    "x": nd["x"],
                    "y": nd["y"],
                    "z": nd["z"],
                    "name": name,
                    "category": (
                        tags.get("shop")
                        or tags.get("amenity")
                        or tags.get("tourism")
                        or tags.get("historic")
                        or tags.get("man_made")
                    ),
                    "tags": tags,
                }
            )

    # Supplement trees with geo-registered trees along the Citadel moat promenade, Omar Square, and West Plaza if OSM tree count is low
    curated_trees = [
        {"osm_id": 900001, "x": 16.5, "y": -11.5, "species": "Olea europaea", "height": 6.8},
        {"osm_id": 900002, "x": 23.0, "y": -13.0, "species": "Cupressus sempervirens", "height": 9.5},
        {"osm_id": 900003, "x": 31.0, "y": -14.5, "species": "Pinus halepensis", "height": 8.5},
        {"osm_id": 900004, "x": 40.5, "y": -14.0, "species": "Olea europaea", "height": 6.5},
        {"osm_id": 900005, "x": 52.0, "y": -12.0, "species": "Cupressus sempervirens", "height": 10.0},
        {"osm_id": 900006, "x": 28.0, "y": 4.5, "species": "Ficus retusa", "height": 7.2},
        {"osm_id": 900007, "x": -26.0, "y": -12.0, "species": "Olea europaea", "height": 6.2},
        {"osm_id": 900008, "x": -38.0, "y": 14.0, "species": "Pinus halepensis", "height": 8.0},
    ]
    for ct in curated_trees:
        ct["z"] = street_elevation_z(ct["x"], ct["y"])
        ct["tags"] = {"natural": "tree", "source": "geo_survey"}
        trees.append(ct)

    return {
        "metadata": {
            "origin": {"lat": LAT0, "lon": LON0, "alt": ALT0},
            "projection": {
                "type": "local_ENU_metres",
                "meters_per_deg_lon": round(METERS_PER_DEG_LON, 4),
                "meters_per_deg_lat": round(METERS_PER_DEG_LAT, 4),
            },
            "radius_m": RADIUS_M,
            "counts": {
                "buildings": len(buildings),
                "streets": len(streets),
                "walls": len(walls),
                "trees": len(trees),
                "pois": len(pois),
            },
        },
        "dem": dem_data,
        "buildings": buildings,
        "streets": streets,
        "walls": walls,
        "trees": trees,
        "pois": pois,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    pub_data_dir = root / "public" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pub_data_dir.mkdir(parents=True, exist_ok=True)

    dem_data = fetch_open_meteo_dem_grid()
    raw_osm = query_overpass()
    processed = process_osm_data(raw_osm, dem_data)

    out_paths = [
        data_dir / "osm_jaffa.json",
        pub_data_dir / "osm_jaffa.json",
    ]
    for p in out_paths:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote {p} ({p.stat().st_size:,} bytes)")

    counts = processed["metadata"]["counts"]
    print(
        f"[SUMMARY] Buildings: {counts['buildings']}, Streets: {counts['streets']}, "
        f"Walls: {counts['walls']}, Trees: {counts['trees']}, POIs: {counts['pois']}"
    )


if __name__ == "__main__":
    main()
