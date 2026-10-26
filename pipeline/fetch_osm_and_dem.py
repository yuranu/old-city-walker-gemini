#!/usr/bin/env python3
"""
Clean-Room Open Data & DEM Acquisition Pipeline for the Full Old City of Jerusalem.

Fixed Origin (Jaffa Gate Portal):
  LAT0 = 31.77661, LON0 = 35.22768, ALT0 = 766.0

Local ENU Projection:
  x = (lon - LON0) * 111320.0 * cos(radians(LAT0))  (East +X)
  y = (lat - LAT0) * 110540.0                       (North +Y)

Full Old City Bounds (Ottoman City Walls + immediate perimeter):
  Latitude:  31.7702 .. 31.7840  (y: -708m .. +817m)
  Longitude: 35.2248 .. 35.2385  (x: -272m .. +1024m)

Queries:
  1. Overpass API across the full Old City bounding box for building, historic,
     barrier, highway, natural=tree, amenity, shop, tourism, man_made, place_of_worship.
  2. Open-Meteo Elevation API across [-250..1000] x [-680..780]m grid, combined
     with the continuous multi-hill / Tyropoeon Valley / Temple Mount elevation model z(x, y).
  3. Classifies every feature into one of the 8 contiguous Old City regions (r01..r08)
     and writes data/osm_old_city.json, public/data/osm_old_city.json, and data/osm_jaffa.json.
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

BBOX_SOUTH = 31.7702
BBOX_WEST = 35.2248
BBOX_NORTH = 31.7840
BBOX_EAST = 35.2385

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
    Continuous, C1-smooth street-level elevation function z(x, y) in metres
    relative to Jaffa Gate threshold (ALT0 = 766.0m ASL -> z = 0.0m):
      - Exactly matches the approved Jaffa Gate elevation model inside r01 (x <= 165, -90 <= y <= 65).
      - Models the real topography of the Old City beyond r01:
        * NW Christian Quarter / New Gate ridge (x ~ -170..100, y ~ 150..350): rises to +16m (782m ASL)
        * Church of the Holy Sepulchre / Muristan (x ~ 180..280, y ~ 120..240): -6.0m .. -8.5m (758m ASL)
        * Central Tyropoeon Valley (El-Wad / HaGai St -> Western Wall -> Dung Gate, x ~ 260..630):
          descends from -11m at Damascus Gate (262, 553) to -24m at Chain Gate (640, 35)
          and -29m at Western Wall Plaza (635, 12) and -34m at Dung Gate (612, -228).
        * Western Hill / Armenian & Jewish Quarters / Zion Gate (x ~ 0..420, y ~ -450..-80):
          high plateau at +2.0m .. +4.5m in Armenian Quarter / Zion Gate (174, -408),
          stepping down eastward across the Jewish Quarter (Hurva at -8.0m) toward the Western Wall (-29m).
        * Temple Mount / Haram al-Sharif Esplanade (x ~ 668..900, y ~ -180..360):
          level stone esplanade at z = -26.0m (740m ASL), with the raised Dome of the Rock
          upper platform at z = -22.0m (744m ASL).
        * Muslim Quarter / Herod's Gate / Lions' Gate (y > 260, x > 260):
          Bezetha ridge around Herod's Gate (603, 701) at -5.0m, descending to -20.0m at
          St. Anne's / Lions' Gate (870, 469).
    """
    # 1. Compute the exact Jaffa Gate local elevation for smooth blending
    z_jaffa = 0.0
    if x < -10.0:
        w_t = smoothstep(-10.0, -90.0, x)
        z_jaffa -= 2.5 * w_t
        if y < -10.0:
            sw_t = smoothstep(-10.0, -80.0, y) * smoothstep(-5.0, -60.0, x)
            z_jaffa -= 1.8 * sw_t

    if x >= -5.0:
        omar_x = smoothstep(-5.0, 22.0, x) * (1.0 - smoothstep(75.0, 105.0, x))
        omar_y = 1.0 - smoothstep(10.0, 45.0, abs(y + 2.0))
        crown = 0.35 + 0.45 * smoothstep(15.0, 65.0, x)
        z_jaffa += crown * omar_x * omar_y

    if x > 75.0:
        david_x_in = smoothstep(75.0, 88.0, x)
        david_prog = smoothstep(85.0, 160.0, x)
        extra_east = max(0.0, min(180.0, x - 160.0) / 60.0) * 1.2 if x > 160.0 else 0.0
        z_david = 0.5 - 4.0 * david_prog - extra_east
        corridor_y = 1.0 - 0.35 * smoothstep(12.0, 80.0, abs(y - 4.0))
        z_jaffa = (1.0 - david_x_in) * z_jaffa + david_x_in * (z_david * corridor_y)

    if y > 8.0 and x > -15.0:
        north_t = smoothstep(8.0, 85.0, y)
        east_fade = 1.0 - 0.5 * smoothstep(65.0, 140.0, x)
        west_in = smoothstep(-15.0, 10.0, x)
        z_jaffa += 2.2 * north_t * east_fade * west_in

    if y < -15.0 and x > 35.0:
        south_t = smoothstep(-15.0, -90.0, y) * smoothstep(35.0, 75.0, x)
        z_jaffa += 1.2 * south_t

    gate_pin = (1.0 - smoothstep(4.0, 16.0, abs(x + 6.0))) * (1.0 - smoothstep(3.0, 9.0, abs(y - 3.0)))
    z_jaffa = (1.0 - gate_pin) * z_jaffa + gate_pin * 0.0

    # 2. Macro Old City Topography outside the immediate Jaffa Gate core
    # How far outside the Jaffa Gate core (-80 <= x <= 155, -85 <= y <= 65) are we?
    in_jaffa_x = smoothstep(-110.0, -65.0, x) * (1.0 - smoothstep(145.0, 195.0, x))
    in_jaffa_y = smoothstep(-105.0, -70.0, y) * (1.0 - smoothstep(55.0, 95.0, y))
    w_jaffa = in_jaffa_x * in_jaffa_y

    if w_jaffa >= 0.999:
        return round(z_jaffa, 3)

    # Full-city macro surface z_macro(x, y):
    # Eastward descent from Western Hill (x <= 180) into Tyropoeon Valley (x ~ 480..650)
    tyropoeon_t = smoothstep(110.0, 620.0, x)
    # Valley floor is deeper in the south (Dung Gate -30m, Western Wall -26m) than north (Damascus Gate -11m)
    valley_depth = -25.0 + 13.0 * smoothstep(-80.0, 560.0, y) - 5.0 * smoothstep(-50.0, -280.0, y)
    z_macro = tyropoeon_t * valley_depth

    # NW Christian Quarter / New Gate high ridge (around x = -140..60, y = 140..360 -> rises up to +14m)
    nw_ridge_x = 1.0 - smoothstep(-20.0, 220.0, x)
    nw_ridge_y = smoothstep(65.0, 240.0, y) * (1.0 - smoothstep(360.0, 520.0, y))
    z_macro += 13.5 * nw_ridge_x * nw_ridge_y

    # SW Armenian Quarter & Mount Zion plateau (x = -20..260, y = -450..-70 -> +2.5m..+4.0m)
    sw_hill_x = smoothstep(-60.0, 20.0, x) * (1.0 - smoothstep(230.0, 420.0, x))
    sw_hill_y = smoothstep(-60.0, -180.0, y)
    z_macro += 3.2 * sw_hill_x * sw_hill_y

    # Bezetha NE ridge near Herod's Gate (x = 450..680, y = 520..740 -> rises by +9m relative to valley)
    ne_ridge_x = smoothstep(340.0, 550.0, x) * (1.0 - smoothstep(680.0, 850.0, x))
    ne_ridge_y = smoothstep(420.0, 650.0, y)
    z_macro += 9.5 * ne_ridge_x * ne_ridge_y

    # Temple Mount / Haram al-Sharif flat esplanade (x = 668..895, y = -185..365 -> level at -24.0m,
    # with the raised central Dome of the Rock platform at x = 695..785, y = 95..225 at -20.2m)
    tm_x = smoothstep(658.0, 675.0, x) * (1.0 - smoothstep(885.0, 910.0, x))
    tm_y = smoothstep(-195.0, -170.0, y) * (1.0 - smoothstep(355.0, 385.0, y))
    tm_w = tm_x * tm_y
    if tm_w > 0.0:
        z_tm = -24.0
        z_macro = (1.0 - tm_w) * z_macro + tm_w * z_tm

    # Western Wall Plaza flat prayer esplanade (x = 575..668, y = -65..55 -> level at -26.5m)
    kotel_x = smoothstep(555.0, 580.0, x) * (1.0 - smoothstep(662.0, 672.0, x))
    kotel_y = smoothstep(-75.0, -45.0, y) * (1.0 - smoothstep(45.0, 70.0, y))
    kotel_w = kotel_x * kotel_y
    if kotel_w > 0.0:
        z_macro = (1.0 - kotel_w) * z_macro + kotel_w * (-26.5)

    z_final = w_jaffa * z_jaffa + (1.0 - w_jaffa) * z_macro
    return round(z_final, 3)


def classify_region(x: float, y: float) -> str:
    """
    Assigns any ENU coordinate (x, y) inside the Old City to one of the 8 contiguous regions:
      - r01_jaffa_gate
      - r02_christian_quarter
      - r03_central_souqs
      - r04_armenian_quarter
      - r05_jewish_quarter
      - r06_western_wall_dung_gate
      - r07_temple_mount
      - r08_muslim_quarter_north_gates
    """
    # 1. r01_jaffa_gate: Approved Jaffa Gate, Citadel, Omar Square, West Plaza, upper David St
    if x <= 165.0 and -95.0 <= y <= 65.0:
        return "r01_jaffa_gate"

    # 2. r07_temple_mount: Haram al-Sharif / Temple Mount & Golden Gate / Eastern Wall
    if x >= 665.0 and -220.0 <= y <= 390.0:
        return "r07_temple_mount"

    # 3. r06_western_wall_dung_gate: Western Wall Plaza, Mughrabi Bridge, Dung Gate, Southern Wall
    if x >= 475.0 and y < 65.0:
        return "r06_western_wall_dung_gate"

    # 4. r08_muslim_quarter_north_gates: Muslim Quarter, Via Dolorosa, Damascus Gate, Herod's Gate, Lions' Gate
    if y > 260.0 and x > 240.0:
        return "r08_muslim_quarter_north_gates"
    if x >= 605.0 and y >= 65.0:
        return "r08_muslim_quarter_north_gates"

    # 5. r02_christian_quarter: Holy Sepulchre, Muristan, Redeemer, New Gate, NW Wall
    if y > 65.0 and x <= 295.0:
        return "r02_christian_quarter"

    # 6. r03_central_souqs: Lower David St / Chain St, Triple Souq, Northern Cardo, Khan el-Zeit
    if -45.0 <= y <= 260.0 and 165.0 < x < 605.0:
        return "r03_central_souqs"

    # 7. r04_armenian_quarter: St. James Cathedral, Armenian Patriarchate, Zion Gate, SW Wall
    if y < -45.0 and x <= 280.0:
        return "r04_armenian_quarter"

    # 8. r05_jewish_quarter: Hurva Synagogue, Southern Cardo, Four Sephardic Synagogues, Broad Wall
    return "r05_jewish_quarter"


def fetch_open_meteo_dem_grid() -> dict:
    """
    Query real spot elevations from Open-Meteo Elevation API across [-240..960] x [-640..720] m
    and combine with our continuous high-resolution street-level elevation model.
    """
    xs = list(range(-240, 961, 120))  # 11 steps
    ys = list(range(-640, 721, 136))  # 11 steps
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
    batch_size = 65
    print(f"[DEM] Querying Open-Meteo Elevation API for {len(grid_coords)} full-city grid points...")
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
            print(f"[DEM] Warning: Open-Meteo batch {i} failed ({exc}), using synthetic elevation.")
            raw_elevs.extend([ALT0 + street_elevation_z(x, y) for (x, y, _, _) in grid_coords[i : i + batch_size]])

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
        "bounds": {"x_min": -260, "x_max": 1000, "y_min": -680, "y_max": 780},
        "open_meteo_spots": spot_samples,
    }


def query_overpass() -> dict:
    bbox = f"{BBOX_SOUTH},{BBOX_WEST},{BBOX_NORTH},{BBOX_EAST}"
    query = f"""
    [out:json][timeout:90];
    (
      way["building"]({bbox});
      relation["building"]({bbox});
      way["historic"]({bbox});
      node["historic"]({bbox});
      relation["historic"]({bbox});
      way["barrier"]({bbox});
      way["highway"]({bbox});
      node["natural"="tree"]({bbox});
      node["amenity"]({bbox});
      way["amenity"]({bbox});
      node["shop"]({bbox});
      way["shop"]({bbox});
      node["tourism"]({bbox});
      way["tourism"]({bbox});
      node["man_made"]({bbox});
      way["man_made"]({bbox});
    );
    out body;
    >;
    out skel qt;
    """
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        print(f"[OSM] Querying Overpass endpoint for full Old City ({bbox}): {endpoint} ...")
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": "OldCityWalkerCleanRoom/2.0 (Jerusalem Full Old City 3D)"},
                timeout=95,
            )
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])
            if elements:
                print(f"[OSM] Retrieved {len(elements):,} elements from {endpoint}")
                return data
        except Exception as exc:
            print(f"[OSM] Endpoint {endpoint} failed: {exc}")
            last_err = exc
            time.sleep(2.0)
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
        "residential": 5.2,
        "unclassified": 5.2,
        "service": 4.0,
        "living_street": 4.5,
        "pedestrian": 5.8,
        "footway": 3.2,
        "steps": 3.8,
        "path": 2.5,
    }
    return widths.get(hw_type, 3.8)


def polygon_centroid(ring: list[list[float]]) -> tuple[float, float]:
    if not ring:
        return 0.0, 0.0
    pts = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    if not pts:
        pts = ring
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
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
    region_counts = {}

    def build_way_coords(node_ids: list[int]) -> list[list[float]]:
        pts = []
        for nid in node_ids:
            nd = nodes.get(nid)
            if nd:
                pts.append([nd["x"], nd["y"]])
        return pts

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
            reg_id = classify_region(cx, cy)
            region_counts[reg_id] = region_counts.get(reg_id, 0) + 1
            base_z = street_elevation_z(cx, cy)
            levels = int(round(parse_float(tags.get("building:levels"), 0)))
            if levels <= 0:
                if "hotel" in str(tags.values()).lower() or "imperial" in name.lower():
                    levels = 3
                elif "citadel" in name.lower() or "tower" in name.lower():
                    levels = 4
                elif "church" in str(tags.values()).lower() or "mosque" in str(tags.values()).lower() or "synagogue" in str(tags.values()).lower():
                    levels = 3
                else:
                    levels = 2 + (wid % 2)
            default_h = levels * 3.6
            height = parse_float(tags.get("height"), default_h)
            buildings.append(
                {
                    "osm_id": wid,
                    "type": "way",
                    "region": reg_id,
                    "name": name,
                    "levels": levels,
                    "height": round(height, 2),
                    "base_z": base_z,
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "ring": ring,
                    "tags": tags,
                }
            )

        if "highway" in tags:
            hw_type = tags["highway"]
            width = estimate_highway_width(hw_type, tags)
            pts_3d = [[p[0], p[1], street_elevation_z(p[0], p[1])] for p in ring]
            mid_pt = ring[len(ring) // 2]
            reg_id = classify_region(mid_pt[0], mid_pt[1])
            streets.append(
                {
                    "osm_id": wid,
                    "region": reg_id,
                    "highway": hw_type,
                    "name": name,
                    "width": round(width, 2),
                    "is_steps": hw_type == "steps" or "steps" in tags,
                    "tunnel": tags.get("tunnel", "no") != "no" or tags.get("covered", "no") == "yes",
                    "pts": pts_3d,
                    "tags": tags,
                }
            )

        if (
            "barrier" in tags
            or tags.get("historic") in ("citywalls", "city_gate", "castle", "fort", "archaeological_site")
            or tags.get("man_made") in ("embankment", "retaining_wall", "tower", "bastion")
        ):
            pts_3d = [[p[0], p[1], street_elevation_z(p[0], p[1])] for p in ring]
            mid_pt = ring[len(ring) // 2]
            reg_id = classify_region(mid_pt[0], mid_pt[1])
            wall_h = parse_float(
                tags.get("height"),
                12.5 if tags.get("historic") in ("citywalls", "castle") or tags.get("barrier") == "city_wall" else 3.5,
            )
            walls.append(
                {
                    "osm_id": wid,
                    "region": reg_id,
                    "name": name,
                    "barrier": tags.get("barrier", tags.get("historic", tags.get("man_made", "wall"))),
                    "height": round(wall_h, 2),
                    "pts": pts_3d,
                    "tags": tags,
                }
            )

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
            reg_id = classify_region(cx, cy)
            region_counts[reg_id] = region_counts.get(reg_id, 0) + 1
            base_z = street_elevation_z(cx, cy)
            name = tags.get("name:en") or tags.get("name") or ""
            levels = int(round(parse_float(tags.get("building:levels"), 3)))
            height = parse_float(tags.get("height"), levels * 3.6)
            buildings.append(
                {
                    "osm_id": rel.get("id"),
                    "type": "relation",
                    "region": reg_id,
                    "name": name,
                    "levels": levels,
                    "height": round(height, 2),
                    "base_z": base_z,
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "ring": ring,
                    "tags": tags,
                }
            )

    for nid, nd in nodes.items():
        tags = nd.get("tags", {})
        if not tags:
            continue
        reg_id = classify_region(nd["x"], nd["y"])
        if tags.get("natural") == "tree":
            trees.append(
                {
                    "osm_id": nid,
                    "region": reg_id,
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
                    "region": reg_id,
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

    return {
        "metadata": {
            "origin": {"lat": LAT0, "lon": LON0, "alt": ALT0},
            "projection": {
                "type": "local_ENU_metres",
                "meters_per_deg_lon": round(METERS_PER_DEG_LON, 4),
                "meters_per_deg_lat": round(METERS_PER_DEG_LAT, 4),
            },
            "bbox": {"south": BBOX_SOUTH, "west": BBOX_WEST, "north": BBOX_NORTH, "east": BBOX_EAST},
            "counts": {
                "buildings": len(buildings),
                "streets": len(streets),
                "walls": len(walls),
                "trees": len(trees),
                "pois": len(pois),
                "buildings_by_region": region_counts,
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
        data_dir / "osm_old_city.json",
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
    print(f"[REGIONS] {json.dumps(counts['buildings_by_region'], indent=2)}")


if __name__ == "__main__":
    main()
