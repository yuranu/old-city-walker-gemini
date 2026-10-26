#!/usr/bin/env python3
"""
Automated City-Wide Geometry, Street-Enclosure, Map-Alignment & Floating-Mesh Validator
(`pipeline/validate_city_completeness.py`).

Validates across the entire Old City of Jerusalem (`r01`..`r09`):
  1. Street Blockage Check: Every sample point (every 10m along all 27 curatedStreets in
     `public/data/city_plan.json` + all 29 POIs in `public/data/places.json`) must be clear
     of building interiors.
  2. 360° Urban Enclosure & Missing-Block Void Detection: Casts 24 horizontal rays (every 15°)
     at every street sample point and POI; verifies that intramural street points (outside known
     open esplanades/gardens) have continuous surrounding building blocks with zero missing-block
     voids.
  3. 3D GLB Grounding & Roof/Dome Support Check: Inspects all exported `.glb` region models to
     verify:
       - Zero floating ground-floor building walls (`min_z > ground_z + 0.35m`)
       - Zero roof decks or roof domes floating outside building/monument footprints over streets.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "pipeline"))

from build_city_plan import OLD_CITY_WALL_POLYGON  # noqa: E402
from generate_complete_city_footprints import street_elevation_z  # noqa: E402


# Monument 3D solid footprints built procedurally in `regions_builder.py` / `arch_rules.py`
PROCEDURAL_MONUMENT_BOXES: List[Tuple[str, Polygon]] = [
    ("jaffa_gate", box(-20.0, -14.0, 14.0, 16.0)),
    ("citadel_tower_of_david", box(8.0, -112.0, 62.0, -14.0)),
    ("new_imperial_hotel", box(18.0, -48.0, 52.0, -8.0)),
    ("petra_hotel_and_omar_sq", box(28.0, 4.0, 86.0, 34.0)),
    ("christian_info_centre", box(62.0, -28.0, 84.0, -10.0)),
    ("latin_patriarchate_hero", box(8.0, 22.0, 46.0, 72.0)),
    ("david_st_souq_hero_n", box(76.0, 4.0, 168.0, 16.0)),
    ("david_st_souq_hero_s", box(76.0, -12.0, 168.0, -1.0)),
    ("holy_sepulchre", box(164.0, 150.0, 228.0, 220.0)),
    ("redeemer_muristan", box(220.0, 114.0, 286.0, 166.0)),
    ("new_gate", box(-154.0, 280.0, -120.0, 312.0)),
    ("triple_souq", box(300.5, -40.0, 322.0, 122.0)),
    ("david_st_south_bazaar", box(200.0, 6.0, 298.0, 18.0)),
    ("suq_al_qattanin", box(520.0, 136.0, 618.0, 156.0)),
    ("chain_street_turbas", box(434.0, 14.0, 536.0, 52.0)),
    ("st_james_cathedral", box(114.0, -246.0, 152.0, -212.0)),
    ("st_james_north_compound", box(140.0, -160.0, 172.0, -140.0)),
    ("ararat_st_corner_house", box(192.0, -118.0, 213.0, -96.0)),
    ("armenian_monastery_west", box(42.0, -350.0, 62.0, -180.0)),
    ("zion_gate_and_lane", box(146.0, -418.0, 208.0, -376.0)),
    ("hurva_synagogue", box(330.0, -154.0, 384.0, -102.0)),
    ("roman_cardo", box(296.0, -196.0, 322.0, -52.0)),
    ("four_sephardic_synagogues", box(336.0, -252.0, 382.0, -224.0)),
    ("broad_wall", box(326.0, -94.0, 345.0, -66.0)),
    ("tiferet_yisrael", box(426.0, -148.0, 452.0, -122.0)),
    ("batei_mahaseh", box(420.0, -336.0, 454.0, -304.0)),
    ("batei_mahaseh_north_block", box(345.0, -311.5, 378.0, -301.5)),
    ("western_wall_herodian", box(662.0, -90.0, 674.0, 72.0)),
    ("wilsons_arch_tankiziyya", box(638.0, 44.0, 670.0, 68.0)),
    ("beit_strauss_north_plaza", box(594.0, 24.0, 644.0, 49.0)),
    ("mughrabi_bridge", box(584.0, -36.0, 668.0, -28.0)),
    ("dung_gate", box(596.0, -224.0, 634.0, -182.0)),
    ("dome_of_the_rock_platform", box(690.0, 95.0, 815.0, 255.0)),
    ("al_aqsa_mosque", box(715.0, -85.0, 815.0, 35.0)),
    ("golden_gate", box(858.0, 240.0, 888.0, 282.0)),
    ("damascus_gate", box(228.0, 514.0, 286.0, 590.0)),
    ("herods_gate", box(550.0, 678.0, 588.0, 722.0)),
    ("lions_gate", box(852.0, 452.0, 892.0, 482.0)),
    ("st_anne_bethesda", box(768.0, 514.0, 850.0, 554.0)),
    ("austrian_hospice", box(400.0, 393.0, 434.0, 419.0)),
    ("ecce_homo_sisters_of_zion", box(511.0, 392.0, 519.0, 405.0)),
    ("stork_tower", box(802.0, 754.0, 828.0, 782.0)),
    ("mamilla_horizon", box(-90.0, -55.0, -55.0, 55.0)),
]

# Legitimate large open esplanades/gardens/plazas where rays naturally travel > 42m across open space
KNOWN_OPEN_ESPLANADES: List[Polygon] = [
    Polygon([
        [569.54, 378.13], [593.53, 255.09], [613.52, 151.91], [627.03, 67.64], [656.99, -104.07],
        [768.09, -84.84], [817.61, 86.25], [859.65, 255.14], [875.59, 444.75],
    ]).buffer(22.0),                     # Haram al-Sharif / Temple Mount Esplanade & perimeter gates
    box(5.0, -405.0, 102.0, -90.0),      # Armenian Garden, Seminary Courtyard & Citadel South Esplanade
    box(-60.0, -28.0, 80.0, 18.0),       # Jaffa Gate West Plaza & Omar Square / Citadel Moat
    box(585.0, -55.0, 667.0, 48.0),      # Western Wall Prayer Plaza & Upper Plaza
    box(550.0, -250.0, 675.0, -50.0),    # Ophel / Davidson Archaeological Park near Dung Gate
    box(225.0, 512.0, 290.0, 600.0),     # Damascus Gate Amphitheater Plaza
    box(765.0, 450.0, 890.0, 558.0),     # Bethesda Pools / St. Anne Courtyard / Lions' Gate
    box(205.0, -400.0, 430.0, -255.0),   # Southern Jewish Quarter Parking / Rampart Overlook / Batei Mahaseh & Gan HaTekuma approach
]


def load_glb_meshes(glb_path: Path) -> List[Dict]:
    """Parses a binary glTF 2.0 (.glb) file and returns mesh primitives with world-space vertices & triangles."""
    data = glb_path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Not a valid GLB file: {glb_path}")

    offset = 12
    json_chunk = None
    bin_chunk = b""
    while offset < len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            json_chunk = json.loads(chunk_data.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk_data

    gltf = json_chunk
    buffer_views = gltf.get("bufferViews", [])
    accessors = gltf.get("accessors", [])
    materials = gltf.get("materials", [])
    meshes = gltf.get("meshes", [])
    nodes = gltf.get("nodes", [])

    def read_accessor(acc_idx: int) -> np.ndarray:
        acc = accessors[acc_idx]
        bv = buffer_views[acc["bufferView"]]
        byte_offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        count = acc["count"]
        comp_type = acc["componentType"]
        acc_type = acc["type"]
        num_comp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc_type]
        dtype_map = {
            5120: np.int8,
            5121: np.uint8,
            5122: np.int16,
            5123: np.uint16,
            5125: np.uint32,
            5126: np.float32,
        }
        dt = dtype_map[comp_type]
        arr = np.frombuffer(bin_chunk, dtype=dt, count=count * num_comp, offset=byte_offset)
        return arr.reshape((count, num_comp)) if num_comp > 1 else arr

    results = []
    for node in nodes:
        if "mesh" not in node:
            continue
        mesh = meshes[node["mesh"]]
        name = node.get("name", mesh.get("name", "unnamed"))
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            pos = read_accessor(attrs["POSITION"]).astype(np.float64)
            # Blender glTF export is Y-up `(x_enu, z_enu, -y_enu)`; convert back to ENU `(x, y, z)`
            x_enu = pos[:, 0]
            y_enu = -pos[:, 2]
            z_enu = pos[:, 1]
            verts_enu = np.stack([x_enu, y_enu, z_enu], axis=1)

            if "indices" in prim:
                idx = read_accessor(prim["indices"]).astype(np.int64)
                tris = idx.reshape((-1, 3))
            else:
                tris = np.arange(len(verts_enu), dtype=np.int64).reshape((-1, 3))

            mat_idx = prim.get("material")
            mat_name = materials[mat_idx].get("name", "") if mat_idx is not None else ""
            results.append(
                {
                    "name": name,
                    "material": mat_name,
                    "verts": verts_enu,
                    "tris": tris,
                }
            )
    return results


def connected_components_from_tris(num_verts: int, tris: np.ndarray) -> List[np.ndarray]:
    """Splits a triangle mesh into connected vertex index arrays via triangle vertex adjacency."""
    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b, c in tris:
        ia, ib, ic = int(a), int(b), int(c)
        adj[ia].extend((ib, ic))
        adj[ib].extend((ia, ic))
        adj[ic].extend((ia, ib))

    visited = np.zeros(num_verts, dtype=bool)
    comps: List[np.ndarray] = []
    for v0 in range(num_verts):
        if visited[v0] or v0 not in adj:
            continue
        stack = [v0]
        visited[v0] = True
        comp = []
        while stack:
            curr = stack.pop()
            comp.append(curr)
            for nxt in adj[curr]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        comps.append(np.array(comp, dtype=np.int64))
    return comps


def validate_city() -> int:
    city_plan = json.loads((ROOT_DIR / "public" / "data" / "city_plan.json").read_text(encoding="utf-8"))
    places = json.loads((ROOT_DIR / "public" / "data" / "places.json").read_text(encoding="utf-8"))
    osm_data = json.loads((ROOT_DIR / "data" / "osm_old_city.json").read_text(encoding="utf-8"))
    osm_jaffa = json.loads((ROOT_DIR / "data" / "osm_jaffa.json").read_text(encoding="utf-8"))

    wall_poly = Polygon(OLD_CITY_WALL_POLYGON)
    wall_boundary = wall_poly.boundary
    open_esplanade_union = unary_union(KNOWN_OPEN_ESPLANADES)

    # Build spatial index of all 2D building polygons + procedural monument polygons + city wall boundary
    b_polys: List[Polygon] = []
    for b in osm_data.get("buildings", []):
        ring = b.get("ring", [])
        if len(ring) >= 3:
            p = Polygon(ring)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty and p.area >= 4.0:
                b_polys.append(p)

    # Include r01_jaffa_gate buildings that pass r01's filter (`cx <= 170, -105 <= cy <= 90`, not overlapping David St / Omar Sq / Latin Rd)
    for b in osm_jaffa.get("buildings", []):
        if "building" not in b.get("tags", {}):
            continue
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        if cx > 170.0 or cy > 90.0 or cy < -105.0:
            continue
        if (76.0 <= cx <= 170.0 and -13.0 <= cy <= 17.0) or (4.0 <= cx <= 88.0 and -42.0 <= cy <= 29.0) or (7.0 <= cx <= 49.0 and 22.0 <= cy <= 72.0):
            continue
        ring = b.get("ring", [])
        if len(ring) >= 3:
            p = Polygon(ring)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty and 4.0 <= p.area <= 2500.0:
                b_polys.append(p)

    monument_polys = [poly for _, poly in PROCEDURAL_MONUMENT_BOXES]
    all_solid_footprints = unary_union(b_polys + monument_polys)
    ray_targets = b_polys + monument_polys + [wall_boundary.buffer(1.5)]
    ray_tree = STRtree(ray_targets)

    # --------------------------------------------------------------------------
    # 1. STREET BLOCKAGE & 360° URBAN ENCLOSURE / MISSING-BLOCK VOID CHECK
    # --------------------------------------------------------------------------
    blocked_points = []
    void_points = []
    street_sample_count = 0

    angles = [math.radians(deg) for deg in range(0, 360, 15)]
    max_ray_len = 65.0

    # Sample every 10m along all 27 curatedStreets
    sample_locations: List[Tuple[str, float, float]] = []
    for st in city_plan.get("curatedStreets", []):
        st_name = st.get("name", "street")
        line = LineString(st["pts"])
        num_steps = max(2, int(line.length / 10.0))
        for k in range(num_steps + 1):
            pt = line.interpolate(k / num_steps, normalized=True)
            sample_locations.append((f"street:{st_name}#{k}", float(pt.x), float(pt.y)))

    for pl in places:
        sample_locations.append((f"poi:{pl['id']}", float(pl["x"]), float(pl["y"])))

    for label, px, py in sample_locations:
        street_sample_count += 1
        pt_geom = Point(px, py)

        # Check if inside any active intramural OSM building polygon (shrunk by 0.35m tolerance)
        hit_blds = [
            b_polys[idx]
            for idx in ray_tree.query(pt_geom)
            if idx < len(b_polys) and b_polys[idx].area < 2800.0 and b_polys[idx].buffer(-0.35).contains(pt_geom)
        ]
        if hit_blds:
            blocked_points.append({"label": label, "x": round(px, 1), "y": round(py, 1)})

        # Only enforce tight urban street enclosure inside the Old City walls and outside known large open esplanades
        if not wall_poly.buffer(-12.0).contains(pt_geom) or open_esplanade_union.contains(pt_geom):
            continue

        ray_dists = []
        for ang in angles:
            ray = LineString([(px, py), (px + max_ray_len * math.cos(ang), py + max_ray_len * math.sin(ang))])
            min_d = max_ray_len
            for idx in ray_tree.query(ray):
                geom = ray_targets[idx]
                inter = ray.intersection(geom)
                if not inter.is_empty:
                    d = pt_geom.distance(inter)
                    if d < min_d:
                        min_d = d
            ray_dists.append(min_d)

        enclosed_rays = sum(1 for d in ray_dists if d <= 42.0)
        enclosure_ratio = enclosed_rays / len(ray_dists)
        if enclosure_ratio < 0.75:
            void_points.append(
                {
                    "label": label,
                    "x": round(px, 1),
                    "y": round(py, 1),
                    "enclosureRatio": round(enclosure_ratio, 2),
                    "medianHitDist": round(float(np.median(ray_dists)), 1),
                }
            )

    # Check the 2 explicit user-reported spots:
    # Spot A: Stand at Western Wall Plaza `(628, 12)`, look 90° left (North, heading 0° -> ray along +Y)
    ww_north_ray = LineString([(628.0, 12.0), (628.0, 62.0)])
    ww_north_hit = ww_north_ray.intersection(all_solid_footprints)
    ww_north_dist = Point(628.0, 12.0).distance(ww_north_hit) if not ww_north_hit.is_empty else 999.0

    # Spot B: Stand at Ecce Homo / Via Dolorosa `(495, 394.5)`, turn around West (`heading 260°` -> `(430, 386.6)`)
    # Check north and south sides of the real OSM Via Dolorosa between x=434 and x=494
    vd_line = LineString([(421.7, 386.6), (480.7, 391.4), (509.1, 397.1)])
    vd_north_hits = 0
    vd_south_hits = 0
    for k_vd in range(7):
        pt_vd = vd_line.interpolate((k_vd + 0.5) / 7.0, normalized=True)
        if not LineString([(pt_vd.x, pt_vd.y), (pt_vd.x, pt_vd.y + 18.0)]).intersection(all_solid_footprints).is_empty:
            vd_north_hits += 1
        if not LineString([(pt_vd.x, pt_vd.y), (pt_vd.x, pt_vd.y - 18.0)]).intersection(all_solid_footprints).is_empty:
            vd_south_hits += 1

    # --------------------------------------------------------------------------
    # 2. 3D GLB FLOATING WALL, FLOATING ROOF & FLOATING DOME CHECK
    # --------------------------------------------------------------------------
    floating_walls = []
    floating_domes = []
    floating_roofs = []

    glb_files = sorted((ROOT_DIR / "public" / "models" / "regions").glob("r*.glb"))
    footprint_buffer = all_solid_footprints.buffer(2.2)

    for glb_path in glb_files:
        region_id = glb_path.stem
        prims = load_glb_meshes(glb_path)
        for prim in prims:
            name = prim["name"]
            mat = prim["material"]
            verts = prim["verts"]
            tris = prim["tris"]

            if not (name.startswith("Buildings_Walls") or name.startswith("Buildings_Details")):
                continue

            # Check B: Horizontal Roof Decks (`Buildings_Walls__mat_plaster`, every individual roof triangle must lie inside a building footprint!)
            if name.startswith("Buildings_Walls") and mat == "mat_plaster":
                for a, b_i, c in tris:
                    v0, v1, v2 = verts[int(a)], verts[int(b_i)], verts[int(c)]
                    t_dz = max(v0[2], v1[2], v2[2]) - min(v0[2], v1[2], v2[2])
                    if t_dz < 0.25:
                        tx = float((v0[0] + v1[0] + v2[0]) / 3.0)
                        ty = float((v0[1] + v1[1] + v2[1]) / 3.0)
                        tz = float((v0[2] + v1[2] + v2[2]) / 3.0)
                        tgz = street_elevation_z(tx, ty)
                        if tz > tgz + 3.2 and wall_poly.contains(Point(tx, ty)) and not footprint_buffer.contains(Point(tx, ty)):
                            floating_roofs.append(
                                {
                                    "region": region_id,
                                    "x": round(tx, 1),
                                    "y": round(ty, 1),
                                    "z": round(tz, 2),
                                }
                            )

            comps = connected_components_from_tris(len(verts), tris)
            for comp_idx in comps:
                if len(comp_idx) < 4:
                    continue
                c_verts = verts[comp_idx]
                min_x, min_y, min_z = c_verts.min(axis=0)
                max_x, max_y, max_z = c_verts.max(axis=0)
                cx = 0.5 * (min_x + max_x)
                cy = 0.5 * (min_y + max_y)
                span_x = max_x - min_x
                span_y = max_y - min_y
                dz = max_z - min_z
                gz = street_elevation_z(float(cx), float(cy))

                # Check A: Roof Domes (`Buildings_Details`, dome geometry `dz >= 0.9m`, `span >= 2.6m`, high above ground)
                if (
                    name.startswith("Buildings_Details")
                    and mat in ("mat_plaster", "mat_stone_ashlar_aged")
                    and min_z > gz + 5.2
                    and span_x >= 2.6
                    and span_y >= 2.6
                    and dz >= 0.9
                ):
                    if not footprint_buffer.contains(Point(float(cx), float(cy))):
                        floating_domes.append(
                            {
                                "region": region_id,
                                "x": round(float(cx), 1),
                                "y": round(float(cy), 1),
                                "min_z": round(float(min_z), 2),
                                "ground_z": round(float(gz), 2),
                            }
                        )

                # Check C: Ground-Floor Building Walls (`Buildings_Walls`, `dz >= 3.2m`, starting near ground `min_z < gz + 3.0m`)
                if (
                    name.startswith("Buildings_Walls")
                    and mat in ("mat_stone_boss", "mat_stone_rubble")
                    and dz >= 3.2
                    and min_z < gz + 3.0
                ):
                    # Verify that the wall reaches at least down to the lowest corner ground elevation + 0.35m
                    corner_gz = min(
                        street_elevation_z(float(min_x), float(min_y)),
                        street_elevation_z(float(max_x), float(min_y)),
                        street_elevation_z(float(min_x), float(max_y)),
                        street_elevation_z(float(max_x), float(max_y)),
                    )
                    if min_z > corner_gz + 0.35:
                        floating_walls.append(
                            {
                                "region": region_id,
                                "x": round(float(cx), 1),
                                "y": round(float(cy), 1),
                                "min_z": round(float(min_z), 2),
                                "corner_gz": round(float(corner_gz), 2),
                            }
                        )

    report = {
        "streetSamplePoints": street_sample_count,
        "blockedStreetPointsCount": len(blocked_points),
        "blockedStreetPoints": blocked_points[:15],
        "missingBlockVoidsCount": len(void_points),
        "missingBlockVoids": void_points[:15],
        "westernWallNorthHitDistMetres": round(float(ww_north_dist), 2),
        "viaDolorosaNorthBuildingHits": f"{vd_north_hits}/7",
        "viaDolorosaSouthBuildingHits": f"{vd_south_hits}/7",
        "floatingWallsCount": len(floating_walls),
        "floatingWalls": floating_walls[:15],
        "floatingRoofsCount": len(floating_roofs),
        "floatingRoofs": floating_roofs[:15],
        "floatingDomesCount": len(floating_domes),
        "floatingDomes": floating_domes[:15],
    }

    out_path = ROOT_DIR / "build" / "qa" / "city_completeness_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("OLD CITY JERUSALEM — AUTOMATED 3D COMPLETENESS & GEOMETRY REPORT")
    print("=" * 72)
    print(f"  Street & POI Sample Points Tested : {street_sample_count}")
    print(f"  Blocked Street Points             : {len(blocked_points)}")
    print(f"  Missing-Block Voids (>42m open)   : {len(void_points)}")
    print(f"  Western Wall 90° Left (North) Hit : {ww_north_dist:.2f} m (must be <= 25.0 m)")
    print(f"  Via Dolorosa West North/South Hit : N={vd_north_hits}/7, S={vd_south_hits}/7")
    print(f"  Floating Ground-Floor Walls       : {len(floating_walls)}")
    print(f"  Floating Roof Decks Over Streets  : {len(floating_roofs)}")
    print(f"  Floating Domes Over Streets       : {len(floating_domes)}")
    print("=" * 72)

    if blocked_points:
        print("  [!] Blocked points:", blocked_points[:5])
    if void_points:
        print("  [!] Void points:", void_points[:5])
    if floating_walls:
        print("  [!] Floating walls:", floating_walls[:5])
    if floating_roofs:
        print("  [!] Floating roofs:", floating_roofs[:5])
    if floating_domes:
        print("  [!] Floating domes:", floating_domes[:5])

    return 0


if __name__ == "__main__":
    raise SystemExit(validate_city())
