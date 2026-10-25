#!/usr/bin/env python3
"""
pipeline/generate_complete_city_footprints.py

Goes back to basics on the Old City of Jerusalem general outline and building placement:
1. Cleans `data/osm_old_city.json` of non-building boundary relations (`5862586`, `7880713`,
   archaeological dig bounds) that previously caused false coverage counts or giant polygons.
2. Retains all ~2,250 genuine OpenStreetMap building footprints inside the Old City Walls,
   and subdivides valid oversized (`> 58m`) OSM compound blocks into courtyard wings instead
   of discarding them.
3. Uses the complete OpenStreetMap street & alley graph (`streets` in `data/osm_old_city.json`
   + primary Old City thoroughfares) to generate:
     - Pass A: Street-fronting, street-aligned Jerusalem building blocks along every mapped
       OSM street and alley that lacked building footprints in OSM (especially in the NE Muslim
       Quarter / Bab Hutta / Herod's Gate / Via Dolorosa, NW Christian Quarter / New Gate,
       SW Armenian Quarter, and Southern Jewish Quarter / Dung Gate).
     - Pass B: Urban-block interior courtyard buildings filling unmapped block interiors between
       streets while preserving courtyard lanes and all historic open spaces/gardens/plazas.
4. Writes the complete building dataset to `data/osm_old_city.json` and renders a 2D diagnostic
   plan (`build/qa/osm_complete_city_outline.png`) for visual verification.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from build_city_plan import OLD_CITY_WALL_POLYGON, build_city_plan, point_in_polygon_2d
from fetch_osm_and_dem import classify_region, street_elevation_z

Vec2 = Tuple[float, float]

# Non-building OSM ways/relations (boundary polygons, open archaeological parks, pools, malls outside wall)
NON_BUILDING_OSM_IDS = {
    5862586,     # Ottoman City Wall relation
    7880713,     # UNESCO World Heritage boundary relation
    147608623,   # City of David archaeological park outside walls
    147608637,   # Givati Parking Lot dig outside walls
    290701310,   # Garden of the Redemption open archaeological park
    126086870,   # Hezekiah's Pool water basin
    126864878,   # Pool of Bethesda open basin
    126864879,   # Pool of Bethesda
    158063844,   # Mamilla Mall outside walls
    158063849,   # Mamilla Mall outside walls
    19253751,    # Dome of the Rock site relation (custom built in r07)
    280309331,   # Al-Aqsa Mosque (custom built in r07)
    1022672085,  # Church of the Holy Sepulchre (custom built in r02)
    136164,      # Tower of David relation (custom built in r01)
    1080613977,  # Eastern Temple Mount wall way
}


def dist_point_to_segment(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    dx, dy = x1 - x0, y1 - y0
    l2 = dx * dx + dy * dy
    if l2 < 1e-8:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / l2))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def dist_to_wall_perimeter(px: float, py: float) -> float:
    n = len(OLD_CITY_WALL_POLYGON)
    best = 1e9
    for i in range(n):
        x0, y0 = OLD_CITY_WALL_POLYGON[i]
        x1, y1 = OLD_CITY_WALL_POLYGON[(i + 1) % n]
        d = dist_point_to_segment(px, py, x0, y0, x1, y1)
        if d < best:
            best = d
    return best


def is_in_open_reserve(x: float, y: float) -> bool:
    """
    Returns True if (x, y) is inside a historic open esplanade, plaza, garden, or custom
    monument footprint where generic urban infill buildings must NOT be placed.
    """
    # 1. Temple Mount / Haram al-Sharif Esplanade & Southern/Eastern Wall Apron
    if x >= 662.0 and -210.0 <= y <= 388.0:
        return True
    # 2. Western Wall Prayer Plaza & Upper Plaza & Mughrabi Ramp
    if 574.0 <= x <= 668.0 and -68.0 <= y <= 62.0:
        return True
    # 3. Jaffa Gate, Citadel, Moat & Omar Square core (`r01` already has dense curated + OSM buildings)
    if x <= 160.0 and -92.0 <= y <= 62.0:
        return True
    # 4. Armenian Pine Garden (Rose Garden) along the western city wall
    if 6.0 <= x <= 50.0 and -365.0 <= y <= -90.0:
        return True
    # 5. Church of the Holy Sepulchre, Mosque of Omar & 360° Surround
    if 160.0 <= x <= 264.0 and 146.0 <= y <= 245.0:
        return True
    # 6. Muristan Fountain & Lutheran Church of the Redeemer & 360° Surround
    if 214.0 <= x <= 298.0 and 114.0 <= y <= 176.0:
        return True
    # 7. St. James Cathedral Compound & 360° Surround
    if 104.0 <= x <= 166.0 and -255.0 <= y <= -200.0:
        return True
    # 8. Hurva Synagogue & Hurva Square & 360° Surround
    if 324.0 <= x <= 386.0 and -172.0 <= y <= -96.0:
        return True
    # 9. Four Sephardic Synagogues & Tiferet Yisrael & Batei Mahaseh Square
    if 340.0 <= x <= 392.0 and -258.0 <= y <= -196.0:
        return True
    if 462.0 <= x <= 520.0 and -190.0 <= y <= -140.0:
        return True
    if 418.0 <= x <= 456.0 and -324.0 <= y <= -294.0:
        return True
    # 10. Damascus Gate Amphitheater & Inner Fork Plaza
    if 228.0 <= x <= 286.0 and 516.0 <= y <= 595.0:
        return True
    # 11. Herod's Gate & Lions' Gate & Zion Gate & New Gate & Dung Gate immediate portals
    if 550.0 <= x <= 586.0 and 678.0 <= y <= 725.0:
        return True
    if 852.0 <= x <= 895.0 and 452.0 <= y <= 482.0:
        return True
    if 164.0 <= x <= 208.0 and -428.0 <= y <= -388.0:
        return True
    if -156.0 <= x <= -122.0 and 280.0 <= y <= 314.0:
        return True
    if 594.0 <= x <= 636.0 and -225.0 <= y <= -180.0:
        return True
    # 12. Crusader Church of St. Anne & Pools of Bethesda & 360° Surround
    if 768.0 <= x <= 868.0 and 496.0 <= y <= 558.0:
        return True
    # 13. Austrian Hospice & Ecce Homo Arch & 360° Surround
    if 418.0 <= x <= 564.0 and 382.0 <= y <= 430.0:
        return True
    # 14. Stork Tower (Burj al-Laqlaq) NE Bastion Plaza
    if 792.0 <= x <= 835.0 and 735.0 <= y <= 782.0:
        return True
    # 15. Hezekiah's Pool (Patriarch's Pool) open basin in Christian Quarter
    if 114.0 <= x <= 154.0 and 34.0 <= y <= 102.0:
        return True
    return False


def boxes_overlap_2d(b0: Tuple[float, float, float, float], b1: Tuple[float, float, float, float], margin: float = 0.0) -> bool:
    return not (
        b0[2] + margin < b1[0]
        or b0[0] - margin > b1[2]
        or b0[3] + margin < b1[1]
        or b0[1] - margin > b1[3]
    )


def poly_bbox(ring: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in ring]
    ys = [float(p[1]) for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def subdivide_large_building_box(
    b: dict,
    next_id: int,
) -> List[dict]:
    """
    Subdivides a genuine large (> 58m) OSM compound/block polygon into 2–4 smaller
    courtyard/wing building footprints (`<= 42m` span) around its perimeter so it is
    never dropped by the regional builder.
    """
    ring = b["ring"]
    x0, y0, x1, y1 = poly_bbox(ring)
    sx, sy = x1 - x0, y1 - y0
    nx = max(1, int(math.ceil(sx / 36.0)))
    ny = max(1, int(math.ceil(sy / 36.0)))
    dx = sx / nx
    dy = sy / ny
    out = []
    for ix in range(nx):
        for iy in range(ny):
            # Leave a central courtyard if >= 3x3
            if nx >= 3 and ny >= 3 and 0 < ix < nx - 1 and 0 < iy < ny - 1:
                continue
            bx0 = x0 + ix * dx + 0.8
            by0 = y0 + iy * dy + 0.8
            bx1 = x0 + (ix + 1) * dx - 0.8
            by1 = y0 + (iy + 1) * dy - 0.8
            cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
            if not point_in_polygon_2d((cx, cy), ring):
                continue
            if is_in_open_reserve(cx, cy):
                continue
            sub_ring = [
                [round(bx0, 2), round(by0, 2)],
                [round(bx1, 2), round(by0, 2)],
                [round(bx1, 2), round(by1, 2)],
                [round(bx0, 2), round(by1, 2)],
                [round(bx0, 2), round(by0, 2)],
            ]
            reg = classify_region(cx, cy)
            lvls = max(2, min(3, int(b.get("levels", 2))))
            out.append(
                {
                    "osm_id": next_id,
                    "type": "way",
                    "region": reg,
                    "name": b.get("name", ""),
                    "levels": lvls,
                    "height": round(lvls * 3.6, 2),
                    "base_z": street_elevation_z(cx, cy),
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "ring": sub_ring,
                    "tags": {"building": "yes", "source": "osm_subdivided"},
                }
            )
            next_id += 1
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    osm_path = root / "data" / "osm_old_city.json"
    with open(osm_path, "r", encoding="utf-8") as f:
        osm_data = json.load(f)

    # 1. Collect all street/alley segments (OSM + curated thoroughfares) and build `carve_out_union`
    plan = build_city_plan()
    street_segments: List[Tuple[float, float, float, float, float]] = []
    carve_geoms = [
        # Custom Landmark & 360° Surround Plazas
        box(662.0, -210.0, 910.0, 388.0),   # Temple Mount Esplanade
        box(574.0, -68.0, 668.0, 62.0),     # Western Wall Plaza
        box(160.0, 146.0, 264.0, 245.0),    # Holy Sepulchre, Mosque of Omar & 360° surround
        box(214.0, 114.0, 316.0, 176.0),    # Muristan & Lutheran Church of the Redeemer & 360° surround (including East Apse plaza)
        box(104.0, -255.0, 166.0, -200.0),  # St. James Cathedral & 360° surround
        box(324.0, -172.0, 386.0, -96.0),    # Hurva Synagogue & Hurva Square & 360° surround
        box(340.0, -258.0, 392.0, -196.0),  # Four Sephardic Synagogues & surround
        box(462.0, -190.0, 520.0, -140.0),  # Tiferet Yisrael & 360° surround
        box(418.0, -324.0, 456.0, -294.0),  # Batei Mahaseh Square
        box(296.0, -198.0, 326.0, -65.0),   # Southern Roman Cardo Colonnade
        box(295.0, 6.0, 332.0, 112.0),      # Crusader Vaulted Triple Souq
        box(432.0, 34.0, 458.0, 56.0),      # Mamluk Turba West
        box(512.0, 52.0, 538.0, 76.0),      # Mamluk Turba East
        box(558.0, 134.0, 668.0, 156.0),    # Suq al-Qattanin
        box(228.0, 514.0, 288.0, 595.0),    # Damascus Gate Plaza & Inner Fork
        box(550.0, 678.0, 586.0, 725.0),    # Herod's Gate Portal
        box(852.0, 452.0, 895.0, 482.0),    # Lions' Gate Portal
        box(768.0, 496.0, 868.0, 558.0),    # St. Anne Church & Bethesda Pools & 360° surround
        box(418.0, 382.0, 564.0, 430.0),    # Austrian Hospice & Ecce Homo Arch & 360° surround
        box(648.0, 455.0, 752.0, 469.0),    # Eastern Via Dolorosa custom corridor
        box(792.0, 738.0, 835.0, 782.0),    # Stork Tower NE Bastion
        box(164.0, -428.0, 208.0, -388.0),  # Zion Gate
        box(-156.0, 280.0, -122.0, 314.0),  # New Gate
        box(594.0, -225.0, 636.0, -180.0),  # Dung Gate
        box(6.0, -365.0, 50.0, -90.0),      # Armenian Pine Garden
        box(114.0, 34.0, 154.0, 102.0),     # Hezekiah's Pool
    ]

    for s in osm_data.get("streets", []):
        pts = s.get("pts", [])
        if len(pts) < 2:
            continue
        hw = float(s.get("width", 3.8)) * 0.5
        clearance = max(2.2, min(4.0, hw + 0.30))
        coords_2d = [(float(p[0]), float(p[1])) for p in pts]
        if len(coords_2d) >= 2:
            ls = LineString(coords_2d)
            if ls.length >= 1.5:
                carve_geoms.append(ls.buffer(max(1.85, min(3.0, hw)), cap_style=2, join_style=2))
        for i in range(len(pts) - 1):
            x0, y0 = float(pts[i][0]), float(pts[i][1])
            x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
            if math.hypot(x1 - x0, y1 - y0) >= 1.0:
                street_segments.append((x0, y0, x1, y1, clearance))

    for cs in plan["curatedStreets"]:
        pts = cs["pts"]
        clearance = 3.8 if cs["tier"] == "primary" else 3.0
        coords_2d = [(float(p[0]), float(p[1])) for p in pts]
        if len(coords_2d) >= 2:
            ls = LineString(coords_2d)
            if ls.length >= 1.5:
                carve_geoms.append(ls.buffer(3.6 if cs["tier"] == "primary" else 3.0, cap_style=2, join_style=2))
        for i in range(len(pts) - 1):
            x0, y0 = float(pts[i][0]), float(pts[i][1])
            x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
            if math.hypot(x1 - x0, y1 - y0) >= 1.0:
                street_segments.append((x0, y0, x1, y1, clearance))

    # Also carve all custom `_build_souq_street_corridor` corridors built in `regions_builder.py`
    custom_souq_corridors = [
        [(132.0, 78.0), (132.0, 165.0)],
        [(82.0, 185.0), (154.0, 185.0)],
        [(-20.0, 250.0), (35.0, 195.0)],
        [(305.0, 162.0), (305.0, 252.0)],
        [(510.0, 72.0), (510.0, 235.0)],
        [(565.0, 145.0), (662.0, 145.0)],
        [(332.0, 9.2), (428.0, 17.6)],
        [(172.0, 3.5), (294.0, 7.5)],
        [(432.0, 33.5), (595.0, 67.5)],
        [(52.0, -105.0), (52.0, -205.0)],
        [(118.0, -265.0), (168.0, -395.0)],
        [(615.0, -178.0), (615.0, -108.0)],
        [(650.0, 462.0), (750.0, 462.0)],
        [(504.0, 413.4), (562.0, 413.4)],
    ]
    for c_pts in custom_souq_corridors:
        carve_geoms.append(LineString(c_pts).buffer(5.2, cap_style=2, join_style=2))
        street_segments.append((c_pts[0][0], c_pts[0][1], c_pts[1][0], c_pts[1][1], 5.2))

    # Also carve 4.8m safety circles around all POI & landmark 360 camera viewpoints
    for pose_file in (
        root / "public" / "data" / "places.json",
        root / "build" / "qa" / "poses_landmarks_360.json",
    ):
        if pose_file.exists():
            pdata = json.loads(pose_file.read_text(encoding="utf-8"))
            plist = pdata.get("places", pdata) if isinstance(pdata, dict) else pdata
            for p in plist:
                if "x" in p and "y" in p:
                    carve_geoms.append(Point(float(p["x"]), float(p["y"])).buffer(4.8))

    carve_out_union = unary_union(carve_geoms)
    print(f"[1/4] Indexed {len(street_segments)} street & alley segments and built unified 2D street/landmark carve-out mask.")

    # 2. Filter, carve & keep genuine OSM buildings, subdividing genuine >58m compound blocks
    raw_buildings = osm_data.get("buildings", [])
    clean_buildings: List[dict] = []
    occupied_boxes: List[Tuple[float, float, float, float, List[List[float]]]] = []
    next_synth_id = 9000000

    kept_osm = 0
    carved_osm = 0
    subdivided_osm = 0

    for b in raw_buildings:
        oid = int(b.get("osm_id", 0))
        if oid in NON_BUILDING_OSM_IDS:
            continue
        if b.get("tags", {}).get("source") in ("osm_street_infill", "osm_block_infill", "osm_subdivided", "osm_carved"):
            # Skip previously generated infill if re-running idempotently
            continue
        ring = b.get("ring", [])
        if len(ring) < 3:
            continue
        x0, y0, x1, y1 = poly_bbox(ring)
        sx, sy = x1 - x0, y1 - y0
        cx, cy = b["centroid"]

        # Outside Old City Walls: keep normal buildings, skip giant polygons
        if not point_in_polygon_2d((cx, cy), OLD_CITY_WALL_POLYGON):
            if sx <= 65.0 and sy <= 65.0:
                clean_buildings.append(b)
            continue

        # Carve out all streets, alleys, landmarks & camera plazas from every intramural OSM building!
        try:
            poly = Polygon([(float(p[0]), float(p[1])) for p in ring])
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            continue

        if poly.is_empty or poly.area < 14.0:
            continue

        diff = poly.difference(carve_out_union)
        if diff.is_empty:
            continue

        sub_polys = list(diff.geoms) if isinstance(diff, MultiPolygon) else ([diff] if isinstance(diff, Polygon) else [g for g in getattr(diff, "geoms", []) if isinstance(g, Polygon)])
        was_carved = (len(sub_polys) > 1) or (abs(diff.area - poly.area) > 1.5)

        for sp_idx, sp in enumerate(sub_polys):
            sp_clean = sp.simplify(0.22, preserve_topology=True)
            if not isinstance(sp_clean, Polygon) or sp_clean.is_empty or sp_clean.area < 18.0:
                continue
            bx0, by0, bx1, by1 = sp_clean.bounds
            bsx, bsy = bx1 - bx0, by1 - by0
            if min(bsx, bsy) < 2.8:
                continue
            rep_pt = sp_clean.representative_point()
            scx, scy = float(rep_pt.x), float(rep_pt.y)
            if not point_in_polygon_2d((scx, scy), OLD_CITY_WALL_POLYGON) or is_in_open_reserve(scx, scy):
                continue

            ext_coords = [[round(float(x), 2), round(float(y), 2)] for x, y in sp_clean.exterior.coords]
            if len(ext_coords) < 4:
                continue

            b_entry = dict(b)
            b_entry["osm_id"] = oid if (sp_idx == 0 and not was_carved) else next_synth_id
            if sp_idx > 0 or was_carved:
                next_synth_id += 1
                carved_osm += 1
            b_entry["region"] = classify_region(scx, scy)
            b_entry["centroid"] = [round(scx, 2), round(scy, 2)]
            b_entry["base_z"] = street_elevation_z(scx, scy)
            b_entry["ring"] = ext_coords

            if bsx > 58.0 or bsy > 58.0:
                if bsx <= 135.0 and bsy <= 135.0:
                    subs = subdivide_large_building_box(b_entry, next_synth_id)
                    next_synth_id += len(subs) + 1
                    for sb in subs:
                        clean_buildings.append(sb)
                        sb_box = poly_bbox(sb["ring"])
                        occupied_boxes.append((*sb_box, sb["ring"]))
                        subdivided_osm += 1
                continue

            clean_buildings.append(b_entry)
            occupied_boxes.append((bx0, by0, bx1, by1, ext_coords))
            kept_osm += 1

    print(f"[2/4] Kept {kept_osm} street-carved intramural OSM buildings ({carved_osm} sliced along streets/plazas) + {subdivided_osm} subdivided compound wings.")

    def is_footprint_valid(pts: List[List[float]], min_bldg_gap: float = 0.6) -> bool:
        bx0, by0, bx1, by1 = poly_bbox(pts)
        cx = 0.25 * sum(p[0] for p in pts[:4])
        cy = 0.25 * sum(p[1] for p in pts[:4])

        if not point_in_polygon_2d((cx, cy), OLD_CITY_WALL_POLYGON):
            return False
        if is_in_open_reserve(cx, cy):
            return False
        if Polygon(pts).intersects(carve_out_union):
            return False

        # Check all 4 corners + midpoint against Old City Wall & open reserves
        sample_pts = pts[:4] + [
            [cx, cy],
            [0.5 * (pts[0][0] + pts[1][0]), 0.5 * (pts[0][1] + pts[1][1])],
            [0.5 * (pts[1][0] + pts[2][0]), 0.5 * (pts[1][1] + pts[2][1])],
            [0.5 * (pts[2][0] + pts[3][0]), 0.5 * (pts[2][1] + pts[3][1])],
            [0.5 * (pts[3][0] + pts[0][0]), 0.5 * (pts[3][1] + pts[0][1])],
        ]
        for sp in sample_pts:
            if not point_in_polygon_2d((sp[0], sp[1]), OLD_CITY_WALL_POLYGON):
                return False
            if is_in_open_reserve(sp[0], sp[1]):
                return False
            if dist_to_wall_perimeter(sp[0], sp[1]) < 4.8:
                return False

        # Check against street segments
        for sx0, sy0, sx1, sy1, s_rad in street_segments:
            if max(sx0, sx1) + s_rad < bx0 or min(sx0, sx1) - s_rad > bx1:
                continue
            if max(sy0, sy1) + s_rad < by0 or min(sy0, sy1) - s_rad > by1:
                continue
            for sp in sample_pts:
                if dist_point_to_segment(sp[0], sp[1], sx0, sy0, sx1, sy1) < s_rad:
                    return False

        # Check against existing buildings
        for ox0, oy0, ox1, oy1, oring in occupied_boxes:
            if not boxes_overlap_2d((bx0, by0, bx1, by1), (ox0, oy0, ox1, oy1), margin=min_bldg_gap):
                continue
            for sp in sample_pts:
                if point_in_polygon_2d((sp[0], sp[1]), oring):
                    return False
                for k in range(len(oring) - 1):
                    if dist_point_to_segment(sp[0], sp[1], oring[k][0], oring[k][1], oring[k + 1][0], oring[k + 1][1]) < min_bldg_gap:
                        return False
            ocx = 0.5 * (ox0 + ox1)
            ocy = 0.5 * (oy0 + oy1)
            if point_in_polygon_2d((ocx, ocy), pts):
                return False

        return True

    # 3. Pass A: Generate Street-Hugging Building Blocks along every OSM street/alley that lacks buildings
    street_infill_count = 0
    for seg_idx, (sx0, sy0, sx1, sy1, s_rad) in enumerate(street_segments):
        dx, dy = sx1 - sx0, sy1 - sy0
        seg_len = math.hypot(dx, dy)
        if seg_len < 6.2:
            continue
        tx, ty = dx / seg_len, dy / seg_len
        nx, ny = -ty, tx

        step_along = 8.8
        n_steps = max(1, int(round(seg_len / step_along)))
        actual_step = seg_len / n_steps

        for k in range(n_steps):
            u0 = k * actual_step + 0.25
            u1 = (k + 1) * actual_step - 0.25
            w_bldg = u1 - u0
            if w_bldg < 5.2:
                continue

            for side in (-1.0, 1.0):
                d_near = s_rad + 0.25
                for depth in (10.2, 7.6, 5.6, 4.2):
                    d_far = d_near + depth
                    snx, sny = nx * side, ny * side
                    # Build CCW rectangle aligned to the exact street segment tangent (tx, ty)
                    if side > 0:
                        p0 = [round(sx0 + tx * u0 + snx * d_near, 2), round(sy0 + ty * u0 + sny * d_near, 2)]
                        p1 = [round(sx0 + tx * u1 + snx * d_near, 2), round(sy0 + ty * u1 + sny * d_near, 2)]
                        p2 = [round(sx0 + tx * u1 + snx * d_far, 2), round(sy0 + ty * u1 + sny * d_far, 2)]
                        p3 = [round(sx0 + tx * u0 + snx * d_far, 2), round(sy0 + ty * u0 + sny * d_far, 2)]
                    else:
                        p0 = [round(sx0 + tx * u1 + snx * d_near, 2), round(sy0 + ty * u1 + sny * d_near, 2)]
                        p1 = [round(sx0 + tx * u0 + snx * d_near, 2), round(sy0 + ty * u0 + sny * d_near, 2)]
                        p2 = [round(sx0 + tx * u0 + snx * d_far, 2), round(sy0 + ty * u0 + sny * d_far, 2)]
                        p3 = [round(sx0 + tx * u1 + snx * d_far, 2), round(sy0 + ty * u1 + sny * d_far, 2)]

                    ring = [p0, p1, p2, p3, p0]
                    if is_footprint_valid(ring, min_bldg_gap=0.25):
                        cx = round(0.25 * (p0[0] + p1[0] + p2[0] + p3[0]), 2)
                        cy = round(0.25 * (p0[1] + p1[1] + p2[1] + p3[1]), 2)
                        reg = classify_region(cx, cy)
                        lvls = 2 + ((seg_idx + k + (1 if side > 0 else 0)) % 2)
                        b_obj = {
                            "osm_id": next_synth_id,
                            "type": "way",
                            "region": reg,
                            "name": "",
                            "levels": lvls,
                            "height": round(lvls * 3.6, 2),
                            "base_z": street_elevation_z(cx, cy),
                            "centroid": [cx, cy],
                            "ring": ring,
                            "tags": {"building": "yes", "source": "osm_street_infill"},
                        }
                        clean_buildings.append(b_obj)
                        occupied_boxes.append((*poly_bbox(ring), ring))
                        next_synth_id += 1
                        street_infill_count += 1
                        break

    print(f"[3/4] Generated {street_infill_count} street-hugging Jerusalem buildings along unmapped OSM streets.")

    # 4. Pass B: Urban-Block Interior Courtyard Houses in unmapped block interiors between streets
    block_infill_count = 0
    grid_step = 11.5
    for gx in range(-180, 895, int(grid_step)):
        for gy in range(-415, 770, int(grid_step)):
            cx = gx + 0.5 * grid_step
            cy = gy + 0.5 * grid_step
            if not point_in_polygon_2d((cx, cy), OLD_CITY_WALL_POLYGON):
                continue
            if is_in_open_reserve(cx, cy):
                continue
            h_seed = abs((gx * 73856093) ^ (gy * 19349663))
            ang = math.radians(-8.0 if cy > 250 else (4.0 if cx < 200 else 0.0))
            ca, sa = math.cos(ang), math.sin(ang)

            for hw_try, hd_try in ((5.2, 4.6), (4.2, 3.8), (3.4, 3.2)):
                corners_local = [(-hw_try, -hd_try), (hw_try, -hd_try), (hw_try, hd_try), (-hw_try, hd_try)]
                ring = [
                    [round(cx + lx * ca - ly * sa, 2), round(cy + lx * sa + ly * ca, 2)]
                    for lx, ly in corners_local
                ]
                ring.append(ring[0])

                if is_footprint_valid(ring, min_bldg_gap=0.65):
                    reg = classify_region(cx, cy)
                    lvls = 2 + (h_seed % 2)
                    b_obj = {
                        "osm_id": next_synth_id,
                        "type": "way",
                        "region": reg,
                        "name": "",
                        "levels": lvls,
                        "height": round(lvls * 3.6, 2),
                        "base_z": street_elevation_z(cx, cy),
                        "centroid": [round(cx, 2), round(cy, 2)],
                        "ring": ring,
                        "tags": {"building": "yes", "source": "osm_block_infill"},
                    }
                    clean_buildings.append(b_obj)
                    occupied_boxes.append((*poly_bbox(ring), ring))
                    next_synth_id += 1
                    block_infill_count += 1
                    break

    print(f"[4/4] Generated {block_infill_count} interior urban-block courtyard buildings.")

    # Update region counts & save data/osm_old_city.json
    reg_counts: Dict[str, int] = {}
    for b in clean_buildings:
        r = b.get("region", "unknown")
        reg_counts[r] = reg_counts.get(r, 0) + 1

    osm_data["buildings"] = clean_buildings
    osm_data["metadata"]["counts"]["buildings"] = len(clean_buildings)
    osm_data["metadata"]["counts"]["buildings_by_region"] = reg_counts

    with open(osm_path, "w", encoding="utf-8") as f:
        json.dump(osm_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Updated {osm_path} with {len(clean_buildings)} total buildings:")
    print(json.dumps(reg_counts, indent=2))

    # Render diagnostic plan build/qa/osm_complete_city_outline.png
    W, H = 1400, 1400
    img = Image.new("RGB", (W, H), (16, 20, 28))
    draw = ImageDraw.Draw(img)

    def to_px(x: float, y: float) -> Tuple[int, int]:
        px = int((x - (-220.0)) / 1200.0 * (W - 80) + 40)
        py = int((800.0 - y) / 1260.0 * (H - 80) + 40)
        return (px, py)

    wall_px = [to_px(p[0], p[1]) for p in OLD_CITY_WALL_POLYGON] + [to_px(*OLD_CITY_WALL_POLYGON[0])]
    draw.polygon(wall_px, fill=(24, 30, 42), outline=(240, 190, 90))

    for b in clean_buildings:
        cx, cy = b["centroid"]
        if not point_in_polygon_2d((cx, cy), OLD_CITY_WALL_POLYGON):
            continue
        pts = [to_px(p[0], p[1]) for p in b["ring"]]
        src = b.get("tags", {}).get("source", "osm")
        if src == "osm_street_infill":
            draw.polygon(pts, fill=(175, 148, 105), outline=(225, 195, 140))
        elif src == "osm_block_infill":
            draw.polygon(pts, fill=(145, 128, 98), outline=(195, 175, 135))
        elif src == "osm_subdivided":
            draw.polygon(pts, fill=(165, 135, 115), outline=(215, 180, 155))
        else:
            draw.polygon(pts, fill=(135, 122, 102), outline=(190, 175, 148))

    for sx0, sy0, sx1, sy1, _ in street_segments:
        draw.line([to_px(sx0, sy0), to_px(sx1, sy1)], fill=(65, 210, 255), width=2)

    out_img = root / "build" / "qa" / "osm_complete_city_outline.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_img)
    print(f"[OK] Saved diagnostic city outline map to {out_img}")


if __name__ == "__main__":
    main()
