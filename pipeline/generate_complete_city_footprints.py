#!/usr/bin/env python3
"""
pipeline/generate_complete_city_footprints.py

Topological Street-Bounded Urban Block (`Insula`) Parcelizer & Over-Street Bridge-Building
Processor for the Old City of Jerusalem:

1. Computes the exact walkable space inside the 16th-century Ottoman City Walls strictly from:
     - Real OpenStreetMap street/alley centerlines (`raw["streets"]`, `1.38m`–`1.90m` half-width)
     - Curated primary/secondary street polylines (`plan["curatedStreets"]`, `1.95m`–`2.20m` half-width)
     - The Haram al-Sharif / Temple Mount inner esplanade (`tm_wall_poly.buffer(-4.5)`)
     - The Western Wall Plaza and true historic courtyards/gardens/custom monument footprints
2. Subtracts the walkable space from `city_wall_poly` to obtain the continuous solid urban blocks
   (`Insulae`) covering >81% of the 4 Quarters with ZERO walkable gaps between buildings and
   ZERO gaps between buildings and street lines.
3. Recursively bisects each `Insula` perpendicular to its local `minimum_rotated_rectangle` (OBB)
   into ~4,000 contiguous, street-hugging building parcels (`18..230 m²`), transferring height/tags
   from overlapping raw OpenStreetMap buildings (`data/osm_old_city_raw.json`).
4. Extracts all intramural OpenStreetMap `tunnel=yes` / `tunnel=building_passage` / `covered=yes`
   street segments plus periodic over-street bridge-building spans (`Qantara`) into
   `osm_data["tunnel_passages"]` so streets pass through vaulted stone tunnels beneath upper-storey
   bridge-buildings.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import polylabel, split, unary_union
from shapely.strtree import STRtree

from build_city_plan import build_city_plan
from fetch_osm_and_dem import classify_region, street_elevation_z

Vec2 = Tuple[float, float]

# Tight, exact custom-built monument footprints & genuine historic open plazas/gardens
EXACT_RESERVE_RECTS: List[Tuple[float, float, float, float]] = [
    # 1. Jaffa Gate, West Plaza, Omar Square, Citadel & Hero Corridors in `r01`
    (-55.0, -24.0, 82.0, 26.0),
    (6.0, -92.0, 66.0, -14.0),
    (10.0, 22.0, 44.0, 58.5),
    (76.0, -11.0, 158.5, 14.0),
    # 2. Church of the Holy Sepulchre (`170..219, 159..213`) & Mosque of Omar (`180..199, 136..151`)
    (170.0, 159.0, 219.0, 213.0),
    (180.0, 136.0, 199.0, 151.0),
    # 3. Muristan Fountain (`224..240, 120..136`) & Lutheran Church of the Redeemer (`264..295, 132..151.5`)
    (224.0, 120.0, 240.0, 136.0),
    (264.0, 132.0, 295.0, 151.5),
    # 4. St. James Cathedral & Parvis (`117.5..147.5, -255.0..-215.5`) & Armenian Garden (`14.0..42.0, -350.0..-180.0`)
    (117.5, -255.0, 147.5, -215.5),
    (14.0, -350.0, 42.0, -180.0),
    # 5. Hurva Synagogue & Hurva Square (`333.5..360.5, -152.5..-103.5`), Golden Menorah (`375.5..381.5, -116.5..-111.0`), & Sidna Omar Minaret (`363.5..369.5, -145.5..-139.5`)
    (333.5, -152.5, 360.5, -103.5),
    (375.5, -116.5, 381.5, -111.0),
    (363.5, -145.5, 369.5, -139.5),
    # 6. Four Sephardic Synagogues (`354.5..378.5, -246.5..-227.5`), Tiferet Yisrael (`430.0..448.5, -146.5..-125.0`), Batei Mahaseh (`426..451, -333..-308`), Broad Wall (`329.5..342.0, -90.0..-68.0`)
    (354.5, -246.5, 378.5, -227.5),
    (430.0, -146.5, 448.5, -125.0),
    (426.0, -333.0, 451.0, -308.0),
    (329.5, -90.0, 342.0, -68.0),
    # 7. Southern Roman Cardo Colonnade (`299.5..317.0, -190.0..-56.0`) & Crusader Vaulted Triple Souq (`297.5..323.5, -36.0..121.0`)
    (299.5, -190.0, 317.0, -56.0),
    (297.5, -36.0, 323.5, 121.0),
    # 8. Mamluk Turba West (`439.5..456.2, 26.5..38.0`), Mamluk Turba East (`516.5..533.5, 39.0..51.0`), Suq al-Qattanin (`522.0..614.0, 139.0..155.0`), & Western Wall / Beit Strauss (`588.0..668.0, -36.0..65.5`)
    (439.5, 26.5, 456.2, 38.0),
    (516.5, 39.0, 533.5, 51.0),
    (522.0, 139.0, 614.0, 155.0),
    (588.0, -36.0, 643.0, 34.0),
    (634.0, -36.0, 668.0, 47.5),
    (641.0, 47.5, 668.0, 65.5),
    # 9. Damascus Gate Forecourt (`242..270, 544..588`)
    (242.0, 544.0, 270.0, 588.0),
    # 10. Herod's Gate (`554..582, 686..722`), Lions' Gate (`858..892, 456..478`), Zion Gate (`174..206, -426..-391`), New Gate (`-154..-126, 284..314`), Dung Gate (`598..630, -222..-184`)
    (554.0, 686.0, 582.0, 722.0),
    (858.0, 456.0, 892.0, 478.0),
    (174.0, -426.0, 206.0, -391.0),
    (-154.0, 284.0, -126.0, 314.0),
    (598.0, -222.0, 630.0, -184.0),
    # 11. Crusader Church of St. Anne & Bethesda Basin (`780..848, 521..544`)
    (780.0, 521.0, 848.0, 544.0),
    # 12. Austrian Hospice custom main villa (`405..433, 398..418`) & Station III/IV Patriarchate Courtyard (`404..414, 376..389`)
    (405.0, 398.0, 433.0, 418.0),
    (404.0, 376.0, 414.0, 389.0),
    # 13. Stork Tower (Burj al-Laqlaq) NE Bastion (`802..826, 754..778`) & Hezekiah's Pool water basin (`120..144, 44..92`)
    (802.0, 754.0, 826.0, 778.0),
    (120.0, 44.0, 144.0, 92.0),
]

# Custom-built vaulted souq monuments where we do not duplicate generic tunnel_passages
CUSTOM_VAULT_EXCLUSION_BOXES: List[Polygon] = [
    box(-28.0, -16.0, 18.0, 16.0),     # Jaffa Gate vaulted L-passage
    box(76.0, -11.0, 144.5, 15.0),     # David St hero souq arches
    box(300.5, -40.0, 322.5, 123.0),   # Crusader Triple Souq
    box(298.0, -194.0, 319.0, -54.0),  # Southern Roman Cardo
    box(520.0, 137.0, 616.0, 157.0),   # Suq al-Qattanin
    box(638.0, 44.0, 670.0, 68.0),     # Wilson's Arch / Tankiziyya
]


def poly_bbox(ring: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in ring]
    ys = [float(p[1]) for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def enrich_building_geometry(b_entry: dict) -> dict:
    """
    Computes and attaches:
      - `centroid`: guaranteed interior pole-of-inaccessibility (`polylabel` / `representative_point`)
      - `min_ground_z`: minimum `street_elevation_z` across all vertices & centroid
      - `base_z`: `street_elevation_z` at interior centroid
    """
    ring = b_entry.get("ring", [])
    pts = [(float(p[0]), float(p[1])) for p in ring]
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if isinstance(poly, Polygon) and not poly.is_empty:
            try:
                in_pt = polylabel(poly, tolerance=0.25)
            except Exception:
                in_pt = poly.representative_point()
            cx, cy = round(float(in_pt.x), 2), round(float(in_pt.y), 2)
        else:
            cx, cy = round(float(b_entry["centroid"][0]), 2), round(float(b_entry["centroid"][1]), 2)
    except Exception:
        cx, cy = round(float(b_entry["centroid"][0]), 2), round(float(b_entry["centroid"][1]), 2)

    b_entry["centroid"] = [cx, cy]
    bz = street_elevation_z(cx, cy)
    b_entry["base_z"] = bz
    all_zs = [bz] + [street_elevation_z(p[0], p[1]) for p in pts]
    b_entry["min_ground_z"] = round(min(all_zs), 3)
    return b_entry


def _obb_metrics(poly: Polygon) -> Tuple[float, float, Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Returns `(long_len, short_len, (ux, uy), (nx, ny), (cx, cy))` of `poly.minimum_rotated_rectangle`.
    """
    mrr = poly.minimum_rotated_rectangle
    if not isinstance(mrr, Polygon) or len(mrr.exterior.coords) < 4:
        b0, b1, b2, b3 = poly.bounds
        return max(b2 - b0, b3 - b1), min(b2 - b0, b3 - b1), (1.0, 0.0), (0.0, 1.0), (0.5 * (b0 + b2), 0.5 * (b1 + b3))
    coords = list(mrr.exterior.coords)
    e0 = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    e1 = (coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    l0 = math.hypot(*e0)
    l1 = math.hypot(*e1)
    if l0 >= l1:
        long_l, short_l = l0, l1
        ux, uy = (e0[0] / l0, e0[1] / l0) if l0 > 1e-6 else (1.0, 0.0)
    else:
        long_l, short_l = l1, l0
        ux, uy = (e1[0] / l1, e1[1] / l1) if l1 > 1e-6 else (1.0, 0.0)
    nx, ny = -uy, ux
    c = mrr.centroid
    return long_l, short_l, (ux, uy), (nx, ny), (float(c.x), float(c.y))


def subdivide_insula_obb(
    poly: Polygon,
    max_area: float = 225.0,
    max_len: float = 20.5,
    depth: int = 0,
) -> List[Polygon]:
    """
    Recursively bisects a street-bounded urban block (`Insula`) perpendicular to its local
    Oriented Bounding Box (`minimum_rotated_rectangle`) longest axis until every parcel is an
    authentic Old City building size (`18..225 m²`, `long_l <= 20.5m`).
    """
    if poly.is_empty or poly.area < 16.0:
        return []
    if depth >= 11 or (poly.area <= max_area and _obb_metrics(poly)[0] <= max_len):
        return [poly]

    long_l, short_l, (ux, uy), (nx, ny), (cx, cy) = _obb_metrics(poly)
    if long_l <= max_len and poly.area <= max_area:
        return [poly]

    # Deterministic slight offset from midpoint so adjacent parcels have organic Ottoman/Mamluk widths
    h_seed = int(abs(cx * 37.0 + cy * 73.0) + depth * 19) % 7
    shift_frac = (-0.08, 0.06, -0.04, 0.0, 0.05, -0.06, 0.07)[h_seed]
    cut_dist = long_l * shift_frac
    px = cx + ux * cut_dist
    py = cy + uy * cut_dist
    cut_line = LineString([(px - nx * 600.0, py - ny * 600.0), (px + nx * 600.0, py + ny * 600.0)])

    try:
        parts = split(poly, cut_line)
        sub_polys: List[Polygon] = []
        for g in parts.geoms:
            if isinstance(g, Polygon) and g.area >= 16.0:
                sub_polys.append(g)
            elif isinstance(g, MultiPolygon):
                sub_polys.extend(p for p in g.geoms if p.area >= 16.0)
        # If any resulting piece is too narrow (< 3.3m short side) or split didn't divide, try orthogonal split or stop
        if len(sub_polys) <= 1 or any(_obb_metrics(sp)[1] < 3.3 for sp in sub_polys):
            if short_l > 16.5:
                cut_line_orth = LineString([(cx - ux * 600.0, cy - uy * 600.0), (cx + ux * 600.0, cy + uy * 600.0)])
                parts_orth = split(poly, cut_line_orth)
                sub_orth = [
                    g for g in parts_orth.geoms
                    if isinstance(g, Polygon) and g.area >= 18.0 and _obb_metrics(g)[1] >= 3.3
                ]
                if len(sub_orth) > 1:
                    out_orth: List[Polygon] = []
                    for sp in sub_orth:
                        out_orth.extend(subdivide_insula_obb(sp, max_area, max_len, depth + 1))
                    return out_orth
            return [poly]

        out: List[Polygon] = []
        for sp in sub_polys:
            out.extend(subdivide_insula_obb(sp, max_area, max_len, depth + 1))
        return out
    except Exception:
        return [poly]


def extract_tunnel_passages(
    osm_streets: Sequence[dict],
    city_wall_poly: Polygon,
    tm_inner_reserve: Polygon,
    reserve_only_union: Polygon,
    insulae_geom: Polygon | MultiPolygon,
) -> List[dict]:
    """
    Extracts:
      1. All intramural OpenStreetMap `tunnel=yes`, `tunnel=building_passage`, and `covered=yes`
         street ways (`83` ways in raw OSM), broken into straight `4.5m..11.5m` vaulted bridge-building bays.
      2. Periodic `7.5m..9.5m` over-street bridge-building spans (`Qantara`) across narrow intramural
         alleys in all 4 Quarters so streets frequently pass through buildings as in the real Old City.
      3. Clamps every tunnel bay to the exact interval where BOTH left and right side piers
         (`±(half_w + 0.55m)`) lie inside `insulae_geom` so zero tunnel piers ever protrude into
         a cross-street intersection!
    """
    custom_excl = unary_union(CUSTOM_VAULT_EXCLUSION_BOXES)
    passages: List[dict] = []
    placed_midpoints: List[Tuple[float, float]] = []

    def _clamp_segment_to_insula_flanks(
        ax0: float,
        ay0: float,
        ax1: float,
        ay1: float,
        half_w: float,
    ) -> Tuple[float, float, float, float] | None:
        seg_l = math.hypot(ax1 - ax0, ay1 - ay0)
        if seg_l < 2.2:
            return None
        ux, uy = (ax1 - ax0) / seg_l, (ay1 - ay0) / seg_l
        rx, ry = -uy, ux
        d_flank = half_w + 0.55
        n_samples = max(6, int(math.ceil(seg_l / 0.5)))
        best_run: List[float] = []
        curr_run: List[float] = []
        for k in range(n_samples + 1):
            t = k / n_samples
            px = ax0 + (ax1 - ax0) * t
            py = ay0 + (ay1 - ay0) * t
            pt_l = Point(px + rx * d_flank, py + ry * d_flank)
            pt_r = Point(px - rx * d_flank, py - ry * d_flank)
            if insulae_geom.contains(pt_l) and insulae_geom.contains(pt_r):
                curr_run.append(t)
                if len(curr_run) > len(best_run):
                    best_run = list(curr_run)
            else:
                curr_run = []
        if len(best_run) < 2:
            return None
        t_min, t_max = best_run[0], best_run[-1]
        if (t_max - t_min) * seg_l < 2.4:
            return None
        return (
            ax0 + (ax1 - ax0) * t_min,
            ay0 + (ay1 - ay0) * t_min,
            ax0 + (ax1 - ax0) * t_max,
            ay0 + (ay1 - ay0) * t_max,
        )

    def _add_polyline_as_tunnel_bays(
        way_id: str | int,
        name: str,
        pts: Sequence[Sequence[float]],
        half_w: float,
        max_sub_len: float = 9.5,
    ) -> None:
        # Clean consecutive points
        coords: List[Tuple[float, float]] = []
        for p in pts:
            pt = (float(p[0]), float(p[1]))
            if not coords or math.hypot(pt[0] - coords[-1][0], pt[1] - coords[-1][1]) >= 1.2:
                coords.append(pt)
        if len(coords) < 2:
            return

        # Expand long segments into contiguous bays of <= max_sub_len
        bays: List[Tuple[float, float, float, float]] = []
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len < 2.4:
                continue
            n_sub = max(1, int(round(seg_len / max_sub_len)))
            for k in range(n_sub):
                t0 = k / n_sub
                t1 = (k + 1) / n_sub
                bx0 = x0 + (x1 - x0) * t0
                by0 = y0 + (y1 - y0) * t0
                bx1 = x0 + (x1 - x0) * t1
                by1 = y0 + (y1 - y0) * t1
                clamped = _clamp_segment_to_insula_flanks(bx0, by0, bx1, by1, half_w)
                if clamped is None:
                    continue
                bx0, by0, bx1, by1 = clamped
                mx, my = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
                mpt = Point(mx, my)
                if not city_wall_poly.buffer(-4.0).contains(mpt):
                    continue
                if tm_inner_reserve.contains(mpt) or custom_excl.contains(mpt) or reserve_only_union.contains(mpt):
                    continue
                bays.append((bx0, by0, bx1, by1))

        if not bays:
            return

        for idx, (bx0, by0, bx1, by1) in enumerate(bays):
            mx, my = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
            # Determine if contiguous with previous/next bay
            is_start = (idx == 0) or (math.hypot(bx0 - bays[idx - 1][2], by0 - bays[idx - 1][3]) > 0.5)
            is_end = (idx == len(bays) - 1) or (math.hypot(bx1 - bays[idx + 1][0], by1 - bays[idx + 1][1]) > 0.5)
            seed_h = int(abs(mx * 19.0 + my * 31.0) + idx * 7)
            reg = classify_region(mx, my)
            passages.append(
                {
                    "id": f"tunnel_{way_id}_{idx}",
                    "region": reg,
                    "name": name,
                    "x0": round(bx0, 2),
                    "y0": round(by0, 2),
                    "x1": round(bx1, 2),
                    "y1": round(by1, 2),
                    "half_width": round(half_w, 2),
                    "levels": 2 if (seed_h % 3 != 0) else 3,
                    "is_start": is_start,
                    "is_end": is_end,
                    "has_dome": (seed_h % 3 == 0),
                    "has_mashrabiya": (seed_h % 2 == 0),
                }
            )
            placed_midpoints.append((mx, my))

    # Pass A: All explicit OSM `tunnel=yes` / `tunnel=building_passage` / `covered=yes` streets
    for s in osm_streets:
        tags = s.get("tags", {})
        is_tunnel = (
            tags.get("tunnel") in ("yes", "building_passage", "avalanche_protector")
            or tags.get("covered") in ("yes", "arcade", "colonade")
        )
        if not is_tunnel:
            continue
        pts = s.get("pts", [])
        if len(pts) < 2:
            continue
        hw_tag = s.get("highway", "footway")
        name = (s.get("name") or "").strip()
        is_named_main = bool(name) or hw_tag in ("primary", "secondary", "tertiary", "residential", "pedestrian", "living_street")
        half_w = 1.95 if is_named_main else 1.45
        _add_polyline_as_tunnel_bays(s.get("osm_id", 0), name, pts, half_w, max_sub_len=9.0)

    # Pass B: Periodic `7.5m..9.0m` Over-Street Bridge-Buildings (`Qantara`) across narrow intramural
    #         streets/alleys so every Quarter (including Christian & Armenian Quarters) has streets
    #         passing under bridge-buildings every ~42m
    for s_idx, s in enumerate(osm_streets):
        tags = s.get("tags", {})
        if tags.get("tunnel") or tags.get("covered"):
            continue
        pts = s.get("pts", [])
        if len(pts) < 2:
            continue
        name = (s.get("name") or "").strip()
        if name == "Western Wall Plaza":
            continue
        hw_tag = s.get("highway", "footway")
        is_named_main = bool(name) or hw_tag in ("primary", "secondary", "tertiary", "residential", "pedestrian", "living_street")
        half_w = 1.95 if is_named_main else 1.45

        for i in range(len(pts) - 1):
            x0, y0 = float(pts[i][0]), float(pts[i][1])
            x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len < 13.5:
                continue
            mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            mpt = Point(mx, my)
            if not city_wall_poly.buffer(-14.0).contains(mpt):
                continue
            if tm_inner_reserve.buffer(8.0).contains(mpt) or custom_excl.contains(mpt) or reserve_only_union.buffer(4.0).contains(mpt):
                continue
            # Ensure >= 38m spacing between synthesized bridge-buildings and existing tunnels
            if any(math.hypot(mx - px, my - py) < 38.0 for px, py in placed_midpoints):
                continue

            span_len = min(8.2, seg_len * 0.48)
            ux, uy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
            bx0 = mx - ux * (0.5 * span_len)
            by0 = my - uy * (0.5 * span_len)
            bx1 = mx + ux * (0.5 * span_len)
            by1 = my + uy * (0.5 * span_len)
            clamped = _clamp_segment_to_insula_flanks(bx0, by0, bx1, by1, half_w)
            if clamped is None:
                continue
            bx0, by0, bx1, by1 = clamped
            seed_h = int(abs(mx * 23.0 + my * 41.0) + s_idx)
            reg = classify_region(mx, my)
            passages.append(
                {
                    "id": f"qantara_{s.get('osm_id', s_idx)}_{i}",
                    "region": reg,
                    "name": name or "Covered Alley Passage",
                    "x0": round(bx0, 2),
                    "y0": round(by0, 2),
                    "x1": round(bx1, 2),
                    "y1": round(by1, 2),
                    "half_width": round(half_w, 2),
                    "levels": 2 if (seed_h % 3 != 0) else 3,
                    "is_start": True,
                    "is_end": True,
                    "has_dome": (seed_h % 2 == 0),
                    "has_mashrabiya": (seed_h % 3 != 0),
                }
            )
            placed_midpoints.append((mx, my))

    return passages


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    raw_osm_path = root / "data" / "osm_old_city_raw.json"
    osm_path = root / "data" / "osm_old_city.json"
    src_path = raw_osm_path if raw_osm_path.exists() else osm_path
    with open(src_path, "r", encoding="utf-8") as f:
        osm_data = json.load(f)

    # 1. Load authentic city plan (exact Ottoman City Wall, Temple Mount Wall, Western Wall Plaza, and OSM streets)
    plan = build_city_plan()
    city_wall_poly = Polygon(plan["cityWall"])
    tm_wall_poly = Polygon(plan["templeMountWall"])
    # Reserve the interior of the Temple Mount (shrunk by 4.5m so the Mamluk Madrasas/Riwaqs along the outer N/W perimeter are preserved!)
    tm_inner_reserve = tm_wall_poly.buffer(-4.5)
    # Exact Western Wall Plaza polygon
    ww_plaza_poly = Polygon(plan["landmarks"][7]["poly"]).buffer(1.2)

    reserve_boxes = [box(rx0, ry0, rx1, ry1) for rx0, ry0, rx1, ry1 in EXACT_RESERVE_RECTS]
    reserve_only_union = unary_union(reserve_boxes + [tm_inner_reserve, ww_plaza_poly])

    carve_geoms = list(reserve_boxes) + [
        tm_inner_reserve,
        ww_plaza_poly,
        city_wall_poly.boundary.buffer(2.5),
    ]

    street_segments: List[Tuple[float, float, float, float, float]] = []

    # Carve ONLY real OpenStreetMap streets (outside the Temple Mount esplanade) with tight medieval alley widths
    for s in osm_data.get("streets", []):
        if s.get("name") == "Western Wall Plaza":
            continue
        pts = s.get("pts", [])
        if len(pts) < 2:
            continue
        coords_2d = [(float(p[0]), float(p[1])) for p in pts]
        ls = LineString(coords_2d)
        if ls.length < 1.5:
            continue
        if tm_inner_reserve.contains(ls.representative_point()):
            continue

        hw_tag = s.get("highway", "footway")
        name = (s.get("name") or "").strip()
        is_named_main = bool(name) or hw_tag in ("primary", "secondary", "tertiary", "residential", "pedestrian", "living_street")
        carve_half_w = 1.98 if is_named_main else 1.48
        carve_geoms.append(ls.buffer(carve_half_w, cap_style=1, join_style=1))

        for i in range(len(pts) - 1):
            x0, y0 = float(pts[i][0]), float(pts[i][1])
            x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
            if math.hypot(x1 - x0, y1 - y0) >= 1.0:
                street_segments.append((x0, y0, x1, y1, carve_half_w))

    # Also ensure the curated primary/secondary OSM street polylines are clear
    for cs in plan["curatedStreets"]:
        pts = cs["pts"]
        coords_2d = [(float(p[0]), float(p[1])) for p in pts]
        if len(coords_2d) >= 2:
            ls = LineString(coords_2d)
            if ls.length >= 1.5:
                carve_w = 2.25 if cs["tier"] == "primary" else 2.00
                carve_geoms.append(ls.buffer(carve_w, cap_style=1, join_style=1))

    # Connect r01 bespoke corridors seamlessly into the r02/r03 OSM street network:
    # 1. North exit of r01 Latin Patriarchate Rd (`x=24.0, y=58.0`) -> Saint Dimitri's Rd (`29.2, 74.3`), Latin Patriarchate Rd (`6.6, 54.3`), & Greek Catholic Patriarchate St (`46.6, 53.4`)
    carve_geoms.append(LineString([(24.0, 52.0), (24.0, 62.0), (29.2, 74.3), (26.3, 84.6)]).buffer(2.55, cap_style=1, join_style=1))
    carve_geoms.append(LineString([(-0.7, 46.1), (6.6, 54.3), (24.0, 61.5), (46.6, 53.4)]).buffer(2.35, cap_style=1, join_style=1))
    # 2. East exit of r01 David St (`x=158.0, y=2.0`) -> Christian Quarter St (`x=162.2, y=15.3`), Saint Mark St (`x=164.3, y=6.9`), & r03 David St (`x=168.0, y=12.6`)
    carve_geoms.append(LineString([(154.0, 2.0), (162.5, 2.0), (162.5, 13.0), (171.0, 13.0)]).buffer(2.85, cap_style=1, join_style=1))
    carve_geoms.append(LineString([(165.5, -16.0), (164.3, 6.9), (162.2, 15.3), (161.0, 32.0)]).buffer(2.45, cap_style=1, join_style=1))
    # 3. Muristan North Market Gateway (`(235.0, 154.0)`) & Suq Aftimos (`(232.0, 128.0) -> (241.5, 149.1) -> (242.2, 161.5)`)
    carve_geoms.append(LineString([(232.0, 128.0), (235.0, 144.0), (235.0, 157.5), (242.2, 161.5)]).buffer(3.15, cap_style=1, join_style=1))
    carve_geoms.append(LineString([(232.0, 128.0), (241.5, 149.1), (242.2, 161.5)]).buffer(2.45, cap_style=1, join_style=1))

    # Ensure the 29 official POI viewpoints in places.json are not inside a building
    places_path = root / "public" / "data" / "places.json"
    if places_path.exists():
        pdata = json.loads(places_path.read_text(encoding="utf-8"))
        plist = pdata.get("places", pdata) if isinstance(pdata, dict) else pdata
        for p in plist:
            if "x" in p and "y" in p:
                px_p, py_p = float(p["x"]), float(p["y"])
                carve_geoms.append(box(px_p - 1.45, py_p - 1.45, px_p + 1.45, py_p + 1.45))

    carve_out_union = unary_union(carve_geoms)

    # 2. Index raw OpenStreetMap buildings by STRtree so we can transfer real names/levels/tags to our Insula parcels
    #    and preserve extramural buildings outside the Old City Walls
    raw_buildings = list(osm_data.get("buildings", []))
    raw_intramural_polys: List[Polygon] = []
    raw_intramural_meta: List[dict] = []
    clean_buildings: List[dict] = []

    for b in raw_buildings:
        ring = b.get("ring", [])
        if len(ring) < 3:
            continue
        x0, y0, x1, y1 = poly_bbox(ring)
        sx, sy = x1 - x0, y1 - y0
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])

        if not city_wall_poly.contains(Point(cx, cy)):
            # Only preserve extramural buildings in the immediate Jaffa Gate / Mamilla western approach (`r01_jaffa_gate`)
            if classify_region(cx, cy) == "r01_jaffa_gate" and sx <= 85.0 and sy <= 85.0 and "building" in b.get("tags", {}):
                clean_buildings.append(enrich_building_geometry(dict(b)))
            continue

        if sx > 140.0 or sy > 140.0:
            continue
        try:
            p = Polygon([(float(pt[0]), float(pt[1])) for pt in ring])
            if not p.is_valid:
                p = p.buffer(0)
            if isinstance(p, Polygon) and p.area >= 12.0:
                raw_intramural_polys.append(p)
                raw_intramural_meta.append(b)
        except Exception:
            pass

    raw_tree = STRtree(raw_intramural_polys) if raw_intramural_polys else None

    # 3. Compute Solid Street-Bounded Urban Blocks (`Insulae`) & Subdivide via OBB Recursive Bisection
    insulae_geom = city_wall_poly.difference(carve_out_union)
    raw_insula_list = (
        list(insulae_geom.geoms)
        if isinstance(insulae_geom, MultiPolygon)
        else ([insulae_geom] if isinstance(insulae_geom, Polygon) else [])
    )

    parcel_polys: List[Polygon] = []
    for ins in raw_insula_list:
        if ins.area < 16.0:
            continue
        try:
            ins_clean = ins.buffer(-0.65, join_style=2).buffer(0.65, join_style=2).difference(carve_out_union)
            if ins_clean.is_empty:
                continue
        except Exception:
            ins_clean = ins

        parts = (
            list(ins_clean.geoms)
            if isinstance(ins_clean, MultiPolygon)
            else ([ins_clean] if isinstance(ins_clean, Polygon) else [])
        )
        for pt_poly in parts:
            if pt_poly.area < 16.0:
                continue
            # Fill tiny < 25m² courtyard pinholes inside solid blocks; preserve genuine courtyards >= 25m²
            holes = [
                inter for inter in pt_poly.interiors
                if Polygon(inter).area >= 25.0
            ]
            pt_solid = Polygon(pt_poly.exterior, holes)
            parcel_polys.extend(subdivide_insula_obb(pt_solid, max_area=225.0, max_len=20.5))

    next_parcel_id = 9100000
    intramural_parcel_count = 0

    for sp in parcel_polys:
        # Inset by 3 cm (`0.06m` mortar joint between adjacent parcels) to eliminate coplanar Z-fighting
        # while keeping shared party walls within `_is_wall_edge_party_blocked` tolerance
        try:
            sp_inset = sp.buffer(-0.03, join_style=2)
            if isinstance(sp_inset, Polygon) and not sp_inset.is_empty and sp_inset.area >= 14.0:
                sp = sp_inset
        except Exception:
            pass

        sp_clean = sp.simplify(0.14, preserve_topology=True)
        if not isinstance(sp_clean, Polygon) or sp_clean.is_empty or sp_clean.area < 14.0:
            continue
        if sp_clean.buffer(-1.15, join_style=2).is_empty:
            continue

        try:
            rep_pt = polylabel(sp_clean, tolerance=0.25)
        except Exception:
            rep_pt = sp_clean.representative_point()
        scx, scy = float(rep_pt.x), float(rep_pt.y)
        if not city_wall_poly.contains(Point(scx, scy)) or reserve_only_union.contains(Point(scx, scy)):
            continue

        ext_coords = [[round(float(x), 2), round(float(y), 2)] for x, y in sp_clean.exterior.coords]
        if len(ext_coords) < 4:
            continue

        # Match against overlapping raw OSM building to inherit authentic levels/tags/name
        matched_meta = None
        if raw_tree is not None:
            best_ov = 0.0
            for idx in raw_tree.query(sp_clean):
                rp = raw_intramural_polys[int(idx)]
                if rp.intersects(sp_clean):
                    ov = rp.intersection(sp_clean).area
                    if ov > best_ov:
                        best_ov = ov
                        matched_meta = raw_intramural_meta[int(idx)]

        seed_h = int(abs(scx * 29.0 + scy * 53.0))
        default_lvls = 3 if (seed_h % 4 == 0) else 2
        if matched_meta is not None:
            lvls = max(2, min(4, int(matched_meta.get("levels", default_lvls))))
            tags = dict(matched_meta.get("tags", {"building": "yes"}))
            name = matched_meta.get("name")
        else:
            lvls = default_lvls
            tags = {"building": "yes"}
            name = None

        b_entry = {
            "osm_id": next_parcel_id,
            "name": name,
            "region": classify_region(scx, scy),
            "centroid": [round(scx, 2), round(scy, 2)],
            "levels": lvls,
            "height": round(lvls * 3.5, 1),
            "tags": tags,
            "ring": ext_coords,
        }
        next_parcel_id += 1
        intramural_parcel_count += 1
        clean_buildings.append(enrich_building_geometry(b_entry))

    # 4. Extract Over-Street Bridge Buildings & Vaulted Street Tunnels (`tunnel_passages`)
    tunnel_passages = extract_tunnel_passages(
        osm_data.get("streets", []),
        city_wall_poly,
        tm_inner_reserve,
        reserve_only_union,
        insulae_geom,
    )

    quarters_area = city_wall_poly.difference(tm_wall_poly).area
    insulae_area = insulae_geom.difference(tm_wall_poly).area
    print(
        f"[1/3] Solid Street-Bounded Insulae Area: {insulae_area:,.0f} / {quarters_area:,.0f} m² "
        f"({100.0 * insulae_area / quarters_area:.1f}% of the 4 Quarters)."
    )
    print(
        f"[2/3] Generated {intramural_parcel_count:,} contiguous street-aligned building parcels "
        f"({len(clean_buildings):,} total with extramural) + {len(tunnel_passages)} over-street bridge/tunnel bays."
    )

    # Update region counts & save data/osm_old_city.json and public/data/osm_old_city.json
    reg_counts: Dict[str, int] = {}
    for b in clean_buildings:
        r = b.get("region", "unknown")
        reg_counts[r] = reg_counts.get(r, 0) + 1

    osm_data["buildings"] = clean_buildings
    osm_data["tunnel_passages"] = tunnel_passages
    osm_data["metadata"]["counts"]["buildings"] = len(clean_buildings)
    osm_data["metadata"]["counts"]["buildings_by_region"] = reg_counts
    osm_data["metadata"]["counts"]["tunnel_passages"] = len(tunnel_passages)

    with open(osm_path, "w", encoding="utf-8") as f:
        json.dump(osm_data, f, indent=2, ensure_ascii=False)
    public_osm_path = root / "public" / "data" / "osm_old_city.json"
    if public_osm_path.exists():
        public_osm_path.write_text(json.dumps(osm_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5. Update `data/osm_jaffa.json` so `r01_jaffa_gate` uses the exact same dense intramural parcels
    #    inside `city_wall_poly` plus the extramural Mamilla buildings outside `city_wall_poly`
    jaffa_raw_path = root / "data" / "osm_jaffa_raw.json"
    jaffa_path = root / "data" / "osm_jaffa.json"
    if jaffa_path.exists():
        if not jaffa_raw_path.exists():
            jaffa_raw_path.write_text(jaffa_path.read_text(encoding="utf-8"), encoding="utf-8")
        j_data = json.loads(jaffa_raw_path.read_text(encoding="utf-8"))
        j_clean: List[dict] = []
        # Keep extramural buildings from `osm_jaffa_raw.json`
        for jb in j_data.get("buildings", []):
            jring = jb.get("ring", [])
            if len(jring) < 3 or "building" not in jb.get("tags", {}):
                j_clean.append(jb)
                continue
            jcx, jcy = float(jb["centroid"][0]), float(jb["centroid"][1])
            if not city_wall_poly.contains(Point(jcx, jcy)):
                j_clean.append(jb)
        # Add all intramural parcels belonging to `r01_jaffa_gate`
        for b in clean_buildings:
            if b.get("region") == "r01_jaffa_gate":
                j_clean.append(b)
        j_data["buildings"] = j_clean
        j_data["tunnel_passages"] = [tp for tp in tunnel_passages if tp.get("region") == "r01_jaffa_gate"]
        jaffa_path.write_text(json.dumps(j_data, indent=2, ensure_ascii=False), encoding="utf-8")
        public_jaffa_path = root / "public" / "data" / "osm_jaffa.json"
        if public_jaffa_path.exists():
            public_jaffa_path.write_text(json.dumps(j_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6. Render diagnostic plan `build/qa/osm_complete_city_outline.png`
    W, H = 1600, 1600
    img = Image.new("RGB", (W, H), (16, 20, 28))
    draw = ImageDraw.Draw(img)

    def to_px(x: float, y: float) -> Tuple[int, int]:
        px = int((x - (-220.0)) / 1200.0 * (W - 80) + 40)
        py = int((800.0 - y) / 1260.0 * (H - 80) + 40)
        return (px, py)

    wall_px = [to_px(p[0], p[1]) for p in plan["cityWall"]] + [to_px(*plan["cityWall"][0])]
    draw.polygon(wall_px, fill=(24, 30, 42), outline=(240, 190, 90))
    tm_px = [to_px(p[0], p[1]) for p in plan["templeMountWall"]] + [to_px(*plan["templeMountWall"][0])]
    draw.polygon(tm_px, fill=(30, 38, 34), outline=(255, 215, 90))

    for b in clean_buildings:
        cx, cy = b["centroid"]
        if not city_wall_poly.contains(Point(cx, cy)):
            continue
        pts = [to_px(p[0], p[1]) for p in b["ring"]]
        draw.polygon(pts, fill=(172, 152, 122), outline=(92, 76, 58))

    for sx0, sy0, sx1, sy1, _ in street_segments:
        draw.line([to_px(sx0, sy0), to_px(sx1, sy1)], fill=(65, 210, 255), width=2)

    # Highlight Over-Street Bridge Buildings (`tunnel_passages`) in warm gold/orange spanning across the blue streets
    for tp in tunnel_passages:
        draw.line([to_px(tp["x0"], tp["y0"]), to_px(tp["x1"], tp["y1"])], fill=(255, 165, 40), width=5)

    out_img = root / "build" / "qa" / "osm_complete_city_outline.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_img)
    print(f"[3/3] Saved diagnostic city outline map to {out_img}")
    print(json.dumps(reg_counts, indent=2))


if __name__ == "__main__":
    main()
