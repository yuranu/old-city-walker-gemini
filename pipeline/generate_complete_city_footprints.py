#!/usr/bin/env python3
"""
pipeline/generate_complete_city_footprints.py

Authentic OpenStreetMap Building & Street Footprint Processor for the Old City of Jerusalem:
1. Starts strictly from the uncorrupted ground-truth `data/osm_old_city_raw.json` (2,232+ real
   intramural OpenStreetMap building polygons across all 4 Quarters).
2. Excludes non-building boundary relations (`5862586`, `7880713`, archaeological dig bounds,
   and `building=wall` retaining wall ways) and interior Haram al-Sharif monuments that have
   dedicated high-detail architectural builders in `r07_temple_mount`.
3. Carves ONLY along real OpenStreetMap streets/alleys (`raw["streets"]`) with tight, authentic
   medieval alley clearances (`1.25m`–`1.65m` half-width) and tight custom landmark footprints.
4. Eliminates ALL synthetic grid-box generators (`osm_street_infill`, `osm_block_infill`,
   `subdivide_large_building_box`, and axis-aligned `split_deeply_concave_polygon`) that
   previously corrupted the Old City map with random boxes and pixel-grid artifacts.
5. Computes guaranteed interior points (`polylabel` / `representative_point`) and `min_ground_z`
   so sloped-ground walls are grounded into the terrain and roofs/domes never float over streets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import polylabel, unary_union

from build_city_plan import build_city_plan, point_in_polygon_2d
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

# Tight, exact custom-built monument footprints & open plazas aligned to true OSM coordinates
EXACT_RESERVE_RECTS: List[Tuple[float, float, float, float]] = [
    # 1. Jaffa Gate, West Plaza, Omar Square, Citadel & Hero Corridors in `r01`
    (-55.0, -24.0, 82.0, 26.0),
    (6.0, -92.0, 66.0, -14.0),
    (10.0, 22.0, 44.0, 68.0),
    (76.0, -11.0, 164.0, 15.0),
    # 2. Church of the Holy Sepulchre (`170..219, 159..213`) & Mosque of Omar (`180..199, 136..151`)
    (170.0, 159.0, 219.0, 213.0),
    (180.0, 136.0, 199.0, 151.0),
    # 3. Muristan Fountain (`224..240, 120..136`) & Lutheran Church of the Redeemer (`264..295, 132..151.5`)
    (224.0, 120.0, 240.0, 136.0),
    (264.0, 132.0, 295.0, 151.5),
    # 4. St. James Cathedral (`117.5..147.5, -244.0..-215.5`)
    (117.5, -244.0, 147.5, -215.5),
    # 5. Hurva Synagogue & Hurva Square (`333.5..360.5, -152.5..-103.5`), Golden Menorah (`375.5..381.5, -116.5..-111.0`), & Sidna Omar Minaret (`363.5..369.5, -147.5..-140.5`)
    (333.5, -152.5, 360.5, -103.5),
    (375.5, -116.5, 381.5, -111.0),
    (363.5, -147.5, 369.5, -140.5),
    # 6. Four Sephardic Synagogues (`348..379, -249..-227`), Tiferet Yisrael (`430.0..448.5, -146.5..-125.0`), Batei Mahaseh (`426..451, -333..-308`), Broad Wall (`329.5..342.0, -90.0..-68.0`)
    (348.0, -249.0, 379.0, -227.0),
    (430.0, -146.5, 448.5, -125.0),
    (426.0, -333.0, 451.0, -308.0),
    (329.5, -90.0, 342.0, -68.0),
    # 7. Southern Roman Cardo Colonnade (`299.5..317.0, -190.0..-56.0`) & Crusader Vaulted Triple Souq (`304.0..331.5, -36.0..121.0`)
    (299.5, -190.0, 317.0, -56.0),
    (304.0, -36.0, 331.5, 121.0),
    # 8. Mamluk Turba West (`436.5..453.5, 29.0..41.0`), Mamluk Turba East (`516.5..533.5, 41.0..53.0`), Suq al-Qattanin (`522.0..614.0, 139.0..155.0`), & Western Wall / Beit Strauss (`588.0..668.0, -36.0..34.0`)
    (436.5, 29.0, 453.5, 41.0),
    (516.5, 41.0, 533.5, 53.0),
    (522.0, 139.0, 614.0, 155.0),
    (588.0, -36.0, 643.0, 34.0),
    (634.0, -36.0, 668.0, 47.5),
    (641.0, 47.5, 668.0, 65.5),
    # 9. Damascus Gate Amphitheater & Inner Fork (`233..282, 518..592`)
    (233.0, 518.0, 282.0, 592.0),
    # 10. Herod's Gate (`554..582, 686..726`), Lions' Gate (`858..892, 456..478`), Zion Gate (`174..206, -426..-391`), New Gate (`-154..-126, 284..314`), Dung Gate (`598..630, -222..-184`)
    (554.0, 686.0, 582.0, 726.0),
    (858.0, 456.0, 892.0, 478.0),
    (174.0, -426.0, 206.0, -391.0),
    (-154.0, 284.0, -126.0, 314.0),
    (598.0, -222.0, 630.0, -184.0),
    # 11. Crusader Church of St. Anne + West Parvis (`806..853, 518..539`) & Pools of Bethesda (`772..801, 525..550`)
    (806.0, 518.0, 853.0, 539.0),
    (772.0, 525.0, 801.0, 550.0),
    # 12. Austrian Hospice custom main villa (`405..433, 398..418`), Station III/IV Patriarchate Courtyard (`404..414, 376..389`), & Ecce Homo Sisters of Zion arch piers (`506..524, 390..407`)
    (405.0, 398.0, 433.0, 418.0),
    (404.0, 376.0, 414.0, 389.0),
    # 13. Stork Tower (Burj al-Laqlaq) NE Bastion (`800..826, 745..778`) & Hezekiah's Pool (`115..152, 36..100`)
    (800.0, 745.0, 826.0, 778.0),
    (115.0, 36.0, 152.0, 100.0),
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
    ww_plaza_poly = Polygon(plan["landmarks"][7]["poly"]).buffer(1.5)

    carve_geoms = [box(rx0, ry0, rx1, ry1) for rx0, ry0, rx1, ry1 in EXACT_RESERVE_RECTS]
    carve_geoms.append(tm_inner_reserve)
    carve_geoms.append(ww_plaza_poly)

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
        # Do not carve open esplanade footpaths inside the Temple Mount
        if tm_inner_reserve.contains(ls.representative_point()):
            continue

        hw_tag = s.get("highway", "footway")
        name = (s.get("name") or "").strip()
        is_named_main = bool(name) or hw_tag in ("primary", "secondary", "tertiary", "residential", "pedestrian", "living_street")
        carve_half_w = 1.95 if is_named_main else 1.45
        carve_geoms.append(ls.buffer(carve_half_w, cap_style=2, join_style=2))

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
                carve_w = 2.45 if cs["tier"] == "primary" else 2.15
                carve_geoms.append(ls.buffer(carve_w, cap_style=2, join_style=2))

    # Ensure the 29 official POI viewpoints in places.json are not inside a building
    places_path = root / "public" / "data" / "places.json"
    if places_path.exists():
        pdata = json.loads(places_path.read_text(encoding="utf-8"))
        plist = pdata.get("places", pdata) if isinstance(pdata, dict) else pdata
        for p in plist:
            if "x" in p and "y" in p:
                px_p, py_p = float(p["x"]), float(p["y"])
                carve_geoms.append(box(px_p - 1.4, py_p - 1.4, px_p + 1.4, py_p + 1.4))

    carve_out_union = unary_union(carve_geoms)
    reserve_only_union = unary_union([box(rx0, ry0, rx1, ry1) for rx0, ry0, rx1, ry1 in EXACT_RESERVE_RECTS] + [tm_inner_reserve, ww_plaza_poly])
    print(f"[1/2] Indexed {len(street_segments)} real OSM street segments and built tight 2D street/monument mask.")

    # 2. Filter & keep genuine OSM buildings (NO synthetic Pass A / Pass B boxes, NO Cartesian grid subdivision!)
    #    Sort intramural buildings by 2D area ascending so specific building parts are placed first and
    #    overlapping parent/compound polygons are cleanly trimmed with zero coplanar Z-fighting!
    raw_buildings = list(osm_data.get("buildings", []))

    def _bld_sort_key(b_item: dict) -> float:
        r = b_item.get("ring", [])
        if len(r) < 3:
            return 0.0
        x0, y0, x1, y1 = poly_bbox(r)
        return (x1 - x0) * (y1 - y0)

    raw_buildings.sort(key=_bld_sort_key)
    clean_buildings: List[dict] = []
    placed_intramural_polys: List[Polygon] = []
    next_synth_id = 9000000

    kept_osm = 0
    carved_osm = 0
    deduped_osm = 0

    for b in raw_buildings:
        oid = int(b.get("osm_id", 0))
        if oid in NON_BUILDING_OSM_IDS:
            continue
        tags = b.get("tags", {})
        if tags.get("building") == "wall":
            continue
        if tags.get("source") in ("osm_street_infill", "osm_block_infill", "osm_subdivided", "osm_carved"):
            continue
        ring = b.get("ring", [])
        if len(ring) < 3:
            continue
        x0, y0, x1, y1 = poly_bbox(ring)
        sx, sy = x1 - x0, y1 - y0
        cx, cy = b["centroid"]

        # Outside Old City Walls: keep normal buildings, skip giant boundary polygons
        if not city_wall_poly.contains(Point(cx, cy)):
            if sx <= 85.0 and sy <= 85.0:
                clean_buildings.append(enrich_building_geometry(dict(b)))
            continue

        # Skip boundary relations > 145m
        if sx > 145.0 or sy > 145.0:
            continue

        try:
            poly = Polygon([(float(p[0]), float(p[1])) for p in ring])
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            continue

        if poly.is_empty or poly.area < 12.0:
            continue

        # Skip buildings whose interior centroid lies inside the Temple Mount esplanade or exact monument reserves
        if reserve_only_union.contains(poly.representative_point()):
            continue

        diff = poly.difference(carve_out_union)
        if diff.is_empty:
            continue

        # Subtract any already-placed overlapping intramural building footprints (with 0.12m clearance)
        # so two buildings NEVER share coplanar exterior walls or Z-fight!
        bx0_p, by0_p, bx1_p, by1_p = diff.bounds
        nearby_placed = [
            p for p in placed_intramural_polys
            if not (p.bounds[2] < bx0_p - 0.5 or p.bounds[0] > bx1_p + 0.5 or p.bounds[3] < by0_p - 0.5 or p.bounds[1] > by1_p + 0.5)
        ]
        if nearby_placed:
            overlap_union = unary_union(nearby_placed).buffer(0.12, cap_style=2, join_style=2)
            if diff.intersects(overlap_union):
                diff_after = diff.difference(overlap_union)
                if abs(diff_after.area - diff.area) > 0.5:
                    deduped_osm += 1
                diff = diff_after
                if diff.is_empty:
                    continue

        sub_polys = (
            list(diff.geoms)
            if isinstance(diff, MultiPolygon)
            else ([diff] if isinstance(diff, Polygon) else [g for g in getattr(diff, "geoms", []) if isinstance(g, Polygon)])
        )
        was_carved = (len(sub_polys) > 1) or (abs(diff.area - poly.area) > 1.5)

        for sp_idx, sp in enumerate(sub_polys):
            # Morphological opening trims off any thin (< 1.6m) wall spikes created by street carving
            try:
                sp_opened = sp.buffer(-0.80, join_style=2).buffer(0.80, join_style=2)
                if not sp_opened.is_empty:
                    sp = sp_opened
            except Exception:
                pass
            sp_parts = (
                list(sp.geoms)
                if isinstance(sp, MultiPolygon)
                else ([sp] if isinstance(sp, Polygon) else [g for g in getattr(sp, "geoms", []) if isinstance(g, Polygon)])
            )
            for part_idx, sp_part in enumerate(sp_parts):
                sp_clean = sp_part.simplify(0.15, preserve_topology=True)
                if not isinstance(sp_clean, Polygon) or sp_clean.is_empty or sp_clean.area < 14.0:
                    continue
                bx0, by0, bx1, by1 = sp_clean.bounds
                bsx, bsy = bx1 - bx0, by1 - by0
                if min(bsx, bsy) < 2.2:
                    continue
                # Reject thin diagonal slivers whose perpendicular core width is < 2.3m
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

                b_entry = dict(b)
                b_entry["osm_id"] = oid if (sp_idx == 0 and part_idx == 0 and not was_carved) else next_synth_id
                if sp_idx > 0 or part_idx > 0 or was_carved:
                    next_synth_id += 1
                    carved_osm += 1
                b_entry["region"] = classify_region(scx, scy)
                b_entry["centroid"] = [round(scx, 2), round(scy, 2)]
                b_entry["ring"] = ext_coords
                b_entry = enrich_building_geometry(b_entry)

                clean_buildings.append(b_entry)
                placed_intramural_polys.append(sp_clean)
                kept_osm += 1

    print(f"[2/2] Preserved {kept_osm} authentic intramural OSM buildings ({carved_osm} trimmed/passaged along real OSM streets; {deduped_osm} overlapping polygons deduplicated; 0 synthetic grid boxes).")

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
    print(f"[OK] Updated {osm_path} with {len(clean_buildings)} total authentic OSM buildings:")
    print(json.dumps(reg_counts, indent=2))

    # Also street-carve `data/osm_jaffa.json` with the exact same tight real-OSM street mask
    jaffa_raw_path = root / "data" / "osm_jaffa_raw.json"
    jaffa_path = root / "data" / "osm_jaffa.json"
    if jaffa_path.exists():
        if not jaffa_raw_path.exists():
            jaffa_raw_path.write_text(jaffa_path.read_text(encoding="utf-8"), encoding="utf-8")
        j_data = json.loads(jaffa_raw_path.read_text(encoding="utf-8"))
        j_clean = []
        j_next_id = 8000000
        for jb in j_data.get("buildings", []):
            jring = jb.get("ring", [])
            if len(jring) < 3 or "building" not in jb.get("tags", {}):
                j_clean.append(jb)
                continue
            jcx, jcy = float(jb["centroid"][0]), float(jb["centroid"][1])
            if not city_wall_poly.contains(Point(jcx, jcy)):
                j_clean.append(jb)
                continue
            try:
                jpoly = Polygon([(float(p[0]), float(p[1])) for p in jring])
                if not jpoly.is_valid:
                    jpoly = jpoly.buffer(0)
                jdiff = jpoly.difference(carve_out_union)
                if jdiff.is_empty:
                    continue
                jgeoms = list(jdiff.geoms) if isinstance(jdiff, MultiPolygon) else ([jdiff] if isinstance(jdiff, Polygon) else [])
                for jg in jgeoms:
                    jsp_c = jg.simplify(0.15, preserve_topology=True)
                    if not isinstance(jsp_c, Polygon) or jsp_c.is_empty or jsp_c.area < 14.0:
                        continue
                    jcoords = [[round(float(x), 2), round(float(y), 2)] for x, y in jsp_c.exterior.coords]
                    if len(jcoords) < 4:
                        continue
                    jb_new = dict(jb)
                    jb_new["osm_id"] = j_next_id
                    j_next_id += 1
                    jb_new["ring"] = jcoords
                    jb_new = enrich_building_geometry(jb_new)
                    j_clean.append(jb_new)
            except Exception:
                j_clean.append(jb)
        j_data["buildings"] = j_clean
        jaffa_path.write_text(json.dumps(j_data, indent=2, ensure_ascii=False), encoding="utf-8")
        public_jaffa_path = root / "public" / "data" / "osm_jaffa.json"
        if public_jaffa_path.exists():
            public_jaffa_path.write_text(json.dumps(j_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Render diagnostic plan build/qa/osm_complete_city_outline.png
    W, H = 1400, 1400
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
        draw.polygon(pts, fill=(155, 138, 112), outline=(215, 195, 160))

    for sx0, sy0, sx1, sy1, _ in street_segments:
        draw.line([to_px(sx0, sy0), to_px(sx1, sy1)], fill=(65, 210, 255), width=2)

    out_img = root / "build" / "qa" / "osm_complete_city_outline.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_img)
    print(f"[OK] Saved diagnostic city outline map to {out_img}")


if __name__ == "__main__":
    main()
