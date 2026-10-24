#!/usr/bin/env python3
"""
Modular Multi-Region 3D Builder for the Full Old City of Jerusalem (`pipeline/regions_builder.py`).

Builds each contiguous region inside the 16th-century Ottoman Old City Walls as its own
self-contained `.glb` chunk (`public/models/regions/<region_id>.glb`), sharing the continuous
elevation model `street_elevation_z(x, y)` from `fetch_osm_and_dem.py` and the external PBR
textures in `public/textures/`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from arch_rules import (
    WOOD_DOOR_UV,
    add_arch_spandrel_wall,
    add_arch_voussoirs,
    add_arched_opening_panel,
    build_building_from_footprint,
    clear_scene_building_rings,
    ensure_ccw_2d,
    register_scene_building_ring,
)


def point_in_polygon_2d(pt: Tuple[float, float], poly: Sequence[Tuple[float, float]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0):
            inside = not inside
    return inside


def _dist_to_polyline(px: float, py: float, pts: Sequence[Tuple[float, float]], radius: float) -> bool:
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        l2 = dx * dx + dy * dy
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / l2)) if l2 > 1e-6 else 0.0
        if math.hypot(px - (x0 + t * dx), py - (y0 + t * dy)) <= radius:
            return True
    return False


def _ring_overlaps_rect(ring: Sequence[Sequence[float]], x0: float, y0: float, x1: float, y1: float) -> bool:
    """Returns True if polygon `ring` overlaps axis-aligned box `[x0, x1] x [y0, y1]`."""
    if not ring:
        return False
    rxs = [float(p[0]) for p in ring]
    rys = [float(p[1]) for p in ring]
    if max(rxs) < x0 or min(rxs) > x1 or max(rys) < y0 or min(rys) > y1:
        return False
    return True


def _ring_overlaps_polyline(ring: Sequence[Sequence[float]], pts: Sequence[Tuple[float, float]], radius: float) -> bool:
    """Returns True if any vertex/edge of `ring` is within `radius` of `pts`, or if `pts` passes inside `ring`."""
    if not ring or len(pts) < 2:
        return False
    poly2d = [(float(p[0]), float(p[1])) for p in ring]
    n = len(poly2d)
    for i in range(n):
        p0 = poly2d[i]
        p1 = poly2d[(i + 1) % n]
        if _dist_to_polyline(p0[0], p0[1], pts, radius):
            return True
        if _dist_to_polyline(0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]), pts, radius):
            return True
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        for k in range(17):
            t = k / 16.0
            if point_in_polygon_2d((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t), poly2d):
                return True
    return False


def _build_souq_street_corridor(
    collector: MeshCollector,
    p0: Vec2,
    p1: Vec2,
    street_w: float = 5.8,
    wall_h: float = 8.4,
    with_arches: bool = True,
    with_shops: bool = True,
    vaulted_roof: bool = False,
    seed: int = 100,
) -> None:
    """
    Builds a continuous, street-aligned Jerusalem limestone corridor between `p0` and `p1`
    with:
      - Recessed arched ground-floor shopfronts/portals with zero-gap stone archivolts
      - Sloped shop awnings, spice/produce stalls, and hanging textile displays
      - Moulded mid-facade belt course, upper window sills, and open painted wooden shutters
      - Pointed stone flying-buttress arches with hanging souq lanterns (or continuous vaulted skylight bays)
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    seg_len = math.hypot(dx, dy)
    if seg_len < 6.0:
        return
    tx, ty = dx / seg_len, dy / seg_len
    nx, ny = ty, -tx
    n_bays = max(1, int(round(seg_len / 13.0)))
    hw = 0.5 * street_w

    uv_choices = ("souq_open", "door_wood", "souq_open", "window_glass")
    awning_mats = ("mat_awning_red", "mat_awning_green", "mat_awning_striped")

    for b_i in range(n_bays):
        t0 = b_i / n_bays
        t1 = (b_i + 1) / n_bays
        tm = 0.5 * (t0 + t1)
        bx0 = (p0[0] + dx * t0, p0[1] + dy * t0)
        bx1 = (p0[0] + dx * t1, p0[1] + dy * t1)
        mx, my = p0[0] + dx * tm, p0[1] + dy * tm
        bz = street_elevation_z(mx, my)

        for side_idx, side_s in enumerate((-1.0, 1.0)):
            # Wall face normal pointing into the street corridor
            fnx, fny = -side_s * nx, -side_s * ny
            ftx, fty = (tx, ty) if side_s < 0 else (-tx, -ty)
            rot_wall = math.atan2(fty, ftx)
            w0 = (bx0[0] + nx * side_s * hw, bx0[1] + ny * side_s * hw)
            w1 = (bx1[0] + nx * side_s * hw, bx1[1] + ny * side_s * hw)
            if side_s > 0:
                w0, w1 = w1, w0

            wall_mat = "mat_stone_ashlar" if (b_i + side_idx) % 2 == 0 else "mat_stone_mizzi"
            collector.add_oriented_box(
                "Buildings_Walls",
                wall_mat,
                w0,
                w1,
                bz - 1.8,
                bz + wall_h,
                -2.6,
                0.0,
                (fnx, fny),
                uv_scale=0.25,
                cap_back=True,
            )
            # Mid-facade horizontal stone string course (`bz + 4.10 .. 4.28m`)
            collector.add_oriented_box(
                "Buildings_Details",
                "mat_stone_mizzi",
                w0,
                w1,
                bz + 4.08,
                bz + 4.28,
                -0.2,
                0.16,
                (fnx, fny),
                uv_scale=0.32,
                cap_back=True,
            )
            # Roofline projecting stone cornice
            collector.add_oriented_box(
                "Buildings_Details",
                "mat_stone_boss",
                w0,
                w1,
                bz + wall_h,
                bz + wall_h + 0.36,
                -2.7,
                0.24,
                (fnx, fny),
                uv_scale=0.30,
                cap_back=True,
            )
            # Shallow traditional stone dome atop every 3rd building bay to enrich the Old City skyline
            if (b_i + side_idx) % 3 == 0:
                collector.add_dome(
                    "Buildings_Details",
                    "mat_stone_ashlar_aged",
                    (mx + nx * side_s * (hw + 1.3), my + ny * side_s * (hw + 1.3), bz + wall_h + 0.25),
                    radius=1.85,
                    height=1.35,
                    rings=5,
                    segments=10,
                )

            for sub_idx, sub_off in enumerate((-0.24, 0.24)):
                st = tm + sub_off / n_bays
                sx = p0[0] + dx * st + nx * side_s * (hw - 0.01)
                sy = p0[1] + dy * st + ny * side_s * (hw - 0.01)
                sz = street_elevation_z(sx, sy)
                if with_shops:
                    uv_key = uv_choices[(seed + b_i * 3 + side_idx + sub_idx) % len(uv_choices)]
                    add_arched_opening_panel(
                        collector,
                        "Buildings_Openings",
                        "mat_wood_doors",
                        (sx, sy),
                        sz,
                        sz + 2.1,
                        1.08,
                        2.35,
                        0.01,
                        (ftx, fty),
                        (fnx, fny),
                        uv_rect=WOOD_DOOR_UV[uv_key],
                        num_voussoirs=7,
                        add_wall_spandrel=False,
                    )
                    add_arch_voussoirs(
                        collector,
                        "Buildings_Details",
                        "mat_stone_boss",
                        (sx, sy, sz + 2.1),
                        (ftx, fty),
                        (fnx, fny),
                        span_w=2.35,
                        arch_h=1.08,
                        ring_thickness=0.34,
                        offset_n0=-0.04,
                        offset_n1=0.22,
                        num_voussoirs=7,
                    )
                    # Projecting sloped shop awning & hanging carpet/scarf rack on alternating bays
                    if (b_i + side_idx + sub_idx) % 2 == 0 and not vaulted_roof:
                        aw_mat = awning_mats[(seed + b_i + side_idx + sub_idx) % len(awning_mats)]
                        collector.add_box(
                            "StreetProps",
                            aw_mat,
                            (sx + fnx * 0.48, sy + fny * 0.48, sz + 3.48),
                            (2.55, 0.92, 0.14),
                            rot_z=rot_wall,
                            uv_scale=0.45,
                            bottom_face=True,
                        )
                    if (b_i + sub_idx) % 2 == 1:
                        rug_mat = "mat_awning_red" if (b_i + side_idx) % 2 == 0 else "mat_awning_striped"
                        collector.add_box(
                            "StreetProps",
                            rug_mat,
                            (sx + ftx * 1.42 + fnx * 0.14, sy + fty * 1.42 + fny * 0.14, sz + 2.15),
                            (0.55, 0.08, 1.45),
                            rot_z=rot_wall,
                            uv_scale=0.5,
                        )
                    if b_i % 2 == 0 and sub_idx == 0:
                        build_souq_display(
                            collector,
                            (sx + ftx * 1.35 + fnx * 0.52, sy + fty * 1.35 + fny * 0.52, sz),
                            (ftx, fty),
                            (fnx, fny),
                            seed=seed + b_i * 17 + side_idx,
                        )
                else:
                    # Compound wall buttress, recessed monastery doorway, framed street plaque & wall lantern when `with_shops=False`
                    if sub_idx == 0:
                        collector.add_box(
                            "Buildings_Details",
                            "mat_stone_boss",
                            (sx + fnx * 0.28, sy + fny * 0.28, sz + 1.8),
                            (0.95, 0.56, 3.6),
                            rot_z=rot_wall,
                            uv_scale=0.26,
                        )
                        # Framed trilingual cobalt-blue ceramic street-sign plaque (with dressed Mizzi stone border) + wall lantern
                        collector.add_box(
                            "Buildings_Details",
                            "mat_stone_mizzi",
                            (sx + ftx * 1.65 + fnx * 0.04, sy + fty * 1.65 + fny * 0.04, sz + 2.85),
                            (0.56, 0.08, 0.36),
                            rot_z=rot_wall,
                            uv_scale=0.45,
                        )
                        collector.add_box(
                            "Buildings_Details",
                            "mat_tile_iznik",
                            (sx + ftx * 1.65 + fnx * 0.07, sy + fty * 1.65 + fny * 0.07, sz + 2.85),
                            (0.44, 0.06, 0.26),
                            rot_z=rot_wall,
                            uv_scale=0.5,
                        )
                        build_wall_lantern(
                            collector,
                            (sx + ftx * 1.65 + fnx * 0.08, sy + fty * 1.65 + fny * 0.08, sz + 3.55),
                            (ftx, fty),
                            (fnx, fny),
                        )
                    else:
                        # Recessed arched wooden monastery / compound portal doorway between the buttress piers
                        add_arched_opening_panel(
                            collector,
                            "Buildings_Openings",
                            "mat_wood_doors",
                            (sx, sy),
                            sz,
                            sz + 2.05,
                            0.92,
                            1.85,
                            0.01,
                            (ftx, fty),
                            (fnx, fny),
                            uv_rect=WOOD_DOOR_UV["door_wood"],
                            num_voussoirs=7,
                            add_wall_spandrel=False,
                        )
                        add_arch_voussoirs(
                            collector,
                            "Buildings_Details",
                            "mat_stone_boss",
                            (sx, sy, sz + 2.05),
                            (ftx, fty),
                            (fnx, fny),
                            span_w=1.85,
                            arch_h=0.92,
                            ring_thickness=0.30,
                            offset_n0=-0.04,
                            offset_n1=0.20,
                            num_voussoirs=7,
                        )
                        collector.add_box(
                            "Buildings_Details",
                            "mat_stone_mizzi",
                            (sx + fnx * 0.08, sy + fny * 0.08, sz + 3.42),
                            (0.85, 0.14, 0.38),
                            rot_z=rot_wall,
                            uv_scale=0.35,
                        )

                # Upper storey arched window with exact matching `span_w` and `arch_h` (zero gap!), stone sill & open wooden shutters
                add_arched_opening_panel(
                    collector,
                    "Buildings_Openings",
                    "mat_wood_doors",
                    (sx, sy),
                    sz + 4.6,
                    sz + 6.0,
                    0.64,
                    1.26,
                    0.01,
                    (ftx, fty),
                    (fnx, fny),
                    uv_rect=WOOD_DOOR_UV["window_glass"],
                    num_voussoirs=5,
                    add_wall_spandrel=False,
                )
                add_arch_voussoirs(
                    collector,
                    "Buildings_Details",
                    "mat_stone_boss",
                    (sx, sy, sz + 6.0),
                    (ftx, fty),
                    (fnx, fny),
                    span_w=1.26,
                    arch_h=0.64,
                    ring_thickness=0.26,
                    offset_n0=-0.04,
                    offset_n1=0.18,
                    num_voussoirs=5,
                )
                # Projecting moulded stone window sill
                collector.add_box(
                    "Buildings_Details",
                    "mat_stone_mizzi",
                    (sx + fnx * 0.12, sy + fny * 0.12, sz + 4.52),
                    (1.52, 0.24, 0.16),
                    rot_z=rot_wall,
                    uv_scale=0.35,
                    bottom_face=True,
                )
                # Paired open louvred wooden shutters flanking the upper window
                for sh_s in (-0.82, 0.82):
                    collector.add_box(
                        "Buildings_Openings",
                        "mat_wood_doors",
                        (sx + ftx * sh_s + fnx * 0.08, sy + fty * sh_s + fny * 0.08, sz + 5.35),
                        (0.36, 0.08, 1.48),
                        rot_z=rot_wall,
                        uv_rect=WOOD_DOOR_UV["shutter_louvre"],
                    )
                # Occasional projecting wooden Mashrabiya (oriel bay window) or wrought-iron balcony
                if (b_i + side_idx) % 3 == 1 and sub_idx == 1:
                    collector.add_box(
                        "Buildings_Details",
                        "mat_bark",
                        (sx + fnx * 0.36, sy + fny * 0.36, sz + 5.45),
                        (1.48, 0.68, 1.85),
                        rot_z=rot_wall,
                        uv_scale=0.35,
                        bottom_face=True,
                    )

        if vaulted_roof:
            # Continuous vaulted stone roof on left & right sides leaving a central 1.1m daylight skylight strip
            for v_side in (-1.0, 1.0):
                collector.add_oriented_box(
                    "Buildings_Walls",
                    "mat_stone_ashlar",
                    bx0,
                    bx1,
                    bz + wall_h - 0.35,
                    bz + wall_h + 0.35,
                    v_side * (0.55 if v_side > 0 else (hw + 0.4)),
                    v_side * ((hw + 0.4) if v_side > 0 else 0.55),
                    (nx, ny),
                    uv_scale=0.26,
                    cap_back=True,
                    cap_bottom=True,
                )
        if with_arches and (b_i % 2 == 0 or vaulted_roof):
            # True pointed 3D stone flying-buttress arch + spandrel wall spanning across the street + hanging souq lantern
            for a_side in (-1.0, 1.0):
                collector.add_box(
                    "Buildings_Walls",
                    "mat_stone_boss",
                    (mx + nx * a_side * (hw - 0.32), my + ny * a_side * (hw - 0.32), bz + 2.6),
                    (0.85, 0.65, 5.2),
                    rot_z=math.atan2(ty, tx),
                    uv_scale=0.26,
                )
            add_arch_spandrel_wall(
                collector,
                "Buildings_Walls",
                "mat_stone_boss",
                (mx, my, bz + 4.35),
                (nx, ny),
                (tx, ty),
                span_w=street_w - 0.5,
                arch_h=1.75,
                z_top=bz + 6.65,
                offset_n0=-0.48,
                offset_n1=0.48,
                num_voussoirs=9,
                emit_soffit=True,
            )
            for face_dir in (-1.0, 1.0):
                add_arch_voussoirs(
                    collector,
                    "Buildings_Details",
                    "mat_stone_mizzi",
                    (mx + tx * (face_dir * 0.48), my + ty * (face_dir * 0.48), bz + 4.35),
                    (nx * face_dir, ny * face_dir),
                    (tx * face_dir, ty * face_dir),
                    span_w=street_w - 0.5,
                    arch_h=1.75,
                    ring_thickness=0.34,
                    offset_n0=-0.04,
                    offset_n1=0.14,
                    num_voussoirs=9,
                )
            collector.add_box("StreetProps", "mat_iron_dark", (mx, my, bz + 5.75), (0.05, 0.05, 0.90))
            collector.add_box("StreetProps", "mat_lantern_glow", (mx, my, bz + 5.18), (0.28, 0.28, 0.38), bottom_face=True)



from fetch_osm_and_dem import street_elevation_z
from props_lib import (
    MeshCollector,
    build_bollard,
    build_cypress_tree,
    build_fascia_sign,
    build_garbage_pile,
    build_olive_tree,
    build_souq_display,
    build_street_bin,
    build_street_lamp,
    build_street_name_plate,
    build_street_trash_scatter,
    build_wall_lantern,
    build_wheelie_bin,
)

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


# ==============================================================================
# 1. OLD CITY WALL PERIMETER POLYGON & SHARED FORTIFICATION BUILDERS
# ==============================================================================

# Approximate closed polygon of the Ottoman Old City Walls (plus immediate 35m exterior apron
# around the gates) used to filter out distant modern West/East Jerusalem buildings outside the walls.
OLD_CITY_WALL_POLYGON: List[Vec2] = [
    (-22.0, 5.0),
    (-188.0, 195.0),
    (-175.0, 292.0),
    (-135.0, 305.0),
    (-75.0, 335.0),
    (-20.0, 372.0),
    (40.0, 372.0),
    (175.0, 505.0),
    (252.0, 565.0),
    (310.0, 622.0),
    (430.0, 652.0),
    (575.0, 706.0),
    (708.0, 750.0),
    (820.0, 778.0),
    (865.0, 692.0),
    (876.0, 560.0),
    (888.0, 408.0),
    (904.0, 260.0),
    (935.0, 58.0),
    (768.0, -86.0),
    (746.0, -178.0),
    (584.0, -204.0),
    (526.0, -258.0),
    (441.0, -360.0),
    (286.0, -410.0),
    (180.0, -416.0),
    (5.0, -390.0),
    (6.0, -115.0),
    (-10.0, -15.0),
]


def is_inside_old_city_walls(x: float, y: float, margin: float = 25.0) -> bool:
    if point_in_polygon_2d((x, y), OLD_CITY_WALL_POLYGON):
        return True
    # Also allow immediate gate plazas within `margin` of the wall polygon segments
    n = len(OLD_CITY_WALL_POLYGON)
    for i in range(n):
        x0, y0 = OLD_CITY_WALL_POLYGON[i]
        x1, y1 = OLD_CITY_WALL_POLYGON[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        l2 = dx * dx + dy * dy
        t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / l2)) if l2 > 1e-6 else 0.0
        if math.hypot(x - (x0 + t * dx), y - (y0 + t * dy)) <= margin:
            return True
    return False


def build_ottoman_wall_polyline(
    collector: MeshCollector,
    pts: Sequence[Vec2],
    wall_height: float = 12.2,
    thickness: float = 3.6,
    gate_gaps: Sequence[Tuple[float, float, float]] = (),
    bastion_every: int = 3,
) -> None:
    """
    Extrudes the monumental 1538 Ottoman Old City Wall along `pts` with:
      - Lower battered bossed plinth (`mat_stone_boss`)
      - Main coursed ashlar curtain wall (`mat_stone_ashlar_aged`)
      - Walkable upper rampart allure (`mat_paving_flagstone`)
      - Crenellated merlons (`1.35m` tall) + arrow-slit embrasures + stone rosettes
      - Rectangular projecting defensive towers every `bastion_every` vertices
      - Skips segments inside `gate_gaps` (`(gx, gy, gap_radius)`) where historic gatehouses sit.
    """
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1.5:
            continue

        # Subdivide wall segments into <= 5.0m sub-segments so the wall hugs topography and joins gatehouses tightly
        n_sub = max(1, int(math.ceil(seg_len / 5.0)))
        nx, ny = dy / seg_len, -dx / seg_len

        for s in range(n_sub):
            t0 = s / n_sub
            t1 = (s + 1) / n_sub
            sp0 = (p0[0] + dx * t0, p0[1] + dy * t0)
            sp1 = (p0[0] + dx * t1, p0[1] + dy * t1)
            mx, my = 0.5 * (sp0[0] + sp1[0]), 0.5 * (sp0[1] + sp1[1])

            if any(math.hypot(mx - gx, my - gy) < (grad * 0.72) for gx, gy, grad in gate_gaps):
                continue

            sub_len = math.hypot(sp1[0] - sp0[0], sp1[1] - sp0[1])
            z_ground = min(street_elevation_z(sp0[0], sp0[1]), street_elevation_z(sp1[0], sp1[1]))
            z_base = z_ground - 2.0
            z_top = z_ground + wall_height

            # 1. Heavy bossed lower plinth (bottom 4.2m)
            collector.add_oriented_box(
                "CityWalls",
                "mat_stone_boss",
                sp0,
                sp1,
                z_base,
                z_ground + 4.2,
                -thickness * 0.52,
                thickness * 0.52,
                (nx, ny),
                uv_scale=0.22,
                cap_back=True,
            )
            # 2. Upper coursed limestone curtain wall
            collector.add_oriented_box(
                "CityWalls",
                "mat_stone_ashlar_aged",
                sp0,
                sp1,
                z_ground + 4.2,
                z_top,
                -thickness * 0.48,
                thickness * 0.48,
                (nx, ny),
                uv_scale=0.24,
                cap_back=True,
            )
            # 3. Rampart allure walkway & projecting horizontal string course
            collector.add_oriented_box(
                "CityWalls",
                "mat_paving_flagstone",
                sp0,
                sp1,
                z_top - 0.25,
                z_top + 0.05,
                -thickness * 0.46,
                thickness * 0.46,
                (nx, ny),
                uv_scale=0.35,
                cap_back=True,
            )
            collector.add_oriented_box(
                "CityWalls",
                "mat_stone_mizzi",
                sp0,
                sp1,
                z_top - 0.35,
                z_top,
                -thickness * 0.54,
                thickness * 0.54,
                (nx, ny),
                uv_scale=0.35,
                cap_back=True,
            )

            # 4. Crenellated merlons along outer crest
            merlon_pitch = 2.4
            n_merlons = max(1, int(sub_len / merlon_pitch))
            tx, ty = (sp1[0] - sp0[0]) / sub_len, (sp1[1] - sp0[1]) / sub_len
            for m_idx in range(n_merlons):
                mt0 = (m_idx + 0.12) / n_merlons
                mt1 = (m_idx + 0.72) / n_merlons
                mp0 = (sp0[0] + (sp1[0] - sp0[0]) * mt0, sp0[1] + (sp1[1] - sp0[1]) * mt0)
                mp1 = (sp0[0] + (sp1[0] - sp0[0]) * mt1, sp0[1] + (sp1[1] - sp0[1]) * mt1)
                collector.add_oriented_box(
                    "CityWalls",
                    "mat_stone_boss",
                    mp0,
                    mp1,
                    z_top,
                    z_top + 1.35,
                    thickness * 0.12,
                    thickness * 0.48,
                    (nx, ny),
                    uv_scale=0.28,
                    cap_back=True,
                )

            # 5. Arrow-slit embrasures & decorative stone rosettes on segments >= 8m
            if sub_len >= 8.0:
                slit_xy = (mx + nx * (thickness * 0.49), my + ny * (thickness * 0.49))
                rot_wall = math.atan2(ty, tx)
                collector.add_box(
                    "Buildings_Openings",
                    "mat_iron_dark",
                    (slit_xy[0], slit_xy[1], z_ground + 7.8),
                    (0.22, 0.12, 1.65),
                    rot_z=rot_wall,
                )

        # 6. Projecting Square Bastion Tower at periodic vertices (outside gate gaps)
        if bastion_every > 0 and (i % bastion_every == 1):
            bx, by = p0
            if not any(math.hypot(bx - gx, by - gy) < (grad + 10.0) for gx, gy, grad in gate_gaps):
                bz = street_elevation_z(bx, by)
                rot_b = math.atan2(dy, dx)
                collector.add_box(
                    "CityWalls",
                    "mat_stone_boss",
                    (bx + nx * 1.4, by + ny * 1.4, bz + (wall_height + 2.2) * 0.5 - 1.0),
                    (7.2, 6.4, wall_height + 4.2),
                    rot_z=rot_b,
                    uv_scale=0.22,
                )


def build_ottoman_gatehouse(
    collector: MeshCollector,
    center_xy: Vec2,
    heading_inward_deg: float,
    gate_name: str,
    width: float = 13.5,
    depth: float = 9.5,
    height: float = 14.5,
    twin_towers: bool = False,
    sealed_double_arch: bool = False,
    sign_rect: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """
    Builds an authentic 16th-century Ottoman/Crusader monumental city gatehouse at `center_xy`
    oriented along `heading_inward_deg` (compass degrees pointing into the city: 0=North, 90=East, 180=South, 270=West).
    Walkable arched passage runs right through `center_xy` (unless `sealed_double_arch=True` for Golden Gate).
    """
    cx, cy = center_xy
    cz = street_elevation_z(cx, cy)
    # Convert compass heading (0=North=+Y, 90=East=+X) to mathematical angle in XY
    rad = math.radians(90.0 - heading_inward_deg)
    # Inward unit vector `fwd`, outward unit vector `out_n = -fwd`, right tangent `rt`
    fx, fy = math.cos(rad), math.sin(rad)
    ox, oy = -fx, -fy
    rx, ry = fy, -fx
    rot_z = math.atan2(ry, rx)

    arch_w = 4.2
    arch_h = 5.4
    side_pier_w = (width - arch_w) * 0.5

    # 0. Integral Left & Right Flanking Rampart Curtain Wing Extensions (welds gatehouse into adjacent city walls with zero gap)
    wing_w = 13.5 if twin_towers else 7.2
    wing_h = max(10.5, height - 2.2)
    wing_base_z = max(cz, -12.5) if sealed_double_arch else cz
    for side in (-1.0, 1.0):
        wx0 = cx + rx * (side * (width * 0.44))
        wy0 = cy + ry * (side * (width * 0.44))
        wx1 = cx + rx * (side * (width * 0.5 + wing_w))
        wy1 = cy + ry * (side * (width * 0.5 + wing_w))
        wp0, wp1 = ((wx0, wy0), (wx1, wy1)) if side > 0 else ((wx1, wy1), (wx0, wy0))
        collector.add_oriented_box(
            "CityWalls",
            "mat_stone_boss",
            wp0,
            wp1,
            wing_base_z - 2.5,
            wing_base_z + 4.2,
            -2.1,
            2.1,
            (ox, oy),
            uv_scale=0.22,
            cap_back=True,
        )
        collector.add_oriented_box(
            "CityWalls",
            "mat_stone_ashlar_aged",
            wp0,
            wp1,
            wing_base_z + 4.2,
            wing_base_z + wing_h,
            -1.9,
            1.9,
            (ox, oy),
            uv_scale=0.24,
            cap_back=True,
        )
        collector.add_oriented_box(
            "CityWalls",
            "mat_stone_mizzi",
            wp0,
            wp1,
            wing_base_z + wing_h - 0.32,
            wing_base_z + wing_h + 0.04,
            -2.05,
            2.05,
            (ox, oy),
            uv_scale=0.32,
            cap_back=True,
        )
        n_w_merlons = 5 if twin_towers else 3
        for wm_idx in range(n_w_merlons):
            w_off = width * 0.5 + 1.0 + wm_idx * 2.2
            for f_nx, f_ny in ((ox, oy), (fx, fy)):
                wmx = cx + rx * (side * w_off) + f_nx * 1.35
                wmy = cy + ry * (side * w_off) + f_ny * 1.35
                collector.add_box(
                    "CityWalls",
                    "mat_stone_boss",
                    (wmx, wmy, wing_base_z + wing_h + 0.68),
                    (1.3, 0.82, 1.35),
                    rot_z=rot_z,
                    uv_scale=0.26,
                )

    if sealed_double_arch:
        # Golden Gate (Bab al-Rahma / Bab al-Dhahabi): Monumental Byzantine/Umayyad/Ottoman double-arched sealed gatehouse
        cz_esp = max(cz, -12.5)
        collector.add_box(
            "CityWalls",
            "mat_stone_boss",
            (cx, cy, 0.5 * (cz - 2.0 + cz_esp)),
            (width + 0.6, depth + 0.6, max(1.0, cz_esp - (cz - 2.0))),
            rot_z=rot_z,
            uv_scale=0.22,
        )
        collector.add_box(
            "CityWalls",
            "mat_stone_ashlar",
            (cx, cy, cz_esp + height * 0.5 - 0.5),
            (width, depth, height + 1.0),
            rot_z=rot_z,
            uv_scale=0.24,
        )
        # Projecting Byzantine carved cornice & frieze at mid-height (`cz_esp + 7.4m`) and roof (`cz_esp + height`)
        collector.add_box(
            "CityWalls",
            "mat_stone_mizzi",
            (cx, cy, cz_esp + 7.45),
            (width + 0.45, depth + 0.45, 0.36),
            rot_z=rot_z,
            uv_scale=0.30,
        )
        collector.add_box(
            "CityWalls",
            "mat_stone_mizzi",
            (cx, cy, cz_esp + height + 0.25),
            (width + 0.6, depth + 0.6, 0.5),
            rot_z=rot_z,
            uv_scale=0.30,
        )
        for face_nx, face_ny, face_tx, face_ty in ((ox, oy, rx, ry), (fx, fy, -rx, -ry)):
            # Central Corinthian marble dividing column & capital between the twin arches (Bab al-Rahma & Bab al-Tawba)
            col_x = cx + face_nx * (depth * 0.5 + 0.16)
            col_y = cy + face_ny * (depth * 0.5 + 0.16)
            collector.add_branch_tube(
                "CityWalls",
                "mat_stone_mizzi",
                (col_x, col_y, cz_esp + 0.2),
                (col_x, col_y, cz_esp + 4.65),
                0.34,
                0.31,
                segments=10,
            )
            collector.add_box(
                "CityWalls",
                "mat_stone_mizzi",
                (col_x, col_y, cz_esp + 4.82),
                (0.86, 0.86, 0.42),
                rot_z=rot_z,
            )
            # Outer flanking pilasters
            for pil_s in (-6.8, 6.8):
                px_p = cx + face_tx * pil_s + face_nx * (depth * 0.5 + 0.12)
                py_p = cy + face_ty * pil_s + face_ny * (depth * 0.5 + 0.12)
                collector.add_box(
                    "CityWalls",
                    "mat_stone_mizzi",
                    (px_p, py_p, cz_esp + 3.7),
                    (0.75, 0.32, 7.4),
                    rot_z=rot_z,
                )
            for sign_s in (-1.0, 1.0):
                ax = cx + face_tx * (sign_s * 3.4) + face_nx * (depth * 0.5 + 0.02)
                ay = cy + face_ty * (sign_s * 3.4) + face_ny * (depth * 0.5 + 0.02)
                # Recessed sealed stone masonry infill (coursed ashlar blocks sealing the twin portals)
                add_arched_opening_panel(
                    collector,
                    "CityWalls",
                    "mat_stone_ashlar_aged",
                    (ax, ay),
                    cz_esp,
                    cz_esp + 4.8,
                    2.0,
                    4.2,
                    0.02,
                    (face_tx, face_ty),
                    (face_nx, face_ny),
                    uv_rect=(0.0, 0.0, 1.0, 1.0),
                    num_voussoirs=11,
                    add_wall_spandrel=False,
                )
                # Inner stone voussoir ring
                add_arch_voussoirs(
                    collector,
                    "CityWalls",
                    "mat_stone_boss",
                    (ax, ay, cz_esp + 4.8),
                    (face_tx, face_ty),
                    (face_nx, face_ny),
                    span_w=4.35,
                    arch_h=2.08,
                    ring_thickness=0.46,
                    offset_n0=-0.04,
                    offset_n1=0.34,
                    num_voussoirs=11,
                    keystone_boost=0.06,
                )
                # Outer carved Byzantine floral archivolt hood
                add_arch_voussoirs(
                    collector,
                    "CityWalls",
                    "mat_stone_mizzi",
                    (ax, ay, cz_esp + 4.8),
                    (face_tx, face_ty),
                    (face_nx, face_ny),
                    span_w=4.95,
                    arch_h=2.38,
                    ring_thickness=0.32,
                    offset_n0=0.02,
                    offset_n1=0.44,
                    num_voussoirs=13,
                    keystone_boost=0.08,
                )
                # Upper clerestory arched lunette windows above each bay
                wx_u = cx + face_tx * (sign_s * 3.4) + face_nx * (depth * 0.5 + 0.04)
                wy_u = cy + face_ty * (sign_s * 3.4) + face_ny * (depth * 0.5 + 0.04)
                add_arched_opening_panel(
                    collector,
                    "Buildings_Openings",
                    "mat_iron_dark",
                    (wx_u, wy_u),
                    cz_esp + 8.6,
                    cz_esp + 10.6,
                    0.85,
                    1.6,
                    0.02,
                    (face_tx, face_ty),
                    (face_nx, face_ny),
                    uv_rect=(0.0, 0.0, 1.0, 1.0),
                    num_voussoirs=7,
                    add_wall_spandrel=False,
                )
                add_arch_voussoirs(
                    collector,
                    "CityWalls",
                    "mat_stone_mizzi",
                    (wx_u, wy_u, cz_esp + 10.6),
                    (face_tx, face_ty),
                    (face_nx, face_ny),
                    span_w=1.7,
                    arch_h=0.92,
                    ring_thickness=0.28,
                    offset_n0=-0.02,
                    offset_n1=0.22,
                    num_voussoirs=7,
                    keystone_boost=0.04,
                )
                # Carved Ottoman circular stone rosette medallion flanking the upper arches
                ros_x = cx + face_tx * (sign_s * 5.6) + face_nx * (depth * 0.5)
                ros_y = cy + face_ty * (sign_s * 5.6) + face_ny * (depth * 0.5)
                collector.add_branch_tube(
                    "CityWalls",
                    "mat_stone_mizzi",
                    (ros_x, ros_y, cz_esp + 9.8),
                    (ros_x + face_nx * 0.18, ros_y + face_ny * 0.18, cz_esp + 9.8),
                    0.56,
                    0.52,
                    segments=12,
                )
        # Stepped Ottoman crenellations (merlons) along both facades of Golden Gate
        for face_nx, face_ny in ((ox, oy), (fx, fy)):
            for m_idx in range(7):
                ms = -width * 0.44 + (m_idx / 6.0) * (width * 0.88)
                mx = cx + rx * ms + face_nx * (depth * 0.42)
                my = cy + ry * ms + face_ny * (depth * 0.42)
                collector.add_box(
                    "CityWalls",
                    "mat_stone_boss",
                    (mx, my, cz_esp + height + 1.1),
                    (1.25, 0.78, 1.35),
                    rot_z=rot_z,
                    uv_scale=0.26,
                )
        # Twin low lead domes on radial stone drums atop the roof of Golden Gate
        for sign_s in (-1.0, 1.0):
            dx_d = cx + rx * (sign_s * 3.4)
            dy_d = cy + ry * (sign_s * 3.4)
            collector.add_branch_tube(
                "CityWalls",
                "mat_stone_mizzi",
                (dx_d, dy_d, cz_esp + height + 0.3),
                (dx_d, dy_d, cz_esp + height + 1.45),
                2.95,
                2.95,
                segments=14,
            )
            collector.add_dome(
                "CityWalls",
                "mat_dome_lead",
                (dx_d, dy_d, cz_esp + height + 1.4),
                radius=2.9,
                height=2.5,
                rings=7,
                segments=14,
            )
        return

    # 1. Left & Right Flanking Gatehouse Piers (leaving the central `arch_w` passage open and walkable!)
    for side in (-1.0, 1.0):
        px = cx + rx * (side * (arch_w * 0.5 + side_pier_w * 0.5))
        py = cy + ry * (side * (arch_w * 0.5 + side_pier_w * 0.5))
        collector.add_box(
            "CityWalls",
            "mat_stone_boss",
            (px, py, cz + height * 0.5 - 0.5),
            (side_pier_w, depth, height + 1.0),
            rot_z=rot_z,
            uv_scale=0.22,
        )
        if twin_towers:
            # Projecting flanking polygonal/square bastion towers (Damascus Gate style)
            tx_b = px + rx * (side * 3.6) + ox * 2.2
            ty_b = py + ry * (side * 3.6) + oy * 2.2
            tw_h = height + 2.8
            collector.add_box(
                "CityWalls",
                "mat_stone_boss",
                (tx_b, ty_b, cz + tw_h * 0.5 - 0.5),
                (6.8, depth + 4.4, tw_h + 1.0),
                rot_z=rot_z,
                uv_scale=0.22,
            )
            # Projecting stone cornice & stepped Ottoman crenellations (merlons) atop both flanking towers
            collector.add_box(
                "CityWalls",
                "mat_stone_mizzi",
                (tx_b, ty_b, cz + tw_h + 0.18),
                (7.3, depth + 4.9, 0.38),
                rot_z=rot_z,
                uv_scale=0.30,
            )
            for m_s in (-2.4, 0.0, 2.4):
                for f_nx, f_ny in ((ox, oy), (fx, fy)):
                    mx_t = tx_b + rx * m_s + f_nx * ((depth + 4.4) * 0.5 - 0.42)
                    my_t = ty_b + ry * m_s + f_ny * ((depth + 4.4) * 0.5 - 0.42)
                    collector.add_box(
                        "CityWalls",
                        "mat_stone_boss",
                        (mx_t, my_t, cz + tw_h + 1.05),
                        (1.4, 0.85, 1.5),
                        rot_z=rot_z,
                        uv_scale=0.26,
                    )
            # Corbeled machicolation box on each flanking tower
            collector.add_box(
                "CityWalls",
                "mat_stone_boss",
                (tx_b + ox * ((depth + 4.4) * 0.5 + 0.38), ty_b + oy * ((depth + 4.4) * 0.5 + 0.38), cz + tw_h - 2.2),
                (2.8, 0.78, 1.9),
                rot_z=rot_z,
                uv_scale=0.25,
                bottom_face=True,
            )

    # 2. Upper Gatehouse Chamber above the arched portal (`z = cz + arch_h .. cz + height`)
    collector.add_box(
        "CityWalls",
        "mat_stone_boss",
        (cx, cy, cz + arch_h + (height - arch_h) * 0.5),
        (arch_w + 0.6, depth, height - arch_h),
        rot_z=rot_z,
        uv_scale=0.22,
        bottom_face=True,
    )

    # 3. Outer & Inner Monumental Pointed Stone Arch Voussoirs + Solid Spandrel Walls (zero haunch holes!)
    for face_sign, n_vec in ((1.0, (ox, oy)), (-1.0, (fx, fy))):
        portal_x = cx + n_vec[0] * (depth * 0.5)
        portal_y = cy + n_vec[1] * (depth * 0.5)
        t_vec = (rx * face_sign, ry * face_sign)
        add_arch_spandrel_wall(
            collector,
            "CityWalls",
            "mat_stone_boss",
            (portal_x, portal_y, cz + 3.4),
            t_vec,
            n_vec,
            span_w=arch_w,
            arch_h=2.0,
            z_top=cz + arch_h + 0.15,
            offset_n0=-depth * 0.5,
            offset_n1=0.0,
            num_voussoirs=11,
            emit_soffit=(face_sign > 0),
        )
        add_arch_voussoirs(
            collector,
            "CityWalls",
            "mat_stone_boss",
            (portal_x, portal_y, cz + 3.4),
            t_vec,
            n_vec,
            span_w=arch_w,
            arch_h=2.0,
            ring_thickness=0.52,
            offset_n0=-0.05,
            offset_n1=0.38,
            num_voussoirs=11,
            keystone_boost=0.07,
        )
        if twin_towers:
            # Tall pointed Mamluk/Ottoman recessed hood arch on Damascus Gate (both outer and inner facades)
            add_arch_voussoirs(
                collector,
                "CityWalls",
                "mat_stone_mizzi",
                (portal_x, portal_y, cz + 5.8),
                t_vec,
                n_vec,
                span_w=arch_w + 1.8,
                arch_h=3.2,
                ring_thickness=0.45,
                offset_n0=-0.02,
                offset_n1=0.32,
                num_voussoirs=13,
                keystone_boost=0.08,
            )
        elif "New Gate" not in gate_name:
            # Cycle 3 Detail: Outer pointed Ottoman tympanum hood arch framing the foundation plaque on Zion, Dung, Herod's & Lions' Gates
            add_arch_voussoirs(
                collector,
                "CityWalls",
                "mat_stone_mizzi",
                (portal_x, portal_y, cz + 4.9),
                t_vec,
                n_vec,
                span_w=arch_w + 1.1,
                arch_h=2.2,
                ring_thickness=0.36,
                offset_n0=-0.02,
                offset_n1=0.28,
                num_voussoirs=11,
                keystone_boost=0.06,
            )

        # Cycle 2 Detail: Recessed Ottoman Arabic Foundation Inscription Tablet (Tympanum Plaque)
        ins_z = cz + (7.0 if twin_towers else arch_h + 1.05)
        collector.add_box(
            "CityWalls",
            "mat_stone_mizzi",
            (portal_x + n_vec[0] * 0.14, portal_y + n_vec[1] * 0.14, ins_z),
            (2.15, 0.22, 0.95),
            rot_z=rot_z,
            uv_scale=0.32,
        )
        collector.add_box(
            "CityWalls",
            "mat_stone_ashlar_aged",
            (portal_x + n_vec[0] * 0.21, portal_y + n_vec[1] * 0.21, ins_z),
            (1.85, 0.12, 0.72),
            rot_z=rot_z,
            uv_scale=0.35,
        )
        # Cycle 3 Gate-Specific Architectural Micro-Details (New Gate lanterns, Lions' Gate Baybars Panthers, Dung Gate lintel, Zion Gate pockmarks)
        for r_side in (-1.0, 1.0):
            ros_off = arch_w * 0.5 + side_pier_w * 0.42
            ros_x = portal_x + rx * (r_side * ros_off)
            ros_y = portal_y + ry * (r_side * ros_off)
            ros_z = cz + arch_h + 2.35
            if "New Gate" in gate_name and face_sign > 0:
                # 1889 Sultan Abdulhamid II New Gate: ornate wrought-iron wall lanterns & rusticated corner quoins instead of medieval loopholes
                build_wall_lantern(
                    collector,
                    (ros_x + n_vec[0] * 0.12, ros_y + n_vec[1] * 0.12, cz + 4.2),
                    t_vec,
                    n_vec,
                )
                build_bollard(
                    collector,
                    portal_x + rx * (r_side * (arch_w * 0.5 + 0.65)) + n_vec[0] * 1.4,
                    portal_y + ry * (r_side * (arch_w * 0.5 + 0.65)) + n_vec[1] * 1.4,
                    cz,
                )
                continue
            collector.add_branch_tube(
                "CityWalls",
                "mat_stone_mizzi",
                (ros_x, ros_y, ros_z),
                (ros_x + n_vec[0] * 0.18, ros_y + n_vec[1] * 0.18, ros_z),
                0.56,
                0.52,
                segments=12,
            )
            collector.add_branch_tube(
                "CityWalls",
                "mat_stone_boss",
                (ros_x + n_vec[0] * 0.16, ros_y + n_vec[1] * 0.16, ros_z),
                (ros_x + n_vec[0] * 0.25, ros_y + n_vec[1] * 0.25, ros_z),
                0.30,
                0.24,
                segments=10,
            )
            if "Lions' Gate" in gate_name and face_sign > 0:
                # Sultan Baybars' Sculpted Heraldic Stone Panthers/Lions flanking Lions' Gate (`cz + 5.2m`)
                lion_z = cz + 5.2
                collector.add_box(
                    "CityWalls",
                    "mat_stone_mizzi",
                    (ros_x + n_vec[0] * 0.12, ros_y + n_vec[1] * 0.12, lion_z),
                    (1.65, 0.24, 0.92),
                    rot_z=rot_z,
                    uv_scale=0.35,
                )
                # Sculpted relief body, chest & head of the heraldic panther facing the portal
                collector.add_box(
                    "CityWalls",
                    "mat_stone_boss",
                    (ros_x + n_vec[0] * 0.24, ros_y + n_vec[1] * 0.24, lion_z - 0.04),
                    (1.18, 0.22, 0.48),
                    rot_z=rot_z,
                    uv_scale=0.40,
                )
                collector.add_box(
                    "CityWalls",
                    "mat_stone_boss",
                    (ros_x - rx * (r_side * 0.44) + n_vec[0] * 0.28, ros_y - ry * (r_side * 0.44) + n_vec[1] * 0.28, lion_z + 0.18),
                    (0.38, 0.24, 0.38),
                    rot_z=rot_z,
                    uv_scale=0.40,
                )
            else:
                # Splayed vertical arrow-slit loophole below the rosette
                slit_z = cz + 4.4
                collector.add_box(
                    "CityWalls",
                    "mat_stone_mizzi",
                    (ros_x + n_vec[0] * 0.08, ros_y + n_vec[1] * 0.08, slit_z),
                    (0.56, 0.18, 2.05),
                    rot_z=rot_z,
                    uv_scale=0.32,
                )
                collector.add_box(
                    "Buildings_Openings",
                    "mat_iron_dark",
                    (ros_x + n_vec[0] * 0.15, ros_y + n_vec[1] * 0.15, slit_z),
                    (0.20, 0.10, 1.78),
                    rot_z=rot_z,
                )

    if "Dung Gate" in gate_name:
        # Characteristic low flat-arched reinforced stone/concrete lintel beam spanning Dung Gate's widened portal
        dg_lx = cx + ox * (depth * 0.5 - 0.05)
        dg_ly = cy + oy * (depth * 0.5 - 0.05)
        collector.add_box(
            "CityWalls",
            "mat_stone_mizzi",
            (dg_lx, dg_ly, cz + 3.45),
            (arch_w + 0.4, 0.65, 0.58),
            rot_z=rot_z,
            uv_scale=0.32,
            bottom_face=True,
        )
    if "Zion Gate" in gate_name:
        # Weathered 1948 battle-pocked stone courses & approach bollards on Zion Gate's outer facade
        for pk_s, pk_z in ((-3.8, 3.2), (-2.9, 6.1), (2.8, 3.6), (3.9, 5.8), (-1.9, 7.6), (2.1, 7.4)):
            collector.add_box(
                "CityWalls",
                "mat_stone_rubble",
                (cx + rx * pk_s + ox * (depth * 0.5 + 0.04), cy + ry * pk_s + oy * (depth * 0.5 + 0.04), cz + pk_z),
                (1.25, 0.14, 0.75),
                rot_z=rot_z,
                uv_scale=0.45,
            )

    # 4. Projecting Corbelled Machicolation Box + 4 Stepped Stone Corbels above Outer Portal (except 1889 New Gate)
    if "New Gate" not in gate_name:
        mach_x = cx + ox * (depth * 0.5 + 0.42)
        mach_y = cy + oy * (depth * 0.5 + 0.42)
        collector.add_box(
            "CityWalls",
            "mat_stone_boss",
            (mach_x, mach_y, cz + height - 2.1),
            (3.8, 0.85, 2.2),
            rot_z=rot_z,
            uv_scale=0.25,
            bottom_face=True,
        )
        for cb_s in (-1.45, -0.48, 0.48, 1.45):
            cbx = cx + rx * cb_s + ox * (depth * 0.5 + 0.32)
            cby = cy + ry * cb_s + oy * (depth * 0.5 + 0.32)
            collector.add_box(
                "CityWalls",
                "mat_stone_mizzi",
                (cbx, cby, cz + height - 3.45),
                (0.36, 0.68, 0.65),
                rot_z=rot_z,
                uv_scale=0.30,
                bottom_face=True,
            )

    # 5. Crenellated Merlons + Beveled Stone Merlon Caps atop the Gatehouse Crown (both outer and inner parapets)
    n_merlons = 5
    for f_nx, f_ny in ((ox, oy), (fx, fy)):
        for m_idx in range(n_merlons):
            ms = -width * 0.42 + (m_idx / (n_merlons - 1)) * (width * 0.84)
            mx = cx + rx * ms + f_nx * (depth * 0.38)
            my = cy + ry * ms + f_ny * (depth * 0.38)
            collector.add_box(
                "CityWalls",
                "mat_stone_boss",
                (mx, my, cz + height + 0.75),
                (1.35, 0.85, 1.5),
                rot_z=rot_z,
                uv_scale=0.26,
            )
            collector.add_box(
                "CityWalls",
                "mat_stone_mizzi",
                (mx, my, cz + height + 1.58),
                (1.45, 0.95, 0.22),
                rot_z=rot_z,
                uv_scale=0.32,
            )

    # 6. Open Iron-Plated Wooden Gate Leaves bolted inside the passage walls
    for side in (-1.0, 1.0):
        lx = cx + rx * (side * (arch_w * 0.5 - 0.14))
        ly = cy + ry * (side * (arch_w * 0.5 - 0.14))
        collector.add_box(
            "Buildings_Openings",
            "mat_wood_doors",
            (lx, ly, cz + 2.1),
            (0.18, 2.1, 4.2),
            rot_z=rot_z,
            uv_rect=WOOD_DOOR_UV["door_wood"],
        )

    # 7. Walkable Flagstone Passage Floor
    p00 = (cx - rx * (arch_w * 0.52) - fx * (depth * 0.55), cy - ry * (arch_w * 0.52) - fy * (depth * 0.55), cz + 0.02)
    p10 = (cx + rx * (arch_w * 0.52) - fx * (depth * 0.55), cy + ry * (arch_w * 0.52) - fy * (depth * 0.55), cz + 0.02)
    p11 = (cx + rx * (arch_w * 0.52) + fx * (depth * 0.55), cy + ry * (arch_w * 0.52) + fy * (depth * 0.55), cz + 0.02)
    p01 = (cx - rx * (arch_w * 0.52) + fx * (depth * 0.55), cy - ry * (arch_w * 0.52) + fy * (depth * 0.55), cz + 0.02)
    collector.add_quad("Ground_Paving", "mat_paving_flagstone", p00, p10, p11, p01, uv_scale=0.38, smooth=True, expected_normal=(0.0, 0.0, 1.0))

    if sign_rect is not None:
        sx = cx + rx * (arch_w * 0.5 + 1.1) + ox * (depth * 0.5 + 0.06)
        sy = cy + ry * (arch_w * 0.5 + 1.1) + oy * (depth * 0.5 + 0.06)
        build_street_name_plate(collector, (sx, sy, cz + 2.7), (ox, oy), sign_rect, width=1.1, height=0.55)


# ==============================================================================
# 2. CONTINUOUS GROUND PAVING BUILDER FOR ANY REGION
# ==============================================================================

def build_region_ground_grid(
    collector: MeshCollector,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    step: float = 5.0,
    exclude_fn: Optional[Callable[[float, float], bool]] = None,
    flagstone_fn: Optional[Callable[[float, float], bool]] = None,
) -> None:
    """
    Builds a continuous, walkable stone-paved ground surface across `[x_min..x_max] x [y_min..y_max]`
    evaluated at exact vertices of `street_elevation_z(x, y)` so adjacent regions stitch cleanly.
    """
    nx = max(1, int(round((x_max - x_min) / step)))
    ny = max(1, int(round((y_max - y_min) / step)))
    xs = [x_min + i * ((x_max - x_min) / nx) for i in range(nx + 1)]
    ys = [y_min + j * ((y_max - y_min) / ny) for j in range(ny + 1)]

    for j in range(ny):
        y0, y1 = ys[j], ys[j + 1]
        ym = 0.5 * (y0 + y1)
        for i in range(nx):
            x0, x1 = xs[i], xs[i + 1]
            xm = 0.5 * (x0 + x1)
            if exclude_fn is not None and exclude_fn(xm, ym):
                continue

            z00 = street_elevation_z(x0, y0)
            z10 = street_elevation_z(x1, y0)
            z11 = street_elevation_z(x1, y1)
            z01 = street_elevation_z(x0, y1)

            use_flagstone = flagstone_fn(xm, ym) if flagstone_fn is not None else True
            mat = "mat_paving_flagstone" if use_flagstone else "mat_paving"
            uv_sc = 0.36 if use_flagstone else 0.42

            collector.add_quad(
                "Ground_Paving",
                mat,
                (x0, y0, z00),
                (x1, y0, z10),
                (x1, y1, z11),
                (x0, y1, z01),
                uv_scale=uv_sc,
                smooth=True,
                expected_normal=(0.0, 0.0, 1.0),
            )


# ==============================================================================
# 3. REGION 2: CHRISTIAN QUARTER (`r02_christian_quarter`)
#    - Church of the Holy Sepulchre (Anastasis Rotunda, Catholicon, Crusader Bell Tower, Parvis)
#    - Lutheran Church of the Redeemer & Tall White Bell Tower
#    - Muristan Fountain & Octagonal Market (Suq Aftimos)
#    - Mosque of Omar Minaret
#    - New Gate (Bab al-Jadid) & NW Ottoman Old City Wall
#    - Christian Quarter Streets & OSM Buildings
# ==============================================================================

def _build_church_of_the_holy_sepulchre(collector: MeshCollector, sign_atlas: Dict[str, dict]) -> None:
    """
    Builds the Church of the Holy Sepulchre complex (`x = 162..220, y = 158..218`):
      1. Southern Parvis Courtyard (`x = 182..212, y = 160..178`) with sunken stone steps & colonnade
      2. Crusader Double-Arched Southern Facade (`y = 178, x = 182..212`):
         - Twin monumental pointed Gothic portals (West open with heavy walnut doors, East walled up)
         - Cushion voussoir archivolts, clustered colonnettes, carved lintel frieze, upper twin arched windows
      3. Chapel of the Franks (`x = 208..216, y = 171..178`): external stone stair & blue/lead-domed loggia
      4. 12th-Century Crusader Bell Tower (`x = 180..190, y = 174..184`, `27m` tall) with coupled Gothic belfry arches
      5. Great Anastasis Rotunda (`center = (176.5, 196.5), radius = 13.8m`) with massive cylindrical stone drum
         and conical-hemispherical ribbed lead dome (`mat_dome_lead`) + golden oculus & cross finial
      6. Catholicon Crusader Choir Dome (`center = (201.5, 196.5), radius = 8.2m`) on octagonal drum
      7. Mosque of Omar & Mamluk Minaret (`(188.5, 143.5)`) across the southern parvis
    """
    parvis_z = street_elevation_z(196.0, 170.0)

    # 1. Southern Parvis Courtyard Flagstone Platform & Perimeter Convent Wings
    p00 = (180.0, 158.0, parvis_z + 0.03)
    p10 = (214.0, 158.0, parvis_z + 0.03)
    p11 = (214.0, 179.0, parvis_z + 0.03)
    p01 = (180.0, 179.0, parvis_z + 0.03)
    collector.add_quad("Ground_Paving", "mat_paving_flagstone", p00, p10, p11, p01, uv_scale=0.34, smooth=True, expected_normal=(0.0, 0.0, 1.0))

    # West Parvis Convent Wing (Chapel of St. James / Greek Orthodox baptistery)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar_aged", (177.5, 168.0, parvis_z + 5.2), (5.2, 18.0, 10.4), uv_scale=0.25)
    # East Parvis Convent Wing (Armenian Chapel of St. John / Abraham Monastery)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar_aged", (216.5, 168.0, parvis_z + 5.5), (5.2, 18.0, 11.0), uv_scale=0.25)

    # 2. Main Basilica Nave & Transept Mass (`x = 166..218, y = 178.8..213.2`)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar_aged", (196.0, 196.0, parvis_z + 7.2), (44.0, 34.4, 14.4), uv_scale=0.24)
    # Upper Clerestory & Roof Cornice
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (196.0, 196.0, parvis_z + 14.5), (44.6, 35.0, 0.48), uv_scale=0.32)

    # 3. Crusader Double-Arched Southern Facade (`y = 178.8, x = 183..211`, facing South `(0, -1)`)
    facade_y = 178.8
    # Lower and upper prominent projecting carved Crusader cornices with corbel friezes
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (197.0, facade_y - 0.26, parvis_z + 6.85), (26.2, 0.56, 0.36), uv_scale=0.35, bottom_face=True)
    collector.add_box("Buildings_Details", "mat_stone_boss", (197.0, facade_y - 0.18, parvis_z + 6.55), (26.0, 0.38, 0.24), uv_scale=0.35, bottom_face=True)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (197.0, facade_y - 0.30, parvis_z + 13.85), (26.6, 0.64, 0.42), uv_scale=0.35, bottom_face=True)
    collector.add_box("Buildings_Details", "mat_stone_boss", (197.0, facade_y - 0.20, parvis_z + 13.52), (26.2, 0.42, 0.26), uv_scale=0.35, bottom_face=True)

    # Twin Crusader Portals: West (`x = 193.2`, open entrance) & East (`x = 200.8`, walled-up in 1187)
    for idx_p, px in enumerate((193.2, 200.8)):
        # Carved Crusader lintel frieze & 3 concentric stepped cushion-voussoir archivolts
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (px, facade_y - 0.22, parvis_z + 3.68), (5.4, 0.46, 0.34), uv_scale=0.35, bottom_face=True)
        for span_o, arch_ho, rt_o, n1_o, mat_o in (
            (4.1, 1.95, 0.40, 0.28, "mat_stone_boss"),
            (4.75, 2.28, 0.38, 0.38, "mat_stone_mizzi"),
            (5.4, 2.60, 0.42, 0.48, "mat_stone_boss"),
        ):
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                mat_o,
                (px, facade_y, parvis_z + 3.8),
                (1.0, 0.0),
                (0.0, -1.0),
                span_w=span_o,
                arch_h=arch_ho,
                ring_thickness=rt_o,
                offset_n0=-0.04,
                offset_n1=n1_o,
                num_voussoirs=13,
                keystone_boost=0.06,
            )
        # Clustered marble colonnettes + carved Corinthian capitals flanking the portal jambs
        for side_c in (-2.62, -2.30, -1.98, 1.98, 2.30, 2.62):
            collector.add_branch_tube(
                "Buildings_Details",
                "mat_stone_mizzi",
                (px + side_c, facade_y - 0.26, parvis_z + 0.25),
                (px + side_c, facade_y - 0.26, parvis_z + 3.45),
                0.14,
                0.13,
                segments=8,
            )
            collector.add_box(
                "Buildings_Details",
                "mat_stone_boss",
                (px + side_c, facade_y - 0.26, parvis_z + 3.55),
                (0.32, 0.32, 0.26),
            )
        if idx_p == 0:
            # West Portal: Recessed tympanum + open dark arched entrance with monumental timber doors
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (px, facade_y - 0.02),
                parvis_z,
                parvis_z + 3.8,
                1.95,
                4.0,
                0.02,
                (1.0, 0.0),
                (0.0, -1.0),
                uv_rect=WOOD_DOOR_UV["door_wood"],
                num_voussoirs=9,
                add_wall_spandrel=False,
            )
        else:
            # East Portal: Walled-up Crusader archway with recessed coursed stone masonry
            add_arched_opening_panel(
                collector,
                "Buildings_Details",
                "mat_stone_ashlar",
                (px, facade_y - 0.02),
                parvis_z,
                parvis_z + 3.8,
                1.95,
                4.0,
                0.02,
                (1.0, 0.0),
                (0.0, -1.0),
                uv_rect=(0.0, 0.0, 1.0, 1.0),
                num_voussoirs=9,
                add_wall_spandrel=False,
            )

        # Upper Story Twin Crusader Pointed Windows (`z = parvis_z + 7.2 .. 12.2`) with 3 concentric archivolts & jamb colonnettes
        for span_w_u, arch_h_u, n1_u, mat_u in (
            (3.3, 1.65, 0.24, "mat_stone_boss"),
            (3.95, 1.98, 0.34, "mat_stone_mizzi"),
            (4.6, 2.30, 0.44, "mat_stone_boss"),
        ):
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                mat_u,
                (px, facade_y, parvis_z + 10.2),
                (1.0, 0.0),
                (0.0, -1.0),
                span_w=span_w_u,
                arch_h=arch_h_u,
                ring_thickness=0.38,
                offset_n0=-0.04,
                offset_n1=n1_u,
                num_voussoirs=11,
                keystone_boost=0.05,
            )
        for u_col in (-2.15, -1.82, 1.82, 2.15):
            collector.add_branch_tube(
                "Buildings_Details",
                "mat_stone_mizzi",
                (px + u_col, facade_y - 0.22, parvis_z + 7.1),
                (px + u_col, facade_y - 0.22, parvis_z + 10.1),
                0.12,
                0.11,
                segments=8,
            )
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (px, facade_y - 0.02),
            parvis_z + 7.2,
            parvis_z + 10.2,
            1.65,
            3.25,
            0.02,
            (1.0, 0.0),
            (0.0, -1.0),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=9,
            add_wall_spandrel=False,
        )

    # Iconic 6-Rung Wooden "Immovable Ladder" resting on the upper cornice below the right window (`x = 200.8, z = parvis_z + 7.0..9.2`)
    for rail_dx in (-0.24, 0.24):
        collector.add_branch_tube(
            "Buildings_Details",
            "mat_bark",
            (200.8 + rail_dx, facade_y - 0.46, parvis_z + 7.02),
            (200.8 + rail_dx, facade_y - 0.08, parvis_z + 9.15),
            0.045,
            0.04,
            segments=6,
        )
    for rung_i in range(6):
        rt = (rung_i + 0.5) / 6.0
        ry_l = (facade_y - 0.46) * (1.0 - rt) + (facade_y - 0.08) * rt
        rz_l = (parvis_z + 7.02) * (1.0 - rt) + (parvis_z + 9.15) * rt
        collector.add_box("Buildings_Details", "mat_bark", (200.8, ry_l, rz_l), (0.52, 0.06, 0.06))

    # 4. Chapel of the Franks (East Side of Southern Facade, `x = 208.0..213.6, y = 173.2..178.8`)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (210.8, 176.0, parvis_z + 4.4), (5.4, 5.6, 8.8), uv_scale=0.25)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (210.8, 176.0, parvis_z + 4.55), (5.8, 6.0, 0.32), uv_scale=0.35)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (210.8, 176.0, parvis_z + 8.75), (5.8, 6.0, 0.36), uv_scale=0.35)
    # Upper arched Crusader loggia windows on South & West faces of the Chapel of the Franks
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        "mat_stone_boss",
        (210.8, 173.2, parvis_z + 6.8),
        (1.0, 0.0),
        (0.0, -1.0),
        span_w=2.4,
        arch_h=1.2,
        ring_thickness=0.36,
        offset_n0=-0.02,
        offset_n1=0.28,
        num_voussoirs=9,
    )
    add_arched_opening_panel(
        collector,
        "Buildings_Openings",
        "mat_wood_doors",
        (210.8, 173.18),
        parvis_z + 4.7,
        parvis_z + 6.8,
        1.15,
        2.25,
        0.02,
        (1.0, 0.0),
        (0.0, -1.0),
        uv_rect=WOOD_DOOR_UV["window_glass"],
        num_voussoirs=9,
        add_wall_spandrel=False,
    )
    collector.add_dome("Buildings_Details", "mat_dome_lead", (210.8, 176.0, parvis_z + 8.85), radius=2.2, height=1.9, rings=5, segments=12)
    # External stone staircase & stepped stone balustrade wall ascending to the Chapel of the Franks
    for st_i in range(12):
        sx_f = 205.8 + st_i * 0.38
        sh_f = (st_i + 1) * 0.36
        collector.add_box(
            "Buildings_Details",
            "mat_stone_ashlar_aged",
            (sx_f, 172.3, parvis_z + sh_f * 0.5),
            (0.42, 1.9, sh_f),
            uv_scale=0.28,
        )
        # Stepped stone parapet wall along the southern edge of the stairs
        collector.add_box(
            "Buildings_Details",
            "mat_stone_mizzi",
            (sx_f, 171.35, parvis_z + sh_f + 0.42),
            (0.42, 0.32, 0.85),
            uv_scale=0.32,
        )
    # Byzantine column stumps & parvis courtyard framing
    for col_x_p in (186.5, 190.2, 203.8):
        collector.add_branch_tube(
            "StreetProps",
            "mat_stone_mizzi",
            (col_x_p, 162.5, parvis_z),
            (col_x_p, 162.5, parvis_z + 1.45),
            0.36,
            0.34,
            segments=10,
        )

    # 5. 12th-Century Crusader Bell Tower (`x = 185.5, y = 180.5`, rising to `parvis_z + 26.5m`)
    bt_x, bt_y = 185.2, 180.2
    collector.add_box("Buildings_Walls", "mat_stone_boss", (bt_x, bt_y, parvis_z + 7.5), (8.6, 8.6, 15.0), uv_scale=0.22)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (bt_x, bt_y, parvis_z + 18.0), (8.0, 8.0, 6.0), uv_scale=0.24)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (bt_x, bt_y, parvis_z + 23.5), (7.4, 7.4, 5.0), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (bt_x, bt_y, parvis_z + 26.2), (7.9, 7.9, 0.45), uv_scale=0.32)
    # Coupled Gothic belfry arches + 3D pointed stone archivolts & central dividing colonnettes on the Bell Tower
    for b_z, y_face in ((parvis_z + 16.5, bt_y - 4.01), (parvis_z + 22.0, bt_y - 3.71)):
        for off_x in (-1.6, 1.6):
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (bt_x + off_x, y_face),
                b_z,
                b_z + 2.2,
                0.95,
                1.80,
                0.01,
                (1.0, 0.0),
                (0.0, -1.0),
                uv_rect=WOOD_DOOR_UV["shutter_louvre"],
                num_voussoirs=7,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (bt_x + off_x, y_face, b_z + 2.2),
                (1.0, 0.0),
                (0.0, -1.0),
                span_w=1.80,
                arch_h=0.95,
                ring_thickness=0.30,
                offset_n0=-0.04,
                offset_n1=0.20,
                num_voussoirs=7,
            )
            collector.add_branch_tube(
                "Buildings_Details",
                "mat_stone_mizzi",
                (bt_x + off_x, y_face - 0.08, b_z),
                (bt_x + off_x, y_face - 0.08, b_z + 2.2),
                0.09,
                0.08,
                segments=6,
            )

    # 6. Great Anastasis Rotunda & Lead Dome (`center = (176.5, 197.5)`, radius = 13.5m, grounded from parvis_z - 2.0m)
    rot_x, rot_y = 176.5, 197.5
    collector.add_branch_tube(
        "Buildings_Walls",
        "mat_stone_ashlar",
        (rot_x, rot_y, parvis_z - 2.0),
        (rot_x, rot_y, parvis_z + 21.5),
        13.6,
        13.4,
        segments=24,
        smooth=True,
        cap_end=True,
    )
    collector.add_dome(
        "Buildings_Details",
        "mat_dome_lead",
        (rot_x, rot_y, parvis_z + 21.4),
        radius=13.4,
        height=10.8,
        rings=8,
        segments=24,
        smooth=True,
    )
    # Golden oculus ring & cross finial atop Anastasis Dome
    collector.add_dome(
        "Buildings_Details",
        "mat_dome_gold",
        (rot_x, rot_y, parvis_z + 31.6),
        radius=3.2,
        height=1.2,
        rings=4,
        segments=16,
    )
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_dome_gold",
        (rot_x, rot_y, parvis_z + 32.5),
        (rot_x, rot_y, parvis_z + 35.2),
        0.12,
        0.08,
        segments=6,
    )
    collector.add_box("Buildings_Details", "mat_dome_gold", (rot_x, rot_y, parvis_z + 34.3), (1.4, 0.14, 0.14))

    # 7. Catholicon Crusader Choir Dome (`center = (202.0, 197.5)`, radius = 7.8m)
    cath_x, cath_y = 202.0, 197.5
    collector.add_branch_tube(
        "Buildings_Walls",
        "mat_stone_ashlar",
        (cath_x, cath_y, parvis_z + 14.0),
        (cath_x, cath_y, parvis_z + 20.2),
        7.8,
        7.6,
        segments=16,
        smooth=False,
        cap_end=True,
    )
    collector.add_dome(
        "Buildings_Details",
        "mat_dome_lead",
        (cath_x, cath_y, parvis_z + 20.1),
        radius=7.6,
        height=6.8,
        rings=7,
        segments=16,
        smooth=True,
    )
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_dome_gold",
        (cath_x, cath_y, parvis_z + 26.8),
        (cath_x, cath_y, parvis_z + 29.0),
        0.10,
        0.07,
        segments=6,
    )
    collector.add_box("Buildings_Details", "mat_dome_gold", (cath_x, cath_y, parvis_z + 28.2), (1.1, 0.12, 0.12))

    # 8. Mosque of Omar & Square Mamluk Minaret across the Southern Parvis (`x = 189.5, y = 143.0`)
    omar_z = street_elevation_z(189.5, 143.0)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar_aged", (189.5, 143.0, omar_z + 4.8), (16.0, 12.0, 9.6), uv_scale=0.25)
    # Tall square Mamluk Minaret of Mosque of Omar (`26m` tall with corbelled gallery & dome)
    min_x, min_y = 184.5, 147.0
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (min_x, min_y, omar_z + 11.5), (3.8, 3.8, 23.0), uv_scale=0.25)
    collector.add_box("Buildings_Details", "mat_stone_boss", (min_x, min_y, omar_z + 23.2), (4.6, 4.6, 0.85), uv_scale=0.28)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (min_x, min_y, omar_z + 25.0), (3.0, 3.0, 2.8), uv_scale=0.28)
    collector.add_dome("Buildings_Details", "mat_dome_lead", (min_x, min_y, omar_z + 26.4), radius=1.6, height=1.8, rings=5, segments=12)


def _build_redeemer_and_muristan(collector: MeshCollector, sign_atlas: Dict[str, dict]) -> None:
    """
    Builds:
      1. Lutheran Church of the Redeemer (`x = 266..292, y = 128..156`) with its 38m white stone Bell Tower
      2. Muristan Fountain & Octagonal Market Plaza (`Suq Aftimos` at `(232.0, 128.0)`)
    """
    red_x, red_y = 278.5, 141.5
    red_z = street_elevation_z(red_x, red_y)

    # 1. Basilica Nave of the Lutheran Church of the Redeemer (`mat_stone_mizzi` crisp white limestone)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (red_x, red_y, red_z + 7.2), (24.0, 18.5, 14.4), uv_scale=0.24)
    # West neo-Romanesque Gable Facade (`x = 266.8`) adjoining the Bell Tower + West Portal & Circular Rose Window
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (266.8, red_y - 4.2, red_z + 8.6), (4.2, 11.5, 17.2), uv_scale=0.24)
    add_arched_opening_panel(
        collector,
        "Buildings_Openings",
        "mat_wood_doors",
        (264.69, red_y - 4.8),
        red_z,
        red_z + 3.2,
        1.30,
        2.60,
        0.01,
        (0.0, -1.0),
        (-1.0, 0.0),
        uv_rect=WOOD_DOOR_UV["door_wood"],
        num_voussoirs=9,
        add_wall_spandrel=False,
    )
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        "mat_stone_boss",
        (264.69, red_y - 4.8, red_z + 3.2),
        (0.0, -1.0),
        (-1.0, 0.0),
        span_w=2.60,
        arch_h=1.30,
        ring_thickness=0.38,
        offset_n0=-0.04,
        offset_n1=0.24,
        num_voussoirs=9,
    )
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_stone_boss",
        (264.7, red_y - 4.8, red_z + 10.6),
        (264.48, red_y - 4.8, red_z + 10.6),
        1.45,
        1.38,
        segments=14,
    )
    collector.add_branch_tube(
        "Buildings_Openings",
        "mat_iron_dark",
        (264.65, red_y - 4.8, red_z + 10.6),
        (264.54, red_y - 4.8, red_z + 10.6),
        1.12,
        1.08,
        segments=14,
        cap_end=True,
    )
    # Pitched terracotta gable roof (trimmed to `x = 268.2..290.5` so it never pokes out past the West Bell Tower!)
    collector.add_box("Buildings_Details", "mat_roof_tile", (red_x + 1.1, red_y, red_z + 15.2), (22.2, 17.0, 1.8), uv_scale=0.30)
    # Soaring 38m White Bell Tower at the West front (`(266.8, 141.5)`)
    tw_x, tw_y = 266.8, 141.5
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (tw_x, tw_y, red_z + 14.0), (6.6, 6.6, 28.0), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_stone_boss", (tw_x, tw_y, red_z + 28.2), (7.2, 7.2, 0.65), uv_scale=0.28)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (tw_x, tw_y, red_z + 31.2), (5.8, 5.8, 5.4), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (tw_x, tw_y, red_z + 34.1), (6.4, 6.4, 0.55), uv_scale=0.30)
    # Iconic pyramidal/octagonal stone & lead spire (`r0=3.2 -> r1=0.16`) + golden cross
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_dome_lead",
        (tw_x, tw_y, red_z + 34.3),
        (tw_x, tw_y, red_z + 40.5),
        3.25,
        0.16,
        segments=8,
        smooth=False,
        cap_end=True,
    )
    collector.add_box("Buildings_Details", "mat_dome_gold", (tw_x, tw_y, red_z + 41.3), (0.18, 0.18, 1.6), uv_scale=0.5)
    collector.add_box("Buildings_Details", "mat_dome_gold", (tw_x, tw_y, red_z + 41.6), (0.95, 0.18, 0.18), uv_scale=0.5)

    # Two tiers of belfry arched windows on all 4 faces of the Redeemer Bell Tower
    for tier_z in (red_z + 23.0, red_z + 29.2):
        for face_ang in (0.0, 90.0, 180.0, 270.0):
            rad = math.radians(face_ang)
            nx, ny = math.cos(rad), math.sin(rad)
            tx, ty = -ny, nx
            dist_n = 2.92 if tier_z > red_z + 26.0 else 3.32
            for off_t in (-1.2, 1.2):
                wx = tw_x + nx * dist_n + tx * off_t
                wy = tw_y + ny * dist_n + ty * off_t
                add_arched_opening_panel(
                    collector,
                    "Buildings_Openings",
                    "mat_wood_doors",
                    (wx, wy),
                    tier_z,
                    tier_z + 2.4,
                    0.85,
                    0.95,
                    0.02,
                    (tx, ty),
                    (nx, ny),
                    uv_rect=WOOD_DOOR_UV["shutter_louvre"],
                    num_voussoirs=7,
                    add_wall_spandrel=False,
                )

    # 2. Muristan Fountain & Market Plaza (`Suq Aftimos` at `(232.0, 128.0)`)
    mur_x, mur_y = 232.0, 128.0
    mur_z = street_elevation_z(mur_x, mur_y)
    # Octagonal stepped fountain basin, cobalt-blue water surface, 4 stone colonnettes, octagonal canopy & bronze finial
    collector.add_branch_tube(
        "StreetProps",
        "mat_stone_mizzi",
        (mur_x, mur_y, mur_z),
        (mur_x, mur_y, mur_z + 0.55),
        4.2,
        4.0,
        segments=8,
        smooth=False,
        cap_end=True,
    )
    collector.add_branch_tube(
        "StreetProps",
        "mat_stone_ashlar",
        (mur_x, mur_y, mur_z + 0.55),
        (mur_x, mur_y, mur_z + 1.35),
        2.7,
        2.55,
        segments=8,
        smooth=False,
        cap_end=True,
    )
    # Shimmering blue water surface inside the octagonal fountain basin
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_tile_iznik",
        (mur_x, mur_y, mur_z + 1.25),
        (mur_x, mur_y, mur_z + 1.42),
        2.25,
        2.22,
        segments=8,
        smooth=False,
        cap_end=True,
    )
    # Low wrought-iron octagonal perimeter fence around the stepped platform
    for f_i in range(8):
        fa = math.radians(f_i * 45.0 + 22.5)
        collector.add_box(
            "StreetProps",
            "mat_iron_dark",
            (mur_x + math.cos(fa) * 3.65, mur_y + math.sin(fa) * 3.65, mur_z + 0.88),
            (0.07, 0.07, 0.68),
        )
    collector.add_branch_tube(
        "StreetProps",
        "mat_iron_dark",
        (mur_x, mur_y, mur_z + 1.15),
        (mur_x, mur_y, mur_z + 1.21),
        3.68,
        3.68,
        segments=8,
        smooth=False,
    )
    # Central tiered stone water column + 4 slender stone colonnettes supporting an octagonal stone canopy
    collector.add_branch_tube(
        "StreetProps",
        "mat_stone_mizzi",
        (mur_x, mur_y, mur_z + 1.35),
        (mur_x, mur_y, mur_z + 3.65),
        0.55,
        0.40,
        segments=8,
        smooth=True,
        cap_end=True,
    )
    for col_a in (45.0, 135.0, 225.0, 315.0):
        crad = math.radians(col_a)
        mcx = mur_x + math.cos(crad) * 1.45
        mcy = mur_y + math.sin(crad) * 1.45
        collector.add_branch_tube(
            "StreetProps",
            "mat_stone_mizzi",
            (mcx, mcy, mur_z + 1.35),
            (mcx, mcy, mur_z + 3.75),
            0.16,
            0.14,
            segments=8,
            cap_end=True,
        )
    # Upper octagonal stone canopy + gilded bronze finial
    collector.add_branch_tube(
        "StreetProps",
        "mat_stone_mizzi",
        (mur_x, mur_y, mur_z + 3.68),
        (mur_x, mur_y, mur_z + 4.35),
        1.85,
        1.25,
        segments=8,
        smooth=False,
        cap_end=True,
    )
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_dome_gold",
        (mur_x, mur_y, mur_z + 4.30),
        (mur_x, mur_y, mur_z + 5.45),
        0.34,
        0.06,
        segments=8,
        smooth=True,
        cap_end=True,
    )
    # Outdoor café parasols & stone planter beds flanking the NW and SE sides of the Muristan plaza
    for umb_a in (110.0, 290.0):
        urad = math.radians(umb_a)
        ux = mur_x + math.cos(urad) * 8.2
        uy = mur_y + math.sin(urad) * 8.2
        uz = street_elevation_z(ux, uy)
        collector.add_branch_tube("StreetProps", "mat_iron_dark", (ux, uy, uz), (ux, uy, uz + 2.55), 0.05, 0.04, segments=6)
        collector.add_branch_tube("StreetProps", "mat_stone_mizzi", (ux, uy, uz + 2.3), (ux, uy, uz + 2.85), 1.65, 0.10, segments=8, cap_end=True)

    # Olive trees and street lamps framing the Muristan octagon
    for ang_deg in (45.0, 135.0, 225.0, 315.0):
        rad = math.radians(ang_deg)
        tx = mur_x + math.cos(rad) * 9.5
        ty = mur_y + math.sin(rad) * 9.5
        tz = street_elevation_z(tx, ty)
        build_olive_tree(collector, tx, ty, tz, height=5.8, seed=int(ang_deg), with_kerb=True)
        lx = mur_x + math.cos(rad + 0.4) * 6.5
        ly = mur_y + math.sin(rad + 0.4) * 6.5
        build_street_lamp(collector, lx, ly, street_elevation_z(lx, ly), height=4.2)


def build_region_r02_christian_quarter(
    collector: MeshCollector,
    osm_data: dict,
    sign_atlas: Dict[str, dict],
) -> int:
    """
    Builds Region 2 (`r02_christian_quarter`):
      - Continuous stone-paved ground surface across `[-195..295] x [55..560]`
      - NW Ottoman Old City Wall from Jaffa Gate N end (`(-21.5, 38.5)`) through Goliath's Tower (`(-172.5, 288.5)`)
        and New Gate (`(-141.1, 297.9)`) to Damascus Gate approach (`(238.1, 541.9)`)
      - New Gate (`Bab al-Jadid`) monumental portal at `(-141.1, 297.9)`
      - Church of the Holy Sepulchre, Crusader Bell Tower, Anastasis & Catholicon Domes, Southern Parvis
      - Lutheran Church of the Redeemer & Tall White Bell Tower + Muristan Fountain Plaza
      - All Christian Quarter OSM buildings inside the Old City Walls + trees & street props
    """
    # 1. Ground paving across Christian Quarter inside and around the NW City Wall
    def r02_exclude(x: float, y: float) -> bool:
        return not is_inside_old_city_walls(x, y, margin=35.0)

    build_region_ground_grid(collector, -195.0, 295.0, 55.0, 560.0, step=6.0, exclude_fn=r02_exclude)

    # 2. NW Ottoman Old City Wall & New Gate (`Bab al-Jadid` at `(-141.1, 297.9)`)
    nw_wall_pts: List[Vec2] = [
        (-21.5, 38.5),
        (-170.0, 160.9),
        (-167.2, 169.5),
        (-175.5, 173.3),
        (-174.1, 187.6),
        (-188.3, 199.4),
        (-165.8, 226.9),
        (-145.7, 254.7),
        (-172.5, 288.5),
        (-158.2, 298.3),
        (-152.5, 290.0),
        (-149.0, 292.4),
        (-141.1, 297.9),  # New Gate
        (-137.3, 300.0),
        (-129.4, 305.0),
        (-119.9, 309.0),
        (-80.4, 335.6),
        (-75.7, 330.8),
        (-21.0, 370.7),
        (6.6, 337.3),
        (37.6, 365.8),
        (34.9, 370.0),
        (79.2, 408.0),
        (171.9, 483.5),
        (172.2, 504.7),
        (209.8, 517.3),
        (238.1, 541.9),
    ]
    new_gate_xy = (-141.1, 297.9)
    build_ottoman_wall_polyline(
        collector,
        nw_wall_pts,
        wall_height=12.4,
        thickness=3.6,
        gate_gaps=[(new_gate_xy[0], new_gate_xy[1], 8.5)],
        bastion_every=3,
    )
    sign_new_gate = sign_atlas.get("latin_rd_plaque")
    sign_rect_ng = (
        (float(sign_new_gate["u0"]), float(sign_new_gate["v0"]), float(sign_new_gate["u1"]), float(sign_new_gate["v1"]))
        if sign_new_gate
        else None
    )
    build_ottoman_gatehouse(
        collector,
        new_gate_xy,
        heading_inward_deg=145.0,
        gate_name="New Gate (Bab al-Jadid)",
        width=16.2,
        depth=9.2,
        height=13.8,
        twin_towers=False,
        sign_rect=sign_rect_ng,
    )

    # 3. Hero Landmarks: Church of the Holy Sepulchre, Redeemer, Muristan Fountain & Christian Quarter Lanes
    _build_church_of_the_holy_sepulchre(collector, sign_atlas)
    _build_redeemer_and_muristan(collector, sign_atlas)

    # Christian Quarter Road (`x = 132.0, y = 78..155`), Greek Patriarchate Lane (`y = 185.0, x = 85..155`),
    # St. Francis Street (`(-25, 255) -> (32, 198)`), and Muristan North Market Arch (`(235.0, 158.0)`)
    _build_souq_street_corridor(collector, (132.0, 78.0), (132.0, 152.0), street_w=6.2, wall_h=8.6, with_arches=True, with_shops=True, seed=201)
    _build_souq_street_corridor(collector, (82.0, 185.0), (154.0, 185.0), street_w=5.8, wall_h=8.4, with_arches=True, with_shops=True, seed=202)
    _build_souq_street_corridor(collector, (-20.0, 250.0), (35.0, 195.0), street_w=6.2, wall_h=8.8, with_arches=True, with_shops=False, seed=203)

    # Muristan North Market Gateway Arch & Flanking 2-Storey Corner Arcade Wings (`Suq Aftimos` entrance at `(235.0, 154.0)`)
    ma_x, ma_y = 235.0, 154.0
    ma_z = street_elevation_z(ma_x, ma_y)
    for sx_off in (-3.4, 3.4):
        collector.add_box("Buildings_Walls", "mat_stone_boss", (ma_x + sx_off, ma_y, ma_z + 3.6), (1.8, 1.8, 7.2), uv_scale=0.25)
    # Flanking 2-storey dressed-stone corner market buildings framing the Muristan gateway arch
    for side_m in (-1.0, 1.0):
        wx_m = ma_x + side_m * 8.2
        collector.add_box("Buildings_Walls", "mat_stone_mizzi", (wx_m, ma_y - 2.0, ma_z + 4.1), (7.8, 5.8, 8.2), uv_scale=0.24)
        collector.add_box("Buildings_Details", "mat_stone_boss", (wx_m, ma_y - 2.0, ma_z + 8.35), (8.2, 6.2, 0.38), uv_scale=0.30)
        for bay_dx in (-1.9, 1.9):
            bx_m = wx_m + bay_dx
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (bx_m, ma_y + 0.92),
                ma_z,
                ma_z + 2.1,
                1.05,
                2.25,
                0.01,
                (-1.0, 0.0),
                (0.0, 1.0),
                uv_rect=WOOD_DOOR_UV["souq_open"],
                num_voussoirs=7,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (bx_m, ma_y + 0.92, ma_z + 2.1),
                (-1.0, 0.0),
                (0.0, 1.0),
                span_w=2.25,
                arch_h=1.05,
                ring_thickness=0.32,
                offset_n0=-0.04,
                offset_n1=0.20,
                num_voussoirs=7,
            )
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (bx_m, ma_y + 0.92),
                ma_z + 4.5,
                ma_z + 6.0,
                0.62,
                1.25,
                0.01,
                (-1.0, 0.0),
                (0.0, 1.0),
                uv_rect=WOOD_DOOR_UV["window_glass"],
                num_voussoirs=5,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (bx_m, ma_y + 0.92, ma_z + 6.0),
                (-1.0, 0.0),
                (0.0, 1.0),
                span_w=1.25,
                arch_h=0.62,
                ring_thickness=0.25,
                offset_n0=-0.04,
                offset_n1=0.18,
                num_voussoirs=5,
            )
        collector.add_box(
            "StreetProps",
            "mat_awning_red" if side_m < 0 else "mat_awning_green",
            (wx_m, ma_y + 1.45, ma_z + 3.45),
            (6.8, 0.95, 0.14),
            uv_scale=0.45,
            bottom_face=True,
        )
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        "mat_stone_boss",
        (ma_x, ma_y + 0.85, ma_z + 3.6),
        (-1.0, 0.0),
        (0.0, 1.0),
        span_w=5.2,
        arch_h=2.1,
        ring_thickness=0.42,
        offset_n0=-1.7,
        offset_n1=0.15,
        num_voussoirs=11,
    )
    add_arch_spandrel_wall(
        collector,
        "Buildings_Walls",
        "mat_stone_mizzi",
        (ma_x, ma_y, ma_z + 3.6),
        (1.0, 0.0),
        (0.0, 1.0),
        span_w=5.3,
        arch_h=2.1,
        z_top=ma_z + 6.8,
        offset_n0=-0.85,
        offset_n1=0.85,
        num_voussoirs=10,
        emit_soffit=True,
    )
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (ma_x, ma_y, ma_z + 7.0), (8.8, 2.1, 0.45), uv_scale=0.30, bottom_face=True)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (ma_x, ma_y + 0.82, ma_z + 7.65), (4.2, 0.45, 0.85), uv_scale=0.30)
    collector.add_box("StreetProps", "mat_lantern_glow", (ma_x, ma_y, ma_z + 5.25), (0.30, 0.30, 0.42), bottom_face=True)

    # 4. Christian Quarter OSM Buildings (filtered to inside Old City Walls & non-overlapping with hero sites/corridors)
    clear_scene_building_rings()
    osm_buildings = [
        b for b in osm_data.get("buildings", [])
        if b.get("region") == "r02_christian_quarter"
    ]

    r02_corridors = [
        [(132.0, 65.0), (132.0, 215.0)],
        [(70.0, 185.0), (165.0, 185.0)],
        [(-35.0, 265.0), (45.0, 185.0)],
    ]

    valid_blds = []
    for b in osm_buildings:
        osm_id = int(b.get("osm_id", 0))
        if osm_id in (5862586, 7880713, 126086870, 158063844, 158063849, 1022672085, 1022663529, 29268902, 690881495, 690894839, 290821793):
            continue
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        if not is_inside_old_city_walls(cx, cy, margin=-4.0):
            continue
        ring = b.get("ring", [])
        if len(ring) < 3:
            continue
        rxs = [float(p[0]) for p in ring]
        rys = [float(p[1]) for p in ring]
        if (max(rxs) - min(rxs)) > 68.0 or (max(rys) - min(rys)) > 68.0:
            continue
        # Keep Holy Sepulchre + Southern Parvis (`160..224, 148..222`) clear
        if _ring_overlaps_rect(ring, 160.0, 148.0, 224.0, 222.0):
            continue
        # Keep Redeemer + Muristan Fountain Plaza + North Arch (`214..294, 112..178`) clear
        if _ring_overlaps_rect(ring, 214.0, 112.0, 294.0, 178.0):
            continue
        # Keep New Gate portal road (`-154..-122, 280..314`) clear
        if _ring_overlaps_rect(ring, -154.0, 280.0, -122.0, 314.0):
            continue
        # Keep Christian Quarter Road, Greek Patriarchate Lane, and St. Francis Street corridors clear
        if any(_ring_overlaps_polyline(ring, corr, 8.5) for corr in r02_corridors):
            continue
        valid_blds.append(b)
        register_scene_building_ring(osm_id, ring)

    built_count = 0
    for idx, b in enumerate(valid_blds):
        ring = b.get("ring", [])
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        bz = street_elevation_z(cx, cy)
        lvls = max(2, min(4, int(b.get("levels", 2))))
        build_building_from_footprint(
            collector,
            ring,
            base_z=bz,
            levels=lvls,
            sign_atlas=sign_atlas,
            building_id=int(b.get("osm_id", idx)),
            is_hero=False,
        )
        built_count += 1

    # 5. Trees & Street Lamps along Christian Quarter lanes, Parvis, and New Gate
    tree_spots = [
        (-132.0, 286.0, "cypress", 10.5, 301),
        (-146.0, 282.0, "olive", 6.4, 302),
        (32.0, 145.0, "cypress", 11.0, 303),
        (45.0, 158.0, "olive", 6.5, 304),
        (-22.0, 260.0, "cypress", 10.8, 305),
        (88.0, 375.0, "olive", 6.8, 306),
        (182.0, 156.0, "olive", 6.2, 307),
        (260.0, 134.0, "cypress", 11.2, 308),
    ]
    for tx, ty, sp, th, seed in tree_spots:
        tz = street_elevation_z(tx, ty)
        if sp == "cypress":
            build_cypress_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)
        else:
            build_olive_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)

    lamp_spots = [
        (-138.0, 290.0),
        (-125.0, 248.0),
        (24.5, 78.0),
        (24.5, 115.0),
        (132.0, 110.0),
        (132.0, 155.0),
        (184.0, 162.0),
        (208.0, 162.0),
        (258.0, 140.0),
    ]
    for lx, ly in lamp_spots:
        build_street_lamp(collector, lx, ly, street_elevation_z(lx, ly), height=4.3)

    return built_count


# ==============================================================================
# 4. REGION 3: CENTRAL SOUQS, TRIPLE SOUQ, NORTHERN CARDO & STREET OF THE CHAIN (`r03_central_souqs`)
# ==============================================================================

def _build_vaulted_triple_souq_and_chain_street(collector: MeshCollector, sign_atlas: Dict[str, dict]) -> None:
    """
    Builds:
      1. The Crusader Vaulted Triple Souq (`Suq el-Attarin`, `Suq el-Lahamin`, `Suq el-Khawajat`)
         & Northern Cardo (`x = 296..330, y = -40..165`) with arched stone bazaar stalls,
         high vaulted stone ceiling bays with daylight skylights, hanging lanterns, and spice displays.
      2. Lower David Street & Street of the Chain (`Tariq Bab al-Silsila`, `x = 168..602, y = 2..68`)
         with flying-buttress arches, Mamluk Ablaq Turba mausoleums (striped red/cream masonry & domes),
         and an Ottoman Sabil fountain.
    """
    # 1. Crusader Vaulted Triple Souq (`x_lanes = [302.0, 313.0, 324.0]`, `y = -36..156`)
    pier_xs = [298.5, 305.5, 309.5, 316.5, 320.5, 327.5]
    y_Segs = list(range(-36, 157, 12))

    for j in range(len(y_Segs) - 1):
        y0, y1 = float(y_Segs[j]), float(y_Segs[j + 1])
        ym = 0.5 * (y0 + y1)
        # Leave cross-street intersection open only where David Street / Street of the Chain crosses (`y in [-2..14]`)
        if -2.0 <= ym < 14.0:
            continue
        bz = street_elevation_z(313.0, ym)

        # Build the 3 parallel market lane walls & arched shopfronts
        for lane_idx, lx in enumerate((302.0, 313.0, 324.0)):
            w_left = lx - 2.8
            w_right = lx + 2.8
            # Side walls (2 storeys, 7.8m tall)
            collector.add_box("Buildings_Walls", "mat_stone_ashlar_aged", (w_left - 1.2, ym, bz + 3.9), (2.4, 11.8, 7.8), uv_scale=0.25)
            collector.add_box("Buildings_Walls", "mat_stone_ashlar_aged", (w_right + 1.2, ym, bz + 3.9), (2.4, 11.8, 7.8), uv_scale=0.25)

            # Open bazaar shopfront arches on both sides of the souq lane
            for side_x, nx_dir in ((w_left + 0.04, 1.0), (w_right - 0.04, -1.0)):
                for bay_off in (-3.0, 3.0):
                    by = ym + bay_off
                    add_arched_opening_panel(
                        collector,
                        "Buildings_Openings",
                        "mat_wood_doors",
                        (side_x, by),
                        bz,
                        bz + 2.0,
                        1.1,
                        2.6,
                        0.02,
                        (0.0, nx_dir),
                        (nx_dir, 0.0),
                        uv_rect=WOOD_DOOR_UV["souq_open"],
                        num_voussoirs=7,
                        add_wall_spandrel=False,
                    )
                    add_arch_voussoirs(
                        collector,
                        "Buildings_Details",
                        "mat_stone_boss",
                        (side_x, by, bz + 2.0),
                        (0.0, nx_dir),
                        (nx_dir, 0.0),
                        span_w=2.7,
                        arch_h=1.15,
                        ring_thickness=0.32,
                        offset_n0=-0.02,
                        offset_n1=0.18,
                        num_voussoirs=7,
                    )
                    # Hanging colorful carpets/textiles & upper stone arched niche above each stall
                    rug_mat = "mat_awning_red" if (j + lane_idx + int(bay_off)) % 2 == 0 else "mat_awning_striped"
                    collector.add_box(
                        "StreetProps",
                        rug_mat,
                        (side_x + nx_dir * 0.22, by + 1.45, bz + 2.35),
                        (0.08, 0.95, 1.65),
                        uv_scale=0.5,
                    )

            # High Crusader vaulted stone roof above every bay, with a narrow central skylight shaft (`1.2m` wide)
            for v_side in (-1.0, 1.0):
                collector.add_box(
                    "Buildings_Walls",
                    "mat_stone_ashlar",
                    (lx + v_side * 1.85, ym, bz + 8.05),
                    (2.5, 11.8, 0.65),
                    uv_scale=0.26,
                    bottom_face=True,
                )
            # Transverse pointed Crusader stone vault arch ribs every 6m + hanging iron chain & bazaar lantern
            for rib_off in (-3.0, 3.0):
                add_arch_voussoirs(
                    collector,
                    "Buildings_Details",
                    "mat_stone_boss",
                    (lx, ym + rib_off, bz + 5.2),
                    (1.0, 0.0),
                    (0.0, 1.0),
                    span_w=5.5,
                    arch_h=2.45,
                    ring_thickness=0.45,
                    offset_n0=-0.35,
                    offset_n1=0.35,
                    num_voussoirs=11,
                )
            collector.add_box(
                "StreetProps",
                "mat_iron_dark",
                (lx, ym, bz + 7.45),
                (0.05, 0.05, 1.3),
            )
            collector.add_box(
                "StreetProps",
                "mat_lantern_glow",
                (lx, ym, bz + 6.6),
                (0.32, 0.32, 0.46),
                bottom_face=True,
            )

            # Spice sacks & produce crates along both sides of the souq lane
            build_souq_display(collector, (w_left + 0.58, ym - 1.8, bz), (0.0, 1.0), (1.0, 0.0), seed=500 + j * 10 + lane_idx)
            if j % 2 == 1:
                build_souq_display(collector, (w_right - 0.58, ym + 1.8, bz), (0.0, -1.0), (-1.0, 0.0), seed=600 + j * 10 + lane_idx)

    # 1b. David Street / Street of the Chain / Triple Souq Crossroads Vaulted Arcade (`x = 297.0..329.0, y = 0.0..12.0`)
    #     Caps the North (`y = 12.0`) and South (`y = 0.0`) Triple Souq cut ends with 2-storey Crusader stone facades,
    #     pointed gateway arches, upper shuttered windows, Mashrabiyas, transverse vault ribs, and bazaar stalls!
    junc_z = street_elevation_z(313.0, 6.0)
    for y_face, fny, ftx in ((0.0, 1.0, 1.0), (12.0, -1.0, -1.0)):
        # Continuous upper-storey masonry wall (`z = junc_z + 5.2 .. 9.1m`) & projecting stone roof cornice across `x = 297.0..329.0`
        collector.add_box(
            "Buildings_Walls",
            "mat_stone_ashlar",
            (313.0, y_face - fny * 0.65, junc_z + 7.15),
            (32.4, 1.3, 3.9),
            uv_scale=0.24,
            bottom_face=True,
        )
        collector.add_box(
            "Buildings_Details",
            "mat_stone_boss",
            (313.0, y_face - fny * 0.55, junc_z + 9.25),
            (32.8, 1.6, 0.38),
            uv_scale=0.30,
            bottom_face=True,
        )
        collector.add_box(
            "Buildings_Details",
            "mat_stone_mizzi",
            (313.0, y_face + fny * 0.08, junc_z + 5.25),
            (32.6, 0.24, 0.24),
            uv_scale=0.32,
            bottom_face=True,
        )
        # 3 Pointed Crusader stone gateway arches opening into `Suq el-Attarin`, `Suq el-Lahamin` & `Suq el-Khawajat`
        for lane_i, lx in enumerate((302.0, 313.0, 324.0)):
            add_arch_spandrel_wall(
                collector,
                "Buildings_Walls",
                "mat_stone_ashlar_aged",
                (lx, y_face, junc_z + 3.1),
                (ftx, 0.0),
                (0.0, fny),
                span_w=5.5,
                arch_h=2.1,
                z_top=junc_z + 5.3,
                offset_n0=-1.2,
                offset_n1=0.05,
                num_voussoirs=9,
                emit_soffit=True,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (lx, y_face + fny * 0.05, junc_z + 3.1),
                (ftx, 0.0),
                (0.0, fny),
                span_w=5.5,
                arch_h=2.1,
                ring_thickness=0.38,
                offset_n0=-0.10,
                offset_n1=0.22,
                num_voussoirs=9,
            )
            # Upper-floor arched window with stone sill & open green louvered shutters above each souq archway
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (lx, y_face + fny * 0.02),
                junc_z + 5.65,
                junc_z + 7.15,
                0.68,
                1.36,
                0.01,
                (ftx, 0.0),
                (0.0, fny),
                uv_rect=WOOD_DOOR_UV["window_glass"],
                num_voussoirs=5,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (lx, y_face + fny * 0.02, junc_z + 7.15),
                (ftx, 0.0),
                (0.0, fny),
                span_w=1.36,
                arch_h=0.68,
                ring_thickness=0.26,
                offset_n0=-0.04,
                offset_n1=0.18,
                num_voussoirs=5,
            )
            collector.add_box(
                "Buildings_Details",
                "mat_stone_mizzi",
                (lx, y_face + fny * 0.12, junc_z + 5.56),
                (1.62, 0.24, 0.16),
                uv_scale=0.35,
            )
            for sh_x in (-0.88, 0.88):
                collector.add_box(
                    "Buildings_Openings",
                    "mat_wood_doors",
                    (lx + sh_x, y_face + fny * 0.08, junc_z + 6.45),
                    (0.36, 0.08, 1.52),
                    uv_rect=WOOD_DOOR_UV["shutter_louvre"],
                )
        # Projecting wooden Mashrabiya bay windows, framed street plaques & active spice stalls on the intermediate piers (`x = 307.5, 318.5`)
        for p_idx, px_j in enumerate((307.5, 318.5)):
            collector.add_box(
                "Buildings_Details",
                "mat_bark",
                (px_j, y_face + fny * 0.38, junc_z + 6.65),
                (1.65, 0.72, 1.95),
                uv_scale=0.35,
                bottom_face=True,
            )
            collector.add_box(
                "Buildings_Details",
                "mat_stone_mizzi",
                (px_j, y_face + fny * 0.06, junc_z + 3.45),
                (0.62, 0.08, 0.38),
                uv_scale=0.45,
            )
            collector.add_box(
                "Buildings_Details",
                "mat_tile_iznik",
                (px_j, y_face + fny * 0.09, junc_z + 3.45),
                (0.50, 0.06, 0.28),
                uv_scale=0.5,
            )
            build_souq_display(
                collector,
                (px_j, y_face + fny * 0.85, junc_z),
                (ftx, 0.0),
                (0.0, fny),
                width=2.1,
                seed=580 + p_idx * 7 + (0 if fny > 0 else 3),
            )
    # Curved Crusader transverse stone flying-buttress arches spanning across the crossroads (`x = 298.5, 307.5, 318.5, 327.5`)
    for rx_j in (298.5, 307.5, 318.5, 327.5):
        add_arch_spandrel_wall(
            collector,
            "Buildings_Walls",
            "mat_stone_boss",
            (rx_j, 6.0, junc_z + 5.2),
            (0.0, 1.0),
            (1.0, 0.0),
            span_w=11.4,
            arch_h=2.5,
            z_top=junc_z + 8.6,
            offset_n0=-0.42,
            offset_n1=0.42,
            num_voussoirs=11,
            emit_soffit=True,
        )
        for nx_r in (-1.0, 1.0):
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_mizzi",
                (rx_j + nx_r * 0.42, 6.0, junc_z + 5.2),
                (0.0, nx_r),
                (nx_r, 0.0),
                span_w=11.4,
                arch_h=2.5,
                ring_thickness=0.36,
                offset_n0=-0.04,
                offset_n1=0.16,
                num_voussoirs=11,
            )
        collector.add_box("StreetProps", "mat_iron_dark", (rx_j, 6.0, junc_z + 7.15), (0.05, 0.05, 1.15))
        collector.add_box("StreetProps", "mat_lantern_glow", (rx_j, 6.0, junc_z + 6.45), (0.32, 0.32, 0.44), bottom_face=True)

    # 2. Street of the Chain (`Tariq Bab al-Silsila`) Pointed Buttress Arches & Mamluk Ablaq Turbas
    # Path runs from `(175, 4)` -> `(300, 8)` -> `(440, 36)` -> `(595, 68)`
    chain_waypoints = [
        (185.0, 4.0),
        (215.0, 5.0),
        (248.0, 6.0),
        (280.0, 7.5),
        (345.0, 14.0),
        (385.0, 22.0),
        (425.0, 31.0),
        (465.0, 40.0),
        (505.0, 49.0),
        (545.0, 58.0),
        (585.0, 66.0),
    ]
    for idx_c, (cx, cy) in enumerate(chain_waypoints):
        cz = street_elevation_z(cx, cy)
        # Grounded stone arch piers on both sides of the 5.8m street corridor (`y = cy - 3.2` and `y = cy + 3.2`)
        for side_y in (-3.2, 3.2):
            collector.add_box(
                "Buildings_Walls",
                "mat_stone_boss",
                (cx, cy + side_y, cz + 3.2),
                (1.35, 1.15, 6.4),
                uv_scale=0.26,
            )
        # 3D pointed Crusader/Mamluk stone flying-buttress arch & spandrel wall spanning the street
        add_arch_spandrel_wall(
            collector,
            "Buildings_Walls",
            "mat_stone_boss",
            (cx, cy, cz + 4.2),
            (0.0, 1.0),
            (1.0, 0.0),
            span_w=5.8,
            arch_h=1.75,
            z_top=cz + 6.4,
            offset_n0=-0.52,
            offset_n1=0.52,
            num_voussoirs=9,
            emit_soffit=True,
        )
        for nx_c in (-1.0, 1.0):
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_mizzi",
                (cx + nx_c * 0.52, cy, cz + 4.2),
                (0.0, nx_c),
                (nx_c, 0.0),
                span_w=5.8,
                arch_h=1.75,
                ring_thickness=0.34,
                offset_n0=-0.04,
                offset_n1=0.16,
                num_voussoirs=9,
            )
        collector.add_box(
            "Buildings_Details",
            "mat_stone_mizzi",
            (cx, cy, cz + 6.52),
            (1.28, 7.6, 0.24),
            uv_scale=0.32,
            bottom_face=True,
        )
        build_street_lamp(collector, cx + 2.2, cy + 2.5, cz, height=4.2)

    # 3. Mamluk *Turbas* & *Madrasas* on Street of the Chain (`Turbat Baraka Khan` at `(445, 46)`, `Madrasa al-Tashtamuriyya` at `(525, 64)`)
    for t_idx, (tx, ty) in enumerate(((445.0, 46.0), (525.0, 64.0))):
        tz = street_elevation_z(tx, ty)
        # Main masonry block grounded 2.0m into the sloped Street of the Chain (`tz - 2.0 .. tz + 10.4`) so there is zero base shadow gap!
        collector.add_box("Buildings_Walls", "mat_stone_ashlar", (tx, ty, tz + 4.2), (16.0, 11.0, 12.4), uv_scale=0.24)
        # Projecting Mamluk roof cornice & stepped crenellations
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (tx, ty, tz + 10.55), (16.5, 11.5, 0.38), uv_scale=0.30)
        for cr_x in (-6.8, -3.4, 0.0, 3.4, 6.8):
            collector.add_box("Buildings_Details", "mat_stone_boss", (tx + cr_x, ty - 5.45, tz + 11.05), (1.15, 0.55, 0.65), uv_scale=0.32)
        # Flush alternating Mamluk Ablaq red/cream limestone courses (`0.05m` proud) — split around window/door piers below `z = +4.2m`!
        for band_z in (1.4, 2.2, 3.0, 4.4, 5.2, 6.0, 6.8, 7.6, 8.4):
            b_mat = "mat_terracotta" if int(band_z * 2) % 2 == 0 else "mat_stone_mizzi"
            if band_z < 4.2:
                for pier_x, pier_w in ((-6.95, 2.1), (-2.4, 2.4), (2.4, 2.4), (6.95, 2.1)):
                    collector.add_box(
                        "Buildings_Details",
                        b_mat,
                        (tx + pier_x, ty - 5.52, tz + band_z),
                        (pier_w, 0.06, 0.34),
                        uv_scale=0.35,
                    )
            else:
                collector.add_box(
                    "Buildings_Details",
                    b_mat,
                    (tx, ty - 5.52, tz + band_z),
                    (16.04, 0.06, 0.34),
                    uv_scale=0.35,
                )
        # Mamluk street facade arched bronze/iron grille windows (`shabaka`) & tall Muqarnas portal (`iwan`)
        for wx_off in (-4.8, 0.0, 4.8):
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (tx + wx_off, ty - 5.51),
                tz + (0.0 if wx_off == 0.0 else 0.9),
                tz + 2.5,
                1.10,
                2.20,
                0.01,
                (1.0, 0.0),
                (0.0, -1.0),
                uv_rect=WOOD_DOOR_UV["door_wood" if wx_off == 0.0 else "window_glass"],
                num_voussoirs=7,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (tx + wx_off, ty - 5.51, tz + 2.5),
                (1.0, 0.0),
                (0.0, -1.0),
                span_w=2.20,
                arch_h=1.10,
                ring_thickness=0.34,
                offset_n0=-0.04,
                offset_n1=0.22,
                num_voussoirs=7,
            )
            if wx_off == 0.0:
                # Outer Mamluk Muqarnas Portal Hood Arch (`iwan`) framing the central entrance
                add_arch_voussoirs(
                    collector,
                    "Buildings_Details",
                    "mat_terracotta",
                    (tx, ty - 5.51, tz + 3.8),
                    (1.0, 0.0),
                    (0.0, -1.0),
                    span_w=3.10,
                    arch_h=1.75,
                    ring_thickness=0.38,
                    offset_n0=-0.02,
                    offset_n1=0.32,
                    num_voussoirs=9,
                )
            else:
                # 3D Mamluk bronze/iron grid window grilles (`shabaka`) + carved stone lintel plaque
                for gx_b in (-0.65, 0.0, 0.65):
                    collector.add_box("Buildings_Openings", "mat_iron_dark", (tx + wx_off + gx_b, ty - 5.56, tz + 1.85), (0.05, 0.06, 1.95))
                for gz_b in (1.35, 1.85, 2.35):
                    collector.add_box("Buildings_Openings", "mat_iron_dark", (tx + wx_off, ty - 5.56, tz + gz_b), (2.05, 0.06, 0.05))
                collector.add_box("Buildings_Details", "mat_stone_mizzi", (tx + wx_off, ty - 5.58, tz + 4.15), (2.35, 0.16, 0.48), uv_scale=0.35)
        # Twin Mamluk mausoleum stone domes (`qubba`) positioned near the front roofline (`ty - 2.2`) so they rise above the street facade
        for off_x in (-3.8, 3.8):
            collector.add_branch_tube(
                "Buildings_Walls",
                "mat_stone_mizzi",
                (tx + off_x, ty - 2.2, tz + 10.4),
                (tx + off_x, ty - 2.2, tz + 13.0),
                2.85,
                2.75,
                segments=12,
                cap_end=True,
            )
            collector.add_dome(
                "Buildings_Details",
                "mat_stone_mizzi",
                (tx + off_x, ty - 2.2, tz + 12.9),
                radius=2.8,
                height=3.2,
                rings=7,
                segments=12,
            )

    # 4. Ottoman Street Sabil (`Sabil Bab al-Silsila` at `(572.0, 68.0)`)
    sb_x, sb_y = 572.0, 68.0
    sb_z = street_elevation_z(sb_x, sb_y)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (sb_x, sb_y, sb_z + 2.6), (4.8, 2.2, 5.2), uv_scale=0.28)
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        "mat_stone_boss",
        (sb_x, sb_y - 1.12, sb_z + 2.4),
        (1.0, 0.0),
        (0.0, -1.0),
        span_w=2.8,
        arch_h=1.4,
        ring_thickness=0.36,
        offset_n0=-0.02,
        offset_n1=0.25,
        num_voussoirs=9,
    )


def build_region_r03_central_souqs(
    collector: MeshCollector,
    osm_data: dict,
    sign_atlas: Dict[str, dict],
) -> int:
    """
    Builds Region 3 (`r03_central_souqs`):
      - Continuous stone-paved ground surface across `[165..605] x [-45..260]` (excluding `r02` NW block)
      - Crusader Vaulted Triple Souq & Northern Cardo (`x = 296..330, y = -36..156`)
      - Lower David Street & Street of the Chain (`Tariq Bab al-Silsila`), Mamluk Turbas, Ottoman Sabil
      - Khan el-Zeit & Central El-Wad / HaGai Valley buildings, trees, and street props
    """
    def r03_exclude(x: float, y: float) -> bool:
        if x < 295.0 and y > 55.0:
            return True
        return False

    build_region_ground_grid(collector, 165.0, 668.0, -45.0, 260.0, step=5.5, exclude_fn=r03_exclude)
    _build_vaulted_triple_souq_and_chain_street(collector, sign_atlas)

    # Khan el-Zeit Souq (`(305, 162) -> (305, 252)`), El-Wad / HaGai Street (`(510, 72) -> (510, 235)`),
    # Street of the Chain (`(332, 9.2) -> (428, 17.6)`), and Vaulted Mamluk Suq al-Qattanin (`(565, 145) -> (662, 145)`)
    _build_souq_street_corridor(collector, (305.0, 162.0), (305.0, 252.0), street_w=6.4, wall_h=8.4, with_arches=True, with_shops=True, seed=301)
    _build_souq_street_corridor(collector, (510.0, 72.0), (510.0, 235.0), street_w=6.2, wall_h=8.4, with_arches=True, with_shops=True, seed=302)
    _build_souq_street_corridor(collector, (565.0, 145.0), (662.0, 145.0), street_w=6.0, wall_h=8.8, with_arches=True, with_shops=True, vaulted_roof=True, seed=303)
    _build_souq_street_corridor(collector, (332.0, 9.2), (428.0, 17.6), street_w=8.4, wall_h=8.4, with_arches=True, with_shops=True, seed=304)

    # Bab al-Qattanin (Cotton Merchants' Gate) Mamluk Ablaq Portal at the east end of Suq al-Qattanin (`(665.0, 145.0)`)
    q_x, q_y = 665.0, 145.0
    q_z = street_elevation_z(q_x, q_y)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (q_x + 1.2, q_y, q_z + 5.6), (2.4, 9.8, 11.2), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_stone_boss", (q_x + 1.2, q_y, q_z + 11.35), (2.9, 10.4, 0.45), uv_scale=0.30)
    for face_dx, fnx, fty in ((-0.08, -1.0, 1.0), (2.48, 1.0, -1.0)):
        for ab_z in (4.5, 5.4, 6.3, 7.2, 8.1, 9.0, 9.9):
            collector.add_box(
                "Buildings_Details",
                "mat_terracotta" if int(ab_z * 2) % 2 == 0 else "mat_stone_boss",
                (q_x + face_dx, q_y, q_z + ab_z),
                (0.24, 9.0, 0.36),
                uv_scale=0.35,
            )
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (q_x + (0.0 if fnx < 0 else 2.40) + fnx * 0.02, q_y),
            q_z,
            q_z + 3.4,
            1.8,
            3.6,
            0.01,
            (0.0, fty),
            (fnx, 0.0),
            uv_rect=WOOD_DOOR_UV["door_wood"],
            num_voussoirs=9,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (q_x + (0.0 if fnx < 0 else 2.40), q_y, q_z + 3.4),
            (0.0, fty),
            (fnx, 0.0),
            span_w=3.6,
            arch_h=1.8,
            ring_thickness=0.45,
            offset_n0=-0.04,
            offset_n1=0.32,
            num_voussoirs=9,
        )
        # Outer Mamluk muqarnas hood arch & circular stone spandrel medallions
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_terracotta",
            (q_x + (0.0 if fnx < 0 else 2.40), q_y, q_z + 4.8),
            (0.0, fty),
            (fnx, 0.0),
            span_w=4.4,
            arch_h=2.4,
            ring_thickness=0.36,
            offset_n0=-0.02,
            offset_n1=0.42,
            num_voussoirs=11,
        )
        for med_y in (-2.9, 2.9):
            rx_m = q_x + (0.0 if fnx < 0 else 2.40)
            collector.add_branch_tube(
                "Buildings_Details",
                "mat_stone_mizzi",
                (rx_m, q_y + med_y, q_z + 6.8),
                (rx_m + fnx * 0.18, q_y + med_y, q_z + 6.8),
                0.48,
                0.44,
                segments=10,
            )

    clear_scene_building_rings()
    osm_buildings = [
        b for b in osm_data.get("buildings", [])
        if b.get("region") == "r03_central_souqs"
    ]

    r03_corridors = [
        [(175.0, 4.0), (300.0, 8.0), (440.0, 36.0), (595.0, 68.0)],
        [(305.0, 155.0), (305.0, 260.0)],
        [(510.0, 58.0), (510.0, 250.0)],
        [(515.0, 145.0), (668.0, 145.0)],
    ]

    valid_blds = []
    for b in osm_buildings:
        osm_id = int(b.get("osm_id", 0))
        if osm_id in (5862586, 7880713):
            continue
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        ring = b.get("ring", [])
        if len(ring) < 3:
            continue
        rxs = [float(p[0]) for p in ring]
        rys = [float(p[1]) for p in ring]
        if (max(rxs) - min(rxs)) > 65.0 or (max(rys) - min(rys)) > 65.0:
            continue
        # Keep Triple Souq corridor (`x = 290..336, y = -42..164`) and Suq al-Qattanin (`x = 558..670, y = 132..158`) clear
        if _ring_overlaps_rect(ring, 290.0, -42.0, 336.0, 164.0):
            continue
        if _ring_overlaps_rect(ring, 558.0, 132.0, 670.0, 158.0):
            continue
        # Keep Street of the Chain, Khan el-Zeit, El-Wad, and Suq al-Qattanin corridors clear
        if any(_ring_overlaps_polyline(ring, corr, 9.5) for corr in r03_corridors):
            continue
        valid_blds.append(b)
        register_scene_building_ring(osm_id, ring)

    built_count = 0
    for idx, b in enumerate(valid_blds):
        ring = b.get("ring", [])
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        bz = street_elevation_z(cx, cy)
        lvls = max(2, min(3, int(b.get("levels", 2))))
        build_building_from_footprint(
            collector,
            ring,
            base_z=bz,
            levels=lvls,
            sign_atlas=sign_atlas,
            building_id=int(b.get("osm_id", idx)),
            is_hero=False,
        )
        built_count += 1

    # Trees, street lamps, and market clutter along Khan el-Zeit and El-Wad
    for tx, ty, sp, th, seed in (
        (338.0, 85.0, "olive", 6.2, 401),
        (415.0, 95.0, "cypress", 10.5, 402),
        (485.0, 120.0, "olive", 6.5, 403),
        (535.0, 88.0, "cypress", 11.0, 404),
        (365.0, -18.0, "olive", 6.0, 405),
    ):
        tz = street_elevation_z(tx, ty)
        if sp == "cypress":
            build_cypress_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)
        else:
            build_olive_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)

    for idx_g, (gx, gy) in enumerate(((297.0, 22.0), (328.0, 64.0), (435.0, 32.0), (515.0, 48.0))):
        build_garbage_pile(collector, gx, gy, street_elevation_z(gx, gy), rot_z=0.4 * idx_g, bag_count=4, seed=450 + idx_g)

    return built_count


# ==============================================================================
# 5. REGION 4: ARMENIAN QUARTER, ZION GATE & ST. JAMES CATHEDRAL (`r04_armenian_quarter`)
# ==============================================================================

def _build_st_james_cathedral_and_patriarchate(collector: MeshCollector, sign_atlas: Dict[str, dict]) -> None:
    """
    Builds the 12th-century St. James Cathedral (`Surb Hakob` at `(132.5, -227.3)`),
    its vaulted narthex porch with cobalt-blue Armenian ceramic tiles (`mat_tile_iznik`),
    conical Armenian cupola dome, and the Armenian Patriarchate Compound & Garden along
    Armenian Patriarchate Road (`x = 46..56, y = -85..-380`).
    """
    cx, cy = 132.5, -227.3
    bz = street_elevation_z(cx, cy)

    # 1. Cathedral Basilica Hall (`28m x 21m x 11.8m`)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (cx, cy, bz + 5.9), (28.0, 21.0, 11.8), uv_scale=0.22)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (cx, cy, bz + 12.0), (28.6, 21.6, 0.45), uv_scale=0.30)

    # 2. Central 12-sided Armenian Drum + Conical Pyramidal Cupola Dome (`z = +11.8 .. +22.5m`)
    collector.add_branch_tube(
        "Buildings_Walls",
        "mat_stone_mizzi",
        (cx, cy, bz + 11.8),
        (cx, cy, bz + 16.5),
        4.8,
        4.6,
        segments=12,
        cap_end=True,
    )
    for ang_deg in range(0, 360, 30):
        ang = math.radians(ang_deg)
        wx = cx + math.cos(ang) * 4.72
        wy = cy + math.sin(ang) * 4.72
        tx, ty = -math.sin(ang), math.cos(ang)
        nx, ny = math.cos(ang), math.sin(ang)
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (wx, wy),
            bz + 12.4,
            bz + 14.8,
            0.55,
            1.1,
            0.02,
            (tx, ty),
            (nx, ny),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=5,
            add_wall_spandrel=False,
        )
    # Conical cupola roof (`r0=4.9 -> r1=0.15`)
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_dome_lead",
        (cx, cy, bz + 16.4),
        (cx, cy, bz + 21.8),
        4.95,
        0.18,
        segments=12,
        cap_end=True,
    )
    # Cross finial
    collector.add_box("Buildings_Details", "mat_dome_gold", (cx, cy, bz + 22.6), (0.18, 0.18, 1.6), uv_scale=0.5)
    collector.add_box("Buildings_Details", "mat_dome_gold", (cx, cy, bz + 22.9), (0.95, 0.18, 0.18), uv_scale=0.5)

    # 3. Southern Arched Narthex Porch (`Etchmiadzin Chapel` entrance) with Blue Armenian Ceramic Tilework (`mat_tile_iznik`)
    ny = cy - 12.2
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (cx, ny, bz + 3.6), (22.0, 3.8, 7.2), uv_scale=0.25)
    # Moulded stone stringcourses framing the upper cobalt-blue Armenian ceramic tile frieze (well above the narthex arch crowns!)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (cx, ny - 1.98, bz + 4.92), (19.2, 0.24, 0.22), uv_scale=0.35)
    collector.add_box("Buildings_Details", "mat_tile_iznik", (cx, ny - 1.94, bz + 5.75), (18.6, 0.12, 1.30), uv_scale=0.45)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (cx, ny - 2.02, bz + 6.55), (19.6, 0.30, 0.28), uv_scale=0.35)
    for arch_x in (-6.2, 0.0, 6.2):
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (cx + arch_x, ny - 1.91),
            bz,
            bz + 2.8,
            1.40,
            3.10,
            0.01,
            (1.0, 0.0),
            (0.0, -1.0),
            uv_rect=WOOD_DOOR_UV["door_wood" if arch_x == 0.0 else "window_glass"],
            num_voussoirs=9,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (cx + arch_x, ny - 1.91, bz + 2.8),
            (1.0, 0.0),
            (0.0, -1.0),
            span_w=3.10,
            arch_h=1.40,
            ring_thickness=0.38,
            offset_n0=-0.04,
            offset_n1=0.28,
            num_voussoirs=9,
        )
        if arch_x != 0.0:
            # Ornate wrought-iron grille screen + central golden/white Armenian cross in the side narthex arches
            for gx_s in (-0.95, 0.0, 0.95):
                collector.add_box("Buildings_Openings", "mat_iron_dark", (cx + arch_x + gx_s, ny - 1.96, bz + 1.55), (0.05, 0.05, 2.95))
            for gz_s in (0.95, 1.85, 2.75):
                collector.add_box("Buildings_Openings", "mat_iron_dark", (cx + arch_x, ny - 1.96, bz + gz_s), (2.95, 0.05, 0.05))
            collector.add_box("Buildings_Details", "mat_stone_mizzi", (cx + arch_x, ny - 1.98, bz + 2.15), (0.14, 0.08, 0.78))
            collector.add_box("Buildings_Details", "mat_stone_mizzi", (cx + arch_x, ny - 1.98, bz + 2.25), (0.56, 0.08, 0.14))
        # Hanging ornate bronze Armenian church oil lamp inside each narthex arch
        collector.add_box("StreetProps", "mat_dome_gold", (cx + arch_x, ny - 2.08, bz + 3.65), (0.04, 0.04, 0.85))
        collector.add_box("StreetProps", "mat_lantern_glow", (cx + arch_x, ny - 2.08, bz + 3.12), (0.26, 0.26, 0.34), bottom_face=True)
    # Carved Armenian stone Khachkars (cross-stones) set into the narthex piers between the arches
    for kh_x in (-9.2, -3.1, 3.1, 9.2):
        collector.add_box("Buildings_Details", "mat_stone_boss", (cx + kh_x, ny - 1.95, bz + 2.2), (0.95, 0.16, 1.65), uv_scale=0.35)
        collector.add_box("Buildings_Details", "mat_tile_iznik", (cx + kh_x, ny - 2.01, bz + 3.55), (0.75, 0.08, 0.55), uv_scale=0.5)

    # 4. Armenian Patriarchate Road Compound Walls, Vaulted Overpasses & Buttress Arches (`x = 52.0, y = -110..-360`)
    _build_souq_street_corridor(
        collector,
        (52.0, -105.0),
        (52.0, -355.0),
        street_w=6.8,
        wall_h=8.2,
        with_arches=False,
        with_shops=False,
        seed=401,
    )
    for py in (-125.0, -165.0, -205.0, -245.0, -285.0, -325.0):
        pz = street_elevation_z(52.0, py)
        for sx in (-3.2, 3.2):
            collector.add_box("Buildings_Walls", "mat_stone_boss", (52.0 + sx, py, pz + 3.2), (1.3, 1.8, 6.4), uv_scale=0.26)
        # 3D Vaulted Stone Overpass (`Qantara`) with pointed arch rings, spandrel wall, upper window & hanging lantern
        add_arch_spandrel_wall(
            collector,
            "Buildings_Walls",
            "mat_stone_ashlar_aged",
            (52.0, py, pz + 3.5),
            (1.0, 0.0),
            (0.0, 1.0),
            span_w=6.2,
            arch_h=2.1,
            z_top=pz + 7.8,
            offset_n0=-1.8,
            offset_n1=1.8,
            num_voussoirs=9,
        )
        for ny_dir in (-1.0, 1.0):
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (52.0, py + ny_dir * 1.8, pz + 3.5),
                (1.0, 0.0),
                (0.0, ny_dir),
                span_w=6.2,
                arch_h=2.1,
                ring_thickness=0.38,
                offset_n0=-0.04,
                offset_n1=0.22,
                num_voussoirs=9,
            )
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (52.0, py, pz + 7.95), (7.6, 3.9, 0.34), uv_scale=0.30, bottom_face=True)
        collector.add_box("StreetProps", "mat_lantern_glow", (52.0, py, pz + 5.15), (0.28, 0.28, 0.38), bottom_face=True)
        build_street_lamp(collector, 54.6, py + 4.0, pz, height=4.2)

    # 5. Ararat Street Bazaar & Residential Corridor (`(118.0, -265.0) -> (172.0, -265.0)`)
    _build_souq_street_corridor(
        collector,
        (118.0, -265.0),
        (172.0, -265.0),
        street_w=5.8,
        wall_h=8.2,
        with_arches=True,
        with_shops=True,
        seed=402,
    )

    # 6. Armenian Garden Western Monastery Facade Articulation, Garden Portal & Raised Terrace Beds (`x = 18..46, y = -220..-345`)
    #    Articulates the West face (`x = 46.0`, facing `(-1, 0)`) of the Armenian Patriarchate Compound overlooking the Garden
    for gy_b in (-232.0, -250.0, -268.0, -286.0, -304.0, -322.0):
        gz_b = street_elevation_z(46.0, gy_b)
        # Sloped ashlar stone buttress pier & cornice waterspout (gargoyle)
        collector.add_box("Buildings_Details", "mat_stone_boss", (45.55, gy_b - 5.2, gz_b + 2.4), (0.95, 1.05, 4.8), uv_scale=0.26)
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (45.25, gy_b - 5.2, gz_b + 8.05), (0.85, 0.28, 0.24), uv_scale=0.35)
        # Ground-floor arched monastery garden portal (`gy_b == -268.0`) or recessed blind stone arch
        is_g_portal = (gy_b == -268.0)
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors" if is_g_portal else "mat_stone_ashlar_aged",
            (45.98, gy_b),
            gz_b + (0.0 if is_g_portal else 0.4),
            gz_b + 2.5,
            1.15,
            2.30,
            0.01,
            (0.0, -1.0),
            (-1.0, 0.0),
            uv_rect=WOOD_DOOR_UV["door_wood"] if is_g_portal else (0.0, 0.0, 1.0, 1.0),
            num_voussoirs=7,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (45.98, gy_b, gz_b + 2.5),
            (0.0, -1.0),
            (-1.0, 0.0),
            span_w=2.30,
            arch_h=1.15,
            ring_thickness=0.34,
            offset_n0=-0.04,
            offset_n1=0.22,
            num_voussoirs=7,
        )
        if is_g_portal:
            # Framed Armenian ceramic plaque, Khachkar cross relief & twin wrought-iron wall lanterns flanking the garden gate
            collector.add_box("Buildings_Details", "mat_stone_mizzi", (45.88, gy_b, gz_b + 4.25), (0.18, 1.25, 0.52), uv_scale=0.35)
            collector.add_box("Buildings_Details", "mat_tile_iznik", (45.82, gy_b, gz_b + 4.25), (0.10, 1.05, 0.36), uv_scale=0.5)
            for l_dy in (-1.65, 1.65):
                build_wall_lantern(collector, (45.85, gy_b + l_dy, gz_b + 3.2), (0.0, -1.0), (-1.0, 0.0))
        # Upper-floor arched monastery window with stone sill & paired green louvered shutters
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (45.98, gy_b),
            gz_b + 4.8,
            gz_b + 6.3,
            0.68,
            1.35,
            0.01,
            (0.0, -1.0),
            (-1.0, 0.0),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=5,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (45.98, gy_b, gz_b + 6.3),
            (0.0, -1.0),
            (-1.0, 0.0),
            span_w=1.35,
            arch_h=0.68,
            ring_thickness=0.26,
            offset_n0=-0.04,
            offset_n1=0.18,
            num_voussoirs=5,
        )
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (45.85, gy_b, gz_b + 4.72), (0.24, 1.60, 0.16), uv_scale=0.35)
        for sh_dy in (-0.88, 0.88):
            collector.add_box(
                "Buildings_Openings",
                "mat_wood_doors",
                (45.90, gy_b + sh_dy, gz_b + 5.60),
                (0.08, 0.36, 1.50),
                uv_rect=WOOD_DOOR_UV["shutter_louvre"],
            )
    # Raised stone-curbed Armenian Garden terrace beds & carved stone garden benches along the western rampart walk (`x = 22..31, y = -246..-332`)
    for gt_y in (-256.0, -284.0, -312.0):
        gt_z = street_elevation_z(26.5, gt_y)
        collector.add_box("StreetProps", "mat_stone_mizzi", (26.5, gt_y, gt_z + 0.32), (8.2, 18.0, 0.64), uv_scale=0.30)
        collector.add_box("Trees_Foliage", "mat_foliage_olive", (26.5, gt_y, gt_z + 0.72), (7.4, 17.2, 0.36), uv_scale=0.6)
        # Carved Jerusalem stone garden bench facing the promenade (`x = 31.4`)
        collector.add_box("StreetProps", "mat_stone_boss", (31.4, gt_y, gt_z + 0.26), (0.65, 2.2, 0.52), uv_scale=0.35)

    # 7. Zion Gate Inner Rampart Lane (`(148.0, -390.0) -> (186.0, -393.0)`):
    #    - 2-Storey Armenian Quarter stone street-front facades on the left (north) side of the lane
    #    - Exterior stone rampart staircase ascending the inner curtain wall + stone bollards on the right (south) side
    for zb_i, zb_x in enumerate((155.0, 168.0, 181.0)):
        zb_y = -387.5 - zb_i * 1.2
        zb_z = street_elevation_z(zb_x, zb_y)
        collector.add_box("Buildings_Walls", "mat_stone_mizzi" if zb_i % 2 == 0 else "mat_stone_ashlar", (zb_x, zb_y + 4.2, zb_z + 4.2), (12.8, 8.4, 8.4), uv_scale=0.24)
        collector.add_box("Buildings_Details", "mat_stone_boss", (zb_x, zb_y + 4.2, zb_z + 4.15), (13.1, 8.6, 0.24), uv_scale=0.32)
        collector.add_box("Buildings_Details", "mat_stone_boss", (zb_x, zb_y + 4.2, zb_z + 8.55), (13.3, 8.8, 0.38), uv_scale=0.30)
        if zb_i % 2 == 0:
            collector.add_dome("Buildings_Details", "mat_stone_ashlar_aged", (zb_x, zb_y + 4.2, zb_z + 8.65), radius=2.1, height=1.5, rings=5, segments=10)
        for bay_dx in (-3.2, 3.2):
            bx_z = zb_x + bay_dx
            by_z = zb_y - 0.02
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (bx_z, by_z),
                zb_z,
                zb_z + 2.15,
                1.05,
                2.20,
                0.01,
                (1.0, 0.0),
                (0.0, -1.0),
                uv_rect=WOOD_DOOR_UV["souq_open" if bay_dx < 0 else "door_wood"],
                num_voussoirs=7,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (bx_z, by_z, zb_z + 2.15),
                (1.0, 0.0),
                (0.0, -1.0),
                span_w=2.20,
                arch_h=1.05,
                ring_thickness=0.32,
                offset_n0=-0.04,
                offset_n1=0.20,
                num_voussoirs=7,
            )
            # Framed cobalt-blue Armenian ceramic shop plaque above the ground arch
            collector.add_box("Buildings_Details", "mat_stone_mizzi", (bx_z, by_z - 0.06, zb_z + 3.55), (1.35, 0.12, 0.42), uv_scale=0.35)
            collector.add_box("Buildings_Details", "mat_tile_iznik", (bx_z, by_z - 0.10, zb_z + 3.55), (1.15, 0.08, 0.28), uv_scale=0.5)
            # Upper-floor arched window with stone sill & paired green louvered shutters
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (bx_z, by_z),
                zb_z + 4.7,
                zb_z + 6.1,
                0.65,
                1.30,
                0.01,
                (1.0, 0.0),
                (0.0, -1.0),
                uv_rect=WOOD_DOOR_UV["window_glass"],
                num_voussoirs=5,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (bx_z, by_z, zb_z + 6.1),
                (1.0, 0.0),
                (0.0, -1.0),
                span_w=1.30,
                arch_h=0.65,
                ring_thickness=0.26,
                offset_n0=-0.04,
                offset_n1=0.18,
                num_voussoirs=5,
            )
            collector.add_box("Buildings_Details", "mat_stone_mizzi", (bx_z, by_z - 0.12, zb_z + 4.62), (1.55, 0.24, 0.16), uv_scale=0.35)
            for sh_dx in (-0.84, 0.84):
                collector.add_box("Buildings_Openings", "mat_wood_doors", (bx_z + sh_dx, by_z - 0.08, zb_z + 5.45), (0.36, 0.08, 1.48), uv_rect=WOOD_DOOR_UV["shutter_louvre"])
        # Projecting wooden Mashrabiya bay window & wrought-iron wall lantern on each building
        collector.add_box("Buildings_Details", "mat_bark", (zb_x, zb_y - 0.38, zb_z + 5.65), (1.55, 0.68, 1.85), uv_scale=0.35, bottom_face=True)
        build_wall_lantern(collector, (zb_x, zb_y - 0.08, zb_z + 3.4), (1.0, 0.0), (0.0, -1.0))
    # Exterior Stone Rampart Staircase ascending the inner Southern City Wall (`x = 158.0..176.0, y = -405.5`) + Stone Bollards
    for st_i in range(14):
        sx_r = 158.0 + st_i * 1.15
        sy_r = -404.8 - st_i * 0.22
        sz_r = street_elevation_z(sx_r, sy_r)
        sh_r = (st_i + 1) * 0.55
        collector.add_box("Ground_Steps", "mat_stone_boss", (sx_r, sy_r, sz_r + 0.5 * sh_r), (1.22, 2.1, sh_r), uv_scale=0.26)
        collector.add_box("CityWalls", "mat_stone_mizzi", (sx_r, sy_r + 0.95, sz_r + sh_r + 0.42), (1.22, 0.34, 0.84), uv_scale=0.32)
    for bol_x in (154.0, 160.0, 166.0, 172.0, 178.0):
        build_bollard(collector, bol_x, -398.5, street_elevation_z(bol_x, -398.5))


def build_region_r04_armenian_quarter(
    collector: MeshCollector,
    osm_data: dict,
    sign_atlas: Dict[str, dict],
) -> int:
    """
    Builds Region 4 (`r04_armenian_quarter`):
      - Continuous stone-paved ground surface across `[-30..285] x [-430..-45]`
      - SW Ottoman City Wall (`Segment 1`: Citadel south -> SW Corner -> Zion Gate -> Jewish Quarter)
      - Zion Gate (`Bab al-Nabi Dawud` at `(190.6, -411.4)`)
      - St. James Cathedral (`(132.5, -227.3)`), Armenian Patriarchate Road, and Armenian Garden groves
      - All valid `r04_armenian_quarter` OSM buildings
    """
    def r04_exclude(x: float, y: float) -> bool:
        if x < 85.0 and y > -86.0:
            return True
        return False

    build_region_ground_grid(collector, -25.0, 285.0, -455.0, -45.0, step=5.5, exclude_fn=r04_exclude)

    # 1. SW Ottoman City Wall (`Segment 1` of relation `5862586`, starting south of Citadel at `y <= -85`)
    sw_wall_pts = [
        (14.5, -85.0),
        (16.0, -118.9),
        (8.8, -158.9),
        (11.3, -175.0),
        (7.2, -193.6),
        (8.6, -225.7),
        (4.2, -242.5),
        (10.0, -296.0),
        (7.7, -349.1),
        (13.2, -372.4),
        (5.7, -389.4),
        (60.6, -395.5),
        (104.0, -408.8),
        (153.5, -406.5),
        (180.9, -413.2),
        (190.6, -411.4),
        (225.5, -415.9),
        (267.0, -402.8),
        (286.5, -407.1),
    ]
    build_ottoman_wall_polyline(
        collector,
        sw_wall_pts,
        wall_height=12.5,
        thickness=3.2,
        gate_gaps=[(190.6, -411.4, 8.5)],
        bastion_every=3,
    )

    # 2. Zion Gate (`Bab al-Nabi Dawud` at `(190.6, -411.4)`, facing North `8 deg` into the city)
    build_ottoman_gatehouse(
        collector,
        (190.6, -411.4),
        heading_inward_deg=8.0,
        gate_name="Zion Gate (Bab al-Nabi Dawud)",
        width=16.5,
        depth=12.5,
        height=14.2,
    )

    # 2b. Dormition Abbey on Mount Zion (`(154.0, -438.0)`, rising immediately south-west of Zion Gate)
    da_x, da_y = 154.0, -438.0
    da_z = street_elevation_z(da_x, da_y)
    collector.add_branch_tube(
        "Buildings_Walls",
        "mat_stone_mizzi",
        (da_x, da_y, da_z - 2.0),
        (da_x, da_y, da_z + 18.5),
        11.2,
        11.0,
        segments=16,
        cap_end=True,
    )
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_dome_lead",
        (da_x, da_y, da_z + 18.4),
        (da_x, da_y, da_z + 29.8),
        11.3,
        0.25,
        segments=16,
        cap_end=True,
    )
    for t_ang in (45.0, 135.0, 225.0, 315.0):
        trad = math.radians(t_ang)
        tx_d = da_x + math.cos(trad) * 10.8
        ty_d = da_y + math.sin(trad) * 10.8
        collector.add_branch_tube("Buildings_Walls", "mat_stone_mizzi", (tx_d, ty_d, da_z), (tx_d, ty_d, da_z + 20.5), 2.1, 2.0, segments=10, cap_end=True)
        collector.add_branch_tube("Buildings_Details", "mat_dome_lead", (tx_d, ty_d, da_z + 20.4), (tx_d, ty_d, da_z + 25.0), 2.15, 0.12, segments=10, cap_end=True)
    # Dormition Bell Tower (`(171.0, -440.0)`)
    collector.add_branch_tube("Buildings_Walls", "mat_stone_mizzi", (171.0, -440.0, da_z - 2.0), (171.0, -440.0, da_z + 28.5), 3.4, 3.2, segments=12, cap_end=True)
    collector.add_branch_tube("Buildings_Details", "mat_dome_lead", (171.0, -440.0, da_z + 28.4), (171.0, -440.0, da_z + 36.5), 3.4, 0.15, segments=12, cap_end=True)

    # 3. St. James Cathedral & Armenian Patriarchate Compound
    _build_st_james_cathedral_and_patriarchate(collector, sign_atlas)

    # 4. OSM Buildings in `r04_armenian_quarter`
    clear_scene_building_rings()
    osm_buildings = [
        b for b in osm_data.get("buildings", [])
        if b.get("region") == "r04_armenian_quarter"
    ]
    valid_blds = []
    for b in osm_buildings:
        osm_id = int(b.get("osm_id", 0))
        if osm_id in (5862586, 7880713, 265410575):
            continue
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        if cx < 86.0 and cy > -88.0:
            continue
        ring = b.get("ring", [])
        if len(ring) < 3:
            continue
        rxs = [float(p[0]) for p in ring]
        rys = [float(p[1]) for p in ring]
        if (max(rxs) - min(rxs)) > 65.0 or (max(rys) - min(rys)) > 65.0:
            continue
        # Keep St. James Cathedral, forecourt & Ararat Street (`x = 108..178, y = -274..-208`) clear
        if _ring_overlaps_rect(ring, 108.0, -274.0, 178.0, -208.0):
            continue
        # Keep Armenian Garden (`x = 12..46, y = -360..-115`) & Armenian Patriarchate Road (`x = 43..61, y = -395..-85`) clear
        if _ring_overlaps_rect(ring, 12.0, -360.0, 61.0, -85.0):
            continue
        # Keep Zion Gate inner lane & plaza (`x = 125..215, y = -418..-376`) clear
        if _ring_overlaps_rect(ring, 125.0, -418.0, 215.0, -376.0):
            continue
        valid_blds.append(b)
        register_scene_building_ring(osm_id, ring)

    built_count = 0
    for idx, b in enumerate(valid_blds):
        ring = b.get("ring", [])
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        bz = street_elevation_z(cx, cy)
        lvls = max(2, min(3, int(b.get("levels", 2))))
        build_building_from_footprint(
            collector,
            ring,
            base_z=bz,
            levels=lvls,
            sign_atlas=sign_atlas,
            building_id=int(b.get("osm_id", idx)),
            is_hero=False,
        )
        built_count += 1

    # 5. Armenian Garden Pine/Cypress & Olive Groves (`x = 20..42, y = -130..-350`)
    garden_trees = [
        (26.0, -135.0, "cypress", 11.5, 501),
        (34.0, -165.0, "olive", 6.5, 502),
        (24.0, -195.0, "cypress", 12.0, 503),
        (36.0, -225.0, "olive", 6.8, 504),
        (28.0, -260.0, "cypress", 11.2, 505),
        (23.5, -252.0, "olive", 6.6, 511),
        (29.0, -276.0, "cypress", 11.6, 512),
        (35.0, -295.0, "olive", 6.4, 506),
        (24.0, -288.0, "olive", 6.5, 513),
        (28.5, -310.0, "cypress", 11.4, 514),
        (25.0, -330.0, "cypress", 11.8, 507),
        (118.0, -244.0, "cypress", 10.5, 508),
        (146.0, -244.0, "olive", 6.2, 509),
        (182.0, -398.0, "olive", 6.5, 510),
    ]
    for tx, ty, sp, th, seed in garden_trees:
        tz = street_elevation_z(tx, ty)
        if sp == "cypress":
            build_cypress_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)
        else:
            build_olive_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)

    return built_count


# ==============================================================================
# 6. REGION 5: JEWISH QUARTER, HURVA SYNAGOGUE & ROMAN CARDO (`r05_jewish_quarter`)
# ==============================================================================

def _build_hurva_synagogue_and_roman_cardo(collector: MeshCollector, sign_atlas: Dict[str, dict]) -> None:
    """
    Builds:
      1. The Hurva Synagogue (`Beit El / HaHurva` at `(346.7, -141.4)`) with its 4 monumental corner piers,
         giant arched multi-light windows, 12-windowed cylindrical stone drum, and soaring 24m white/lead dome,
         plus Hurva Square (`y = -126..-98`) and the Golden Menorah Pavilion (`(378.0, -108.0)`).
      2. The Southern Roman Cardo (`x = 308.0, y = -195..-55`) with twin rows of monolithic limestone columns,
         stone architrave beams, and Crusader pointed-arch shopfronts.
      3. The Four Sephardic Synagogues (`(366.0, -238.0)`) and Hezekiah's Broad Wall (`(335.0, -78.0)`).
    """
    # 1. Hurva Synagogue (`(346.7, -141.4)`)
    hx, hy = 346.7, -141.4
    hz = street_elevation_z(hx, hy)

    # Squaredressed Mizzi stone base (`22m x 22m`, extending down to `hz - 2.5m` so it is grounded on the slope)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (hx, hy, hz + 5.15), (22.0, 22.0, 15.3), uv_scale=0.22)
    # 4 monumental projecting corner stone piers (`3.2m x 3.2m`, grounded down to `hz - 2.5m`)
    for sx in (-10.2, 10.2):
        for sy in (-10.2, 10.2):
            collector.add_box("Buildings_Details", "mat_stone_boss", (hx + sx, hy + sy, hz + 5.55), (3.2, 3.2, 16.1), uv_scale=0.24)
    # Upper stone cornice & balustrade
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (hx, hy, hz + 13.0), (23.0, 23.0, 0.55), uv_scale=0.30)

    # Giant arched multi-light synagogue windows on all 4 facades + North Entrance Portal facing Hurva Square
    for side_dir, (fx, fy), (tx, ty), (nx, ny) in (
        ("N", (hx, hy + 11.04), (-1.0, 0.0), (0.0, 1.0)),
        ("S", (hx, hy - 11.04), (1.0, 0.0), (0.0, -1.0)),
        ("E", (hx + 11.04, hy), (0.0, 1.0), (1.0, 0.0)),
        ("W", (hx - 11.04, hy), (0.0, -1.0), (-1.0, 0.0)),
    ):
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (fx, fy),
            hz + 4.8,
            hz + 8.8,
            2.4,
            5.2,
            0.02,
            (tx, ty),
            (nx, ny),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=11,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (fx, fy, hz + 8.8),
            (tx, ty),
            (nx, ny),
            span_w=5.4,
            arch_h=2.5,
            ring_thickness=0.45,
            offset_n0=-0.02,
            offset_n1=0.26,
            num_voussoirs=11,
        )
        # Cycle 2 Detail: Twin tall flanking arched windows on each facade (`off_t in (-6.6, 6.6)`)
        for fl_off in (-6.6, 6.6):
            wfx = fx + tx * fl_off
            wfy = fy + ty * fl_off
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (wfx, wfy),
                hz + 4.8,
                hz + 8.4,
                0.95,
                1.85,
                0.02,
                (tx, ty),
                (nx, ny),
                uv_rect=WOOD_DOOR_UV["window_glass"],
                num_voussoirs=7,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (wfx, wfy, hz + 8.4),
                (tx, ty),
                (nx, ny),
                span_w=2.0,
                arch_h=1.0,
                ring_thickness=0.32,
                offset_n0=-0.02,
                offset_n1=0.20,
                num_voussoirs=7,
            )
        if side_dir == "N":
            # Triple arched stone entrance porch facing Hurva Square (grounded down to `hz - 1.8m`)
            collector.add_box("Buildings_Walls", "mat_stone_mizzi", (hx, hy + 12.6, hz + 1.4), (12.0, 3.2, 6.4), uv_scale=0.25)
            for px_off in (-3.4, 0.0, 3.4):
                add_arched_opening_panel(
                    collector,
                    "Buildings_Openings",
                    "mat_wood_doors",
                    (hx + px_off, hy + 14.22),
                    hz - 1.2,
                    hz + 2.5,
                    1.1,
                    2.4,
                    0.02,
                    (-1.0, 0.0),
                    (0.0, 1.0),
                    uv_rect=WOOD_DOOR_UV["door_wood"],
                    num_voussoirs=7,
                    add_wall_spandrel=False,
                )
                add_arch_voussoirs(
                    collector,
                    "Buildings_Details",
                    "mat_stone_boss",
                    (hx + px_off, hy + 14.22, hz + 2.5),
                    (-1.0, 0.0),
                    (0.0, 1.0),
                    span_w=2.55,
                    arch_h=1.15,
                    ring_thickness=0.34,
                    offset_n0=-0.02,
                    offset_n1=0.20,
                    num_voussoirs=7,
                )

    # Tall 12-windowed cylindrical stone drum (`z = +12.8 .. +18.2m`, `radius = 8.2m`)
    collector.add_branch_tube(
        "Buildings_Walls",
        "mat_stone_mizzi",
        (hx, hy, hz + 12.8),
        (hx, hy, hz + 18.2),
        8.2,
        8.0,
        segments=16,
        cap_end=True,
    )
    for ang_deg in range(0, 360, 30):
        ang = math.radians(ang_deg)
        wx = hx + math.cos(ang) * 8.12
        wy = hy + math.sin(ang) * 8.12
        tx, ty = -math.sin(ang), math.cos(ang)
        nx, ny = math.cos(ang), math.sin(ang)
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (wx, wy),
            hz + 13.6,
            hz + 16.4,
            0.7,
            1.4,
            0.02,
            (tx, ty),
            (nx, ny),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=5,
            add_wall_spandrel=False,
        )
    # Stone balustrade ring + dark wrought-iron walkway railing at top of drum (`z = +18.0 .. +19.15m`)
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_stone_mizzi",
        (hx, hy, hz + 17.95),
        (hx, hy, hz + 18.55),
        8.35,
        8.35,
        segments=16,
        cap_end=True,
    )
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_iron_dark",
        (hx, hy, hz + 18.95),
        (hx, hy, hz + 19.12),
        8.42,
        8.42,
        segments=20,
    )
    for r_deg in range(0, 360, 15):
        r_rad = math.radians(r_deg)
        collector.add_box(
            "Buildings_Details",
            "mat_iron_dark",
            (hx + math.cos(r_rad) * 8.38, hy + math.sin(r_rad) * 8.38, hz + 18.82),
            (0.08, 0.08, 0.65),
        )
    # Soaring hemispherical White/Cream Hurva Dome (`mat_stone_mizzi`, `z = +18.2 .. +25.6m`)
    collector.add_dome(
        "Buildings_Details",
        "mat_stone_mizzi",
        (hx, hy, hz + 18.2),
        radius=8.1,
        height=7.4,
        rings=10,
        segments=20,
    )
    collector.add_box("Buildings_Details", "mat_dome_gold", (hx, hy, hz + 26.0), (0.22, 0.22, 1.6), uv_scale=0.5)

    # Cycle 2 Hurva Square Furnishings: Stone Planters, Bollards & Café Parasols (`y = -124..-102`)
    for pl_x, pl_y in ((338.0, -118.0), (355.0, -118.0), (364.0, -114.0)):
        pl_z = street_elevation_z(pl_x, pl_y)
        collector.add_box("StreetProps", "mat_stone_mizzi", (pl_x, pl_y, pl_z + 0.42), (2.2, 1.4, 0.84), uv_scale=0.35)
        collector.add_box("Trees_Foliage", "mat_foliage_olive", (pl_x, pl_y, pl_z + 0.95), (1.8, 1.1, 0.45), uv_scale=0.6)
    for b_idx in range(5):
        bx_h = 338.0 + b_idx * 4.2
        by_h = -123.5
        build_bollard(collector, bx_h, by_h, street_elevation_z(bx_h, by_h))
    for umb_x, umb_y in ((334.5, -106.0), (361.5, -106.0)):
        uz = street_elevation_z(umb_x, umb_y)
        collector.add_branch_tube("StreetProps", "mat_iron_dark", (umb_x, umb_y, uz), (umb_x, umb_y, uz + 2.6), 0.06, 0.05, segments=6)
        collector.add_branch_tube("StreetProps", "mat_stone_mizzi", (umb_x, umb_y, uz + 2.35), (umb_x, umb_y, uz + 2.95), 1.85, 0.12, segments=8, cap_end=True)

    # 2. Golden Menorah Monument (`(378.0, -108.0)`) on the eastern terrace of Hurva Square
    mx, my = 378.0, -108.0
    mz = street_elevation_z(mx, my)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (mx, my, mz + 0.65), (3.6, 3.6, 1.3), uv_scale=0.30)
    collector.add_box("Buildings_Details", "mat_dome_gold", (mx, my, mz + 2.2), (0.28, 0.28, 2.0), uv_scale=0.5)
    for arm_w in (0.9, 1.6, 2.3):
        collector.add_box("Buildings_Details", "mat_dome_gold", (mx, my, mz + 2.0 + arm_w * 0.25), (arm_w, 0.16, 0.16), uv_scale=0.5)
        collector.add_box("Buildings_Details", "mat_dome_gold", (mx - arm_w * 0.5, my, mz + 2.55), (0.16, 0.16, 0.9), uv_scale=0.5)
        collector.add_box("Buildings_Details", "mat_dome_gold", (mx + arm_w * 0.5, my, mz + 2.55), (0.16, 0.16, 0.9), uv_scale=0.5)

    # 3. SouthernRoman Cardo Colonnade (`x = 308.0, y = -192..-58`)
    for cy in range(-188, -56, 7):
        cz = street_elevation_z(308.0, float(cy))
        for col_x in (304.2, 311.8):
            # Monolithic Roman limestone column shaft + grounded plinth (`cz - 1.2m .. cz + 0.7m`) + Attic torus base + multi-tier Corinthian capital
            collector.add_box("Buildings_Details", "mat_stone_boss", (col_x, float(cy), cz - 0.25), (0.95, 0.95, 1.9), uv_scale=0.35)
            collector.add_branch_tube("Buildings_Details", "mat_stone_mizzi", (col_x, float(cy), cz + 0.68), (col_x, float(cy), cz + 0.92), 0.46, 0.39, segments=10, cap_end=True)
            collector.add_branch_tube(
                "Buildings_Details",
                "mat_stone_mizzi",
                (col_x, float(cy), cz + 0.88),
                (col_x, float(cy), cz + 4.88),
                0.38,
                0.34,
                segments=10,
                cap_end=True,
            )
            # Flaring Corinthian bell capital (`r=0.34 -> 0.52`) + carved abacus slab
            collector.add_branch_tube("Buildings_Details", "mat_stone_mizzi", (col_x, float(cy), cz + 4.85), (col_x, float(cy), cz + 5.32), 0.35, 0.52, segments=10, cap_end=True)
            collector.add_box("Buildings_Details", "mat_stone_mizzi", (col_x, float(cy), cz + 5.42), (0.96, 0.96, 0.24), uv_scale=0.35)
        # Continuous stone architrave beam, western retaining wall with iron railing, and full-height eastern Byzantine shop arcade supporting the roof canopy
        if cy < -63:
            cz_next = street_elevation_z(308.0, float(cy + 7))
            z_mid = 0.5 * (cz + cz_next)
            z_beam = z_mid + 5.8
            for col_x in (304.2, 311.8):
                collector.add_box("Buildings_Details", "mat_stone_mizzi", (col_x, float(cy) + 3.5, z_beam), (0.78, 7.2, 0.65), uv_scale=0.28, bottom_face=True)
            # Western archaeological retaining wall (`x = 301.2`) + black wrought-iron upper safety railing
            collector.add_box("Buildings_Walls", "mat_stone_boss", (301.2, float(cy) + 3.5, z_mid + 1.4), (1.2, 7.2, 3.8), uv_scale=0.26)
            collector.add_box("StreetProps", "mat_iron_dark", (301.2, float(cy) + 3.5, z_mid + 3.95), (0.08, 7.2, 0.08))
            collector.add_box("StreetProps", "mat_iron_dark", (301.2, float(cy) + 0.1, z_mid + 3.62), (0.08, 0.08, 0.68))
            # Full-height eastern Byzantine shop wall (`x = 315.2`, reaching `z_beam + 0.55m` so the terracotta portico roof rests directly on it with ZERO sky gap!)
            collector.add_box("Buildings_Walls", "mat_stone_boss", (315.2, float(cy) + 3.5, z_mid + 2.75), (1.3, 7.2, 7.5), uv_scale=0.24)
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (314.54, float(cy) + 3.5),
                z_mid,
                z_mid + 2.5,
                1.10,
                2.20,
                0.01,
                (0.0, -1.0),
                (-1.0, 0.0),
                uv_rect=WOOD_DOOR_UV["souq_open" if (cy // 7) % 2 == 0 else "door_wood"],
                num_voussoirs=7,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_mizzi",
                (314.54, float(cy) + 3.5, z_mid + 2.5),
                (0.0, -1.0),
                (-1.0, 0.0),
                span_w=2.20,
                arch_h=1.10,
                ring_thickness=0.32,
                offset_n0=-0.04,
                offset_n1=0.20,
                num_voussoirs=7,
            )
            # Sloped terracotta & timber roof canopy over the eastern colonnade aisle (`x = 311.8..315.6`)
            collector.add_box("Buildings_Details", "mat_roof_tile", (313.6, float(cy) + 3.5, z_beam + 0.45), (3.9, 7.2, 0.35), uv_scale=0.32, bottom_face=True)

    # 4. Four Sephardic Synagogues (`(366.0, -238.0)`), Tiferet Yisrael (`(492.0, -165.0)`), Batei Mahaseh & Broad Wall (`(335.0, -78.0)`)
    sx, sy = 366.0, -238.0
    sz = street_elevation_z(sx, sy)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar_aged", (sx, sy, sz + 3.8), (24.0, 20.0, 7.6), uv_scale=0.24)
    # Mid-facade carved stone stringcourse molding + upper cornice
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (sx, sy, sz + 5.05), (24.3, 20.3, 0.24), uv_scale=0.32)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (sx, sy, sz + 7.75), (24.6, 20.6, 0.35), uv_scale=0.30)
    # Western courtyard facade facing `(344.0, -238.0)` with zero-gap arched synagogue doors, windows, iron grilles, upper clerestory windows & plaques
    for wy_off, is_door in ((-6.5, False), (-2.2, True), (2.2, True), (6.5, False)):
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (sx - 12.01, sy + wy_off),
            sz + (0.0 if is_door else 1.2),
            sz + 2.6,
            1.10,
            2.20,
            0.01,
            (0.0, -1.0),
            (-1.0, 0.0),
            uv_rect=WOOD_DOOR_UV["door_wood" if is_door else "window_glass"],
            num_voussoirs=7,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (sx - 12.01, sy + wy_off, sz + 2.6),
            (0.0, -1.0),
            (-1.0, 0.0),
            span_w=2.20,
            arch_h=1.10,
            ring_thickness=0.34,
            offset_n0=-0.04,
            offset_n1=0.22,
            num_voussoirs=7,
        )
        # Carved Hebrew stone inscription plaque above each synagogue portal/window
        collector.add_box(
            "Buildings_Details",
            "mat_stone_mizzi",
            (sx - 12.10, sy + wy_off, sz + 4.35),
            (0.18, 1.45, 0.52),
            uv_scale=0.35,
        )
        # Upper-storey arched clerestory synagogue window (`z = sz + 5.45 .. 6.85m`)
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (sx - 12.01, sy + wy_off),
            sz + 5.45,
            sz + 6.35,
            0.62,
            1.24,
            0.01,
            (0.0, -1.0),
            (-1.0, 0.0),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=5,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (sx - 12.01, sy + wy_off, sz + 6.35),
            (0.0, -1.0),
            (-1.0, 0.0),
            span_w=1.24,
            arch_h=0.62,
            ring_thickness=0.25,
            offset_n0=-0.04,
            offset_n1=0.18,
            num_voussoirs=5,
        )
        if is_door:
            # Stepped Jerusalem stone entrance threshold before each synagogue portal
            collector.add_box("Ground_Steps", "mat_stone_mizzi", (sx - 12.45, sy + wy_off, sz + 0.09), (0.95, 2.55, 0.18), uv_scale=0.35)
        else:
            collector.add_box(
                "Buildings_Details",
                "mat_stone_mizzi",
                (sx - 12.14, sy + wy_off, sz + 1.12),
                (0.24, 2.45, 0.16),
                uv_scale=0.35,
            )
            # 3D wrought-iron security grille bars on the flanking synagogue windows
            for gy_b in (-0.65, 0.0, 0.65):
                collector.add_box("Buildings_Openings", "mat_iron_dark", (sx - 12.06, sy + wy_off + gy_b, sz + 2.05), (0.05, 0.05, 1.85))
            for gz_b in (1.65, 2.15, 2.65):
                collector.add_box("Buildings_Openings", "mat_iron_dark", (sx - 12.06, sy + wy_off, sz + gz_b), (0.05, 2.05, 0.05))
    # Wall-mounted wrought-iron heritage lanterns on the synagogue piers (plus off-center courtyard lamps that never block the center sightline!)
    for wy_lan in (-4.35, 0.0, 4.35):
        build_wall_lantern(collector, (sx - 12.08, sy + wy_lan, sz + 3.45), (0.0, -1.0), (-1.0, 0.0))
    # Courtyard stone planter beds, benches & olive trees in front of the Four Sephardic Synagogues (`x = 350.5`)
    for py_pl in (-5.2, 5.2):
        pz_pl = street_elevation_z(350.5, sy + py_pl)
        collector.add_box("StreetProps", "mat_stone_mizzi", (350.5, sy + py_pl, pz_pl + 0.42), (1.8, 2.4, 0.84), uv_scale=0.35)
        collector.add_box("Trees_Foliage", "mat_foliage_olive", (350.5, sy + py_pl, pz_pl + 0.95), (1.4, 1.9, 0.48), uv_scale=0.6)
        build_street_lamp(collector, 350.5, sy + py_pl * 1.55, street_elevation_z(350.5, sy + py_pl * 1.55), height=4.1)
    for dx in (-5.5, 5.5):
        for dy in (-4.5, 4.5):
            collector.add_branch_tube(
                "Buildings_Walls",
                "mat_stone_mizzi",
                (sx + dx, sy + dy, sz + 7.6),
                (sx + dx, sy + dy, sz + 9.2),
                3.5,
                3.4,
                segments=12,
                cap_end=True,
            )
            collector.add_dome(
                "Buildings_Details",
                "mat_dome_lead",
                (sx + dx, sy + dy, sz + 9.1),
                radius=3.4,
                height=2.8,
                rings=6,
                segments=12,
            )

    # Tiferet Yisrael Synagogue (`Nisanov Synagogue` at `(492.0, -165.0)`)
    tx_s, ty_s = 492.0, -165.0
    tz_s = street_elevation_z(tx_s, ty_s)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (tx_s, ty_s, tz_s + 6.2), (18.0, 18.0, 14.4), uv_scale=0.22)
    collector.add_box("Buildings_Details", "mat_stone_boss", (tx_s, ty_s, tz_s + 4.35), (18.4, 18.4, 0.28), uv_scale=0.30)
    collector.add_box("Buildings_Details", "mat_stone_boss", (tx_s, ty_s, tz_s + 13.5), (18.6, 18.6, 0.45), uv_scale=0.30)
    # Projecting Triple-Arched Stone Entrance Portico & Upper Balustrade Terrace (`x = tx_s - 9.0 .. tx_s - 10.8`)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (tx_s - 9.9, ty_s, tz_s + 4.25), (1.9, 13.6, 0.34), uv_scale=0.30, bottom_face=True)
    collector.add_box("Buildings_Details", "mat_stone_boss", (tx_s - 10.65, ty_s, tz_s + 4.65), (0.34, 13.6, 0.52), uv_scale=0.32)
    for col_ty in (-6.3, -2.1, 2.1, 6.3):
        collector.add_branch_tube(
            "Buildings_Details",
            "mat_stone_mizzi",
            (tx_s - 10.55, ty_s + col_ty, tz_s),
            (tx_s - 10.55, ty_s + col_ty, tz_s + 4.1),
            0.28,
            0.25,
            segments=10,
            cap_end=True,
        )
        collector.add_box("Buildings_Details", "mat_stone_boss", (tx_s - 10.55, ty_s + col_ty, tz_s + 3.05), (0.64, 0.64, 0.28), uv_scale=0.35)
    for arch_ty in (-4.2, 0.0, 4.2):
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (tx_s - 10.55, ty_s + arch_ty, tz_s + 3.05),
            (0.0, -1.0),
            (-1.0, 0.0),
            span_w=3.65,
            arch_h=1.05,
            ring_thickness=0.30,
            offset_n0=-0.18,
            offset_n1=0.18,
            num_voussoirs=9,
        )
    for wy_t in (-4.8, 0.0, 4.8):
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (tx_s - 9.01, ty_s + wy_t),
            tz_s + (0.0 if wy_t == 0.0 else 4.8),
            tz_s + (3.2 if wy_t == 0.0 else 8.6),
            1.30,
            2.60,
            0.01,
            (0.0, -1.0),
            (-1.0, 0.0),
            uv_rect=WOOD_DOOR_UV["door_wood" if wy_t == 0.0 else "window_glass"],
            num_voussoirs=9,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (tx_s - 9.01, ty_s + wy_t, tz_s + (3.2 if wy_t == 0.0 else 8.6)),
            (0.0, -1.0),
            (-1.0, 0.0),
            span_w=2.60,
            arch_h=1.30,
            ring_thickness=0.38,
            offset_n0=-0.04,
            offset_n1=0.24,
            num_voussoirs=9,
        )
        # Carved circular stone rosette / oculus window in the upper facade above the door and flanking bays
        oc_z = tz_s + (9.2 if wy_t == 0.0 else 2.2)
        collector.add_branch_tube(
            "Buildings_Details",
            "mat_stone_boss",
            (tx_s - 9.0, ty_s + wy_t, oc_z),
            (tx_s - 9.22, ty_s + wy_t, oc_z),
            0.85 if wy_t == 0.0 else 0.62,
            0.80 if wy_t == 0.0 else 0.58,
            segments=12,
        )
        collector.add_branch_tube(
            "Buildings_Openings",
            "mat_iron_dark",
            (tx_s - 9.05, ty_s + wy_t, oc_z),
            (tx_s - 9.16, ty_s + wy_t, oc_z),
            0.58 if wy_t == 0.0 else 0.42,
            0.55 if wy_t == 0.0 else 0.40,
            segments=12,
            cap_end=True,
        )
    collector.add_branch_tube(
        "Buildings_Walls",
        "mat_stone_mizzi",
        (tx_s, ty_s, tz_s + 13.4),
        (tx_s, ty_s, tz_s + 17.6),
        6.4,
        6.2,
        segments=14,
        cap_end=True,
    )
    collector.add_dome(
        "Buildings_Details",
        "mat_stone_mizzi",
        (tx_s, ty_s, tz_s + 17.5),
        radius=6.25,
        height=5.8,
        rings=8,
        segments=16,
    )
    for b_ty in (-171.0, -165.0, -159.0):
        build_bollard(collector, 476.5, b_ty, street_elevation_z(476.5, b_ty))
    for lamp_ty in (-173.5, -156.5):
        build_street_lamp(collector, 476.5, lamp_ty, street_elevation_z(476.5, lamp_ty), height=4.2)

    # Batei Mahaseh 19th-Century Neo-Classical Arched Hospice Facade (`(444.0, -322.0)`, facing NW toward `(430, -310)`)
    bm_x, bm_y = 444.0, -322.0
    bm_z = street_elevation_z(bm_x, bm_y)
    bm_rad = math.radians(40.0)
    tx_bm, ty_bm = math.cos(bm_rad), math.sin(bm_rad)
    nx_bm, ny_bm = -ty_bm, tx_bm  # Points NW `(-0.6428, +0.7660)` directly toward the square at `(430, -310)`
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (bm_x, bm_y, bm_z + 3.8), (18.0, 9.0, 9.6), rot_z=bm_rad, uv_scale=0.24)
    # Mid-facade projecting stone stringcourse (`bm_z + 4.35m`) & upper cornice (`bm_z + 8.75m`)
    collector.add_box("Buildings_Details", "mat_stone_boss", (bm_x, bm_y, bm_z + 4.35), (18.4, 9.4, 0.28), rot_z=bm_rad, uv_scale=0.30)
    collector.add_box("Buildings_Details", "mat_stone_boss", (bm_x, bm_y, bm_z + 8.75), (18.6, 9.6, 0.42), rot_z=bm_rad, uv_scale=0.30)
    # Central Rooftop Neoclassical Attica Pediment & Carved Rothschild Coat-of-Arms Stone Emblem Plaque
    collector.add_box(
        "Buildings_Walls",
        "mat_stone_mizzi",
        (bm_x + nx_bm * 4.15, bm_y + ny_bm * 4.15, bm_z + 9.45),
        (6.8, 0.85, 1.35),
        rot_z=bm_rad,
        uv_scale=0.28,
    )
    collector.add_box(
        "Buildings_Details",
        "mat_stone_boss",
        (bm_x + nx_bm * 4.20, bm_y + ny_bm * 4.20, bm_z + 10.22),
        (7.2, 1.05, 0.28),
        rot_z=bm_rad,
        uv_scale=0.32,
    )
    collector.add_box(
        "Buildings_Details",
        "mat_stone_boss",
        (bm_x + nx_bm * 4.62, bm_y + ny_bm * 4.62, bm_z + 9.45),
        (1.65, 0.16, 0.78),
        rot_z=bm_rad,
        uv_scale=0.35,
    )
    collector.add_box(
        "Buildings_Details",
        "mat_dome_gold",
        (bm_x + nx_bm * 4.68, bm_y + ny_bm * 4.68, bm_z + 9.45),
        (0.65, 0.12, 0.48),
        rot_z=bm_rad,
        uv_scale=0.5,
    )
    # 5-Bay Neo-Classical Ground-Floor Stone Arcade & 5 Upper-Floor Shuttered Arched Windows with Wrought-Iron Balconies
    f_bm_x = bm_x + nx_bm * 4.51
    f_bm_y = bm_y + ny_bm * 4.51
    for bay_t in (-6.8, -3.4, 0.0, 3.4, 6.8):
        bx_bm = f_bm_x + tx_bm * bay_t
        by_bm = f_bm_y + ty_bm * bay_t
        is_bm_door = abs(bay_t) < 4.0
        # Ground-floor arched arcade portal / window
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (bx_bm, by_bm),
            bm_z + (0.0 if is_bm_door else 0.95),
            bm_z + 2.55,
            1.05,
            2.10,
            0.01,
            (-tx_bm, -ty_bm),
            (nx_bm, ny_bm),
            uv_rect=WOOD_DOOR_UV["door_wood" if is_bm_door else "window_glass"],
            num_voussoirs=7,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (bx_bm, by_bm, bm_z + 2.55),
            (-tx_bm, -ty_bm),
            (nx_bm, ny_bm),
            span_w=2.10,
            arch_h=1.05,
            ring_thickness=0.32,
            offset_n0=-0.04,
            offset_n1=0.22,
            num_voussoirs=7,
        )
        # Upper-floor arched window (`z = bm_z + 4.95 .. 6.45m`) with stone sill, green louvered shutters & iron balcony railing
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (bx_bm, by_bm),
            bm_z + 4.95,
            bm_z + 6.45,
            0.72,
            1.44,
            0.01,
            (-tx_bm, -ty_bm),
            (nx_bm, ny_bm),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=5,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (bx_bm, by_bm, bm_z + 6.45),
            (-tx_bm, -ty_bm),
            (nx_bm, ny_bm),
            span_w=1.44,
            arch_h=0.72,
            ring_thickness=0.26,
            offset_n0=-0.04,
            offset_n1=0.18,
            num_voussoirs=5,
        )
        collector.add_box(
            "Buildings_Details",
            "mat_stone_mizzi",
            (bx_bm + nx_bm * 0.24, by_bm + ny_bm * 0.24, bm_z + 4.86),
            (1.85, 0.48, 0.16),
            rot_z=bm_rad,
            uv_scale=0.35,
            bottom_face=True,
        )
        collector.add_box(
            "StreetProps",
            "mat_iron_dark",
            (bx_bm + nx_bm * 0.44, by_bm + ny_bm * 0.44, bm_z + 5.38),
            (1.75, 0.06, 0.78),
            rot_z=bm_rad,
        )
        for sh_bm in (-0.92, 0.92):
            collector.add_box(
                "Buildings_Openings",
                "mat_wood_doors",
                (bx_bm + tx_bm * sh_bm + nx_bm * 0.08, by_bm + ty_bm * sh_bm + ny_bm * 0.08, bm_z + 5.75),
                (0.36, 0.08, 1.52),
                rot_z=bm_rad,
                uv_rect=WOOD_DOOR_UV["shutter_louvre"],
            )
    # Courtyard Olive Tree, Stone Planter Bed, Wrought-Iron & Stone Benches, Bollards & Lamp flanking Batei Mahaseh Square
    build_olive_tree(collector, 431.2, -317.8, street_elevation_z(431.2, -317.8), height=5.8, seed=625, with_kerb=True)
    build_street_lamp(collector, 441.5, -313.5, street_elevation_z(441.5, -313.5), height=4.2)
    for bench_xy in ((435.5, -313.8), (439.2, -316.8)):
        bz_bn = street_elevation_z(bench_xy[0], bench_xy[1])
        collector.add_box("StreetProps", "mat_stone_boss", (bench_xy[0], bench_xy[1], bz_bn + 0.26), (1.85, 0.58, 0.52), rot_z=bm_rad, uv_scale=0.35)
    for bol_xy in ((432.5, -315.5), (435.0, -317.5), (437.5, -319.5)):
        build_bollard(collector, bol_xy[0], bol_xy[1], street_elevation_z(bol_xy[0], bol_xy[1]))

    # Hezekiah's Broad Wall (`HaChoma HaRechava`, 7m wide First Temple stone fortification at `(335.0, -78.0)`)
    bwx, bwy = 335.0, -78.0
    bwz = street_elevation_z(bwx, bwy)
    # Massive 7m-wide First Temple unhewn boulder fortification courses grounded down to `bwz - 2.5m` (zero floating!)
    collector.add_box("Buildings_Walls", "mat_stone_rubble", (bwx, bwy, bwz - 0.1), (7.2, 18.2, 3.2), uv_scale=0.16)
    for b_idx, (bx_off, by_off, bsx, bsy, bsz) in enumerate((
        (-1.4, -5.2, 3.8, 5.4, 3.5),
        (1.5, -4.6, 3.6, 5.8, 3.4),
        (-1.2, 1.2, 4.1, 6.2, 3.6),
        (1.6, 1.8, 3.5, 5.6, 3.3),
        (0.0, 6.2, 6.4, 4.8, 3.2),
    )):
        collector.add_box("Buildings_Walls", "mat_stone_boss", (bwx + bx_off, bwy + by_off, bwz + 0.1), (bsx, bsy, bsz), uv_scale=0.18)
    # Deeply grounded dressed-stone archaeological pit retaining wall + black wrought-iron viewing railing (`bwz - 2.2m .. bwz + 1.45m`)
    for px_w, py_w, sx_w, sy_w in (
        (bwx, bwy - 10.6, 11.4, 0.58),
        (bwx, bwy + 10.6, 11.4, 0.58),
        (bwx - 5.5, bwy, 0.58, 21.6),
        (bwx + 5.5, bwy, 0.58, 21.6),
    ):
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (px_w, py_w, bwz - 0.65), (sx_w, sy_w, 3.1), uv_scale=0.32)
        collector.add_box("StreetProps", "mat_iron_dark", (px_w, py_w, bwz + 1.38), (sx_w, sy_w * 0.25 if sy_w < 1.0 else 0.08, 0.08))
    for r_post_x in (-5.2, -2.6, 0.0, 2.6, 5.2):
        collector.add_box("StreetProps", "mat_iron_dark", (bwx + r_post_x, bwy - 10.6, bwz + 1.12), (0.08, 0.08, 0.55))


def build_region_r05_jewish_quarter(
    collector: MeshCollector,
    osm_data: dict,
    sign_atlas: Dict[str, dict],
) -> int:
    """
    Builds Region 5 (`r05_jewish_quarter`):
      - Continuous stone-paved ground surface across `[280..560] x [-425..-45]`
      - Southern Ottoman City Wall (`Segment 2` part 1: `(286.5, -407.1)` -> `(560.0, -224.0)`)
      - Hurva Synagogue & Hurva Square, Golden Menorah Monument, Southern Roman Cardo, Four Sephardic Synagogues, Broad Wall
      - All valid `r05_jewish_quarter` OSM buildings, olive/cypress trees, and plaza props
    """
    build_region_ground_grid(collector, 280.0, 560.0, -425.0, -45.0, step=5.5)

    # 1. Southern Ottoman City Wall (`Segment 2` part 1 of `5862586`)
    south_wall_pts = [
        (286.5, -407.1),
        (292.1, -389.2),
        (314.4, -392.4),
        (320.2, -372.4),
        (356.1, -380.4),
        (381.9, -374.1),
        (430.9, -342.4),
        (441.0, -359.0),
        (461.4, -346.6),
        (445.0, -315.4),
        (452.3, -298.2),
        (514.2, -267.9),
        (526.3, -257.3),
        (560.0, -224.0),
    ]
    build_ottoman_wall_polyline(
        collector,
        south_wall_pts,
        wall_height=12.0,
        thickness=3.1,
        bastion_every=3,
    )

    # 2. Hurva Synagogue, Hurva Square, Southern Roman Cardo & Four Sephardic Synagogues
    _build_hurva_synagogue_and_roman_cardo(collector, sign_atlas)

    # 3. OSM Buildings in `r05_jewish_quarter`
    clear_scene_building_rings()
    osm_buildings = [
        b for b in osm_data.get("buildings", [])
        if b.get("region") == "r05_jewish_quarter"
    ]
    valid_blds = []
    for b in osm_buildings:
        osm_id = int(b.get("osm_id", 0))
        if osm_id in (5862586, 7880713, 45081545):
            continue
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        ring = b.get("ring", [])
        if len(ring) < 3:
            continue
        rxs = [float(p[0]) for p in ring]
        rys = [float(p[1]) for p in ring]
        if (max(rxs) - min(rxs)) > 65.0 or (max(rys) - min(rys)) > 65.0:
            continue
        # Keep Hurva Synagogue & Hurva Square (`x = 326..388, y = -160..-92`) clear
        if _ring_overlaps_rect(ring, 326.0, -160.0, 388.0, -92.0):
            continue
        # Keep Southern Roman Cardo corridor (`x = 296..320, y = -198..-52`) clear
        if _ring_overlaps_rect(ring, 296.0, -198.0, 320.0, -52.0):
            continue
        # Keep Four Sephardic Synagogues & Square (`x = 330..384, y = -256..-220`) clear
        if _ring_overlaps_rect(ring, 330.0, -256.0, 384.0, -220.0):
            continue
        # Keep Hezekiah's Broad Wall & Plaza (`x = 320..350, y = -104..-62`) clear
        if _ring_overlaps_rect(ring, 320.0, -104.0, 350.0, -62.0):
            continue
        # Keep Tiferet Yisrael & Karaite Street (`x = 452..506, y = -180..-150`) clear
        if _ring_overlaps_rect(ring, 452.0, -180.0, 506.0, -150.0):
            continue
        # Keep Batei Mahaseh Square (`x = 418..455, y = -325..-295`) clear
        if _ring_overlaps_rect(ring, 418.0, -325.0, 455.0, -295.0):
            continue
        valid_blds.append(b)
        register_scene_building_ring(osm_id, ring)

    built_count = 0
    for idx, b in enumerate(valid_blds):
        ring = b.get("ring", [])
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        bz = street_elevation_z(cx, cy)
        lvls = max(2, min(3, int(b.get("levels", 2))))
        build_building_from_footprint(
            collector,
            ring,
            base_z=bz,
            levels=lvls,
            sign_atlas=sign_atlas,
            building_id=int(b.get("osm_id", idx)),
            is_hero=False,
        )
        built_count += 1

    # 4. Hurva Square & Jewish Quarter Trees and Street Lamps
    jq_trees = [
        (336.0, -112.0, "olive", 6.5, 601),
        (358.0, -112.0, "olive", 6.8, 602),
        (368.0, -122.0, "cypress", 10.5, 603),
        (352.0, -218.0, "olive", 6.2, 604),
        (415.0, -165.0, "cypress", 11.0, 605),
        (465.0, -195.0, "olive", 6.4, 606),
    ]
    for tx, ty, sp, th, seed in jq_trees:
        tz = street_elevation_z(tx, ty)
        if sp == "cypress":
            build_cypress_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)
        else:
            build_olive_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)

    for lx, ly in ((340.0, -104.0), (354.0, -104.0), (372.0, -108.0), (308.0, -120.0), (308.0, -160.0)):
        build_street_lamp(collector, lx, ly, street_elevation_z(lx, ly), height=4.3)

    return built_count


# ==============================================================================
# 7. REGION 6: WESTERN WALL (KOTEL), DUNG GATE & DAVIDSON ARCHAEOLOGICAL PARK (`r06_western_wall_dung_gate`)
# ==============================================================================

def _build_western_wall_and_plaza(collector: MeshCollector, sign_atlas: Dict[str, dict]) -> None:
    """
    Builds:
      1. The Monumental Herodian Western Wall (`HaKotel HaMa'aravi` at `x = 666.0, y = -85..65`)
         with massive drafted-margin Herodian bottom courses (`mat_stone_boss`), middle Umayyad/Fatimid
         courses (`mat_stone_ashlar`), upper Ottoman courses (`mat_stone_ashlar_aged`), and caper bush tufts.
      2. The Western Wall Prayer Plaza (`x = 575..666, y = -45..58`), Men's & Women's prayer partition
         (`mechitza`), white Torah reading tables, Wilson's Arch (`y = 46..64`), and the Mughrabi Bridge ramp.
    """
    wall_x = 666.0
    pz = street_elevation_z(645.0, 12.0)

    # 1. Monumental Herodian Western Wall (`x = 666.0, y = -85..68`, `z = pz - 2.0 .. pz + 19.5m`)
    # Bottom 7 Herodian bossed stone courses (`z = pz - 2.0 .. pz + 8.6m`)
    collector.add_box(
        "CityWalls",
        "mat_stone_boss",
        (wall_x + 2.2, -8.5, pz + 3.3),
        (4.4, 154.0, 10.6),
        uv_scale=0.14,
    )
    # Middle Umayyad/Fatimid dressed stone courses (`z = pz + 8.6 .. pz + 14.8m`)
    collector.add_box(
        "CityWalls",
        "mat_stone_ashlar",
        (wall_x + 2.3, -8.5, pz + 11.7),
        (4.2, 154.0, 6.2),
        uv_scale=0.22,
    )
    # Upper Ottoman courses & parapet (`z = pz + 14.8 .. pz + 19.5m`)
    collector.add_box(
        "CityWalls",
        "mat_stone_ashlar_aged",
        (wall_x + 2.4, -8.5, pz + 17.15),
        (4.0, 154.0, 4.7),
        uv_scale=0.26,
    )
    # Organic multi-plane caper bush tufts (`Capparis spinosa`) growing out of crevices in the Herodian stones
    for cy_bush, cz_bush in ((-18.0, 5.2), (-4.0, 6.8), (11.0, 4.9), (24.0, 7.1), (38.0, 5.6)):
        for plane_ang, p_dz in ((-0.35, -0.15), (0.0, 0.05), (0.35, -0.10)):
            ca, sa = math.cos(plane_ang), math.sin(plane_ang)
            p0 = (wall_x - 0.12 - sa * 0.55, cy_bush - ca * 0.65)
            p1 = (wall_x - 0.12 + sa * 0.55, cy_bush + ca * 0.65)
            collector.add_oriented_box(
                "Trees_Foliage",
                "mat_foliage_olive",
                p0,
                p1,
                pz + cz_bush - 0.45 + p_dz,
                pz + cz_bush + 0.45 + p_dz,
                -0.02,
                0.02,
                (-ca, sa),
                uv_scale=0.8,
                cap_back=True,
            )

    # 2. Wilson's Arch (`HaKeshet shel Wilson` on the north side of the prayer plaza, `y = 46..64`)
    #    & Mamluk Tankiziyya Madrasa rising above the arch (`z = pz + 10.4 .. pz + 17.6m`)
    collector.add_box(
        "Buildings_Walls",
        "mat_stone_boss",
        (654.0, 56.0, pz + 5.2),
        (24.0, 18.0, 10.4),
        uv_scale=0.20,
    )
    # Upper Mamluk Tankiziyya building & arched windows overlooking the Western Wall Plaza
    collector.add_box(
        "Buildings_Walls",
        "mat_stone_ashlar",
        (654.0, 56.5, pz + 14.0),
        (23.6, 17.0, 7.2),
        uv_scale=0.24,
    )
    collector.add_box(
        "Buildings_Details",
        "mat_stone_mizzi",
        (654.0, 56.5, pz + 17.8),
        (24.2, 17.6, 0.45),
        uv_scale=0.30,
    )
    for wx_off in (-6.5, 0.0, 6.5):
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (654.0 + wx_off, 47.96),
            pz + 11.6,
            pz + 14.8,
            0.9,
            1.9,
            0.02,
            (1.0, 0.0),
            (0.0, -1.0),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=7,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (654.0 + wx_off, 47.96, pz + 14.8),
            (1.0, 0.0),
            (0.0, -1.0),
            span_w=2.05,
            arch_h=0.95,
            ring_thickness=0.32,
            offset_n0=-0.02,
            offset_n1=0.18,
            num_voussoirs=7,
        )
    # Wilson's Arch deep vaulted opening (`arch_spring_z = pz + 4.2`, matching `add_arch_voussoirs`!)
    add_arched_opening_panel(
        collector,
        "Buildings_Openings",
        "mat_wood_doors",
        (654.0, 46.95),
        pz,
        pz + 4.2,
        3.6,
        11.5,
        0.02,
        (1.0, 0.0),
        (0.0, -1.0),
        uv_rect=WOOD_DOOR_UV["window_glass"],
        num_voussoirs=13,
        add_wall_spandrel=False,
    )
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        "mat_stone_boss",
        (654.0, 46.95, pz + 4.2),
        (1.0, 0.0),
        (0.0, -1.0),
        span_w=11.8,
        arch_h=3.7,
        ring_thickness=0.65,
        offset_n0=-0.04,
        offset_n1=0.35,
        num_voussoirs=13,
    )

    # 3. Prayer Partition (`Mechitza` at `y = -6.0, x = 634..665.8`), Shaded Torah Canopies & Upper Plaza Balustrade
    collector.add_box(
        "StreetProps",
        "mat_stone_mizzi",
        (650.0, -6.0, pz + 0.45),
        (31.0, 0.38, 0.90),
        uv_scale=0.35,
    )
    collector.add_box(
        "StreetProps",
        "mat_iron_dark",
        (650.0, -6.0, pz + 1.45),
        (30.8, 0.14, 1.15),
        uv_scale=0.4,
    )
    for mp_x in range(636, 666, 4):
        collector.add_box("StreetProps", "mat_stone_mizzi", (float(mp_x), -6.0, pz + 1.15), (0.34, 0.42, 2.3), uv_scale=0.35)
    for tx in (654.0, 658.0, 661.5):
        for ty in (-16.0, 6.0, 18.0, 30.0):
            collector.add_box(
                "StreetProps",
                "mat_stone_mizzi",
                (tx, ty, pz + 0.55),
                (1.4, 2.2, 1.1),
                uv_scale=0.4,
            )
    # White shading canopies over the main Men's prayer Torah reading bimahs (`(651.5, 10.0)` and `(651.5, 24.0)`)
    for can_y in (10.0, 24.0):
        collector.add_box("StreetProps", "mat_stone_mizzi", (651.5, can_y, pz + 2.85), (3.4, 4.2, 0.18), uv_scale=0.4, bottom_face=True)
        for cx_p in (-1.55, 1.55):
            for cy_p in (-1.95, 1.95):
                collector.add_box("StreetProps", "mat_iron_dark", (651.5 + cx_p, can_y + cy_p, pz + 1.42), (0.09, 0.09, 2.84))
    # Upper-Plaza Stone Balustrade Wall (`x = 626.0, y = -22..40`) + Israeli Flagpole at `(626.0, 2.0)`
    collector.add_box("StreetProps", "mat_stone_mizzi", (626.0, -14.0, pz + 0.55), (0.65, 16.0, 1.1), uv_scale=0.32)
    collector.add_box("StreetProps", "mat_stone_mizzi", (626.0, 22.0, pz + 0.55), (0.65, 28.0, 1.1), uv_scale=0.32)
    collector.add_branch_tube("StreetProps", "mat_stone_mizzi", (626.0, 2.0, pz), (626.0, 2.0, pz + 11.5), 0.11, 0.06, segments=8)
    collector.add_box("StreetProps", "mat_stone_mizzi", (626.0, 3.1, pz + 10.4), (0.06, 2.1, 1.45), uv_scale=0.5)
    collector.add_box("StreetProps", "mat_tile_iznik", (626.0, 3.1, pz + 10.95), (0.08, 2.1, 0.22), uv_scale=0.5)
    collector.add_box("StreetProps", "mat_tile_iznik", (626.0, 3.1, pz + 9.85), (0.08, 2.1, 0.22), uv_scale=0.5)

    # 4. Mughrabi Bridge Ramp (`x = 592..665.5, y = -32.0`) leading up to Bab al-Maghariba (`z = pz + 7.2m`)
    #    Built with a solid western stone approach embankment, warm natural timber (`mat_bark`), grounded steel support piers, and a stone abutment pier at the Wall!
    w_ground_z = min(pz, street_elevation_z(590.0, -32.0)) - 2.0
    w_top_z = pz + 0.65
    collector.add_box(
        "CityWalls",
        "mat_stone_boss",
        (589.5, -32.0, 0.5 * (w_ground_z + w_top_z)),
        (6.5, 4.4, w_top_z - w_ground_z),
        uv_scale=0.22,
    )
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (589.5, -32.0, w_top_z + 0.55), (6.7, 4.6, 1.1), uv_scale=0.30)
    n_ramp = 12
    for i in range(n_ramp):
        t0 = i / n_ramp
        t1 = (i + 1) / n_ramp
        rx0 = 592.0 + 73.5 * t0
        rx1 = 592.0 + 73.5 * t1
        rmx = 0.5 * (rx0 + rx1)
        rz = pz + 0.35 + 6.85 * (0.5 * (t0 + t1))
        g_z = min(pz, street_elevation_z(rmx, -32.0)) - 2.5
        col_h = max(1.5, rz - g_z)
        # Timber ramp deck & side timber screens (`mat_bark` with horizontal timber rails & steel X-bracing struts!)
        collector.add_box("Buildings_Details", "mat_bark", (rmx, -32.0, rz), (6.3, 3.6, 0.58), uv_scale=0.35)
        collector.add_box("Buildings_Details", "mat_bark", (rmx, -33.7, rz + 0.85), (6.3, 0.22, 1.35), uv_scale=0.35)
        collector.add_box("Buildings_Details", "mat_bark", (rmx, -30.3, rz + 0.85), (6.3, 0.22, 1.35), uv_scale=0.35)
        for rail_z in (0.35, 0.85, 1.48):
            collector.add_box("Buildings_Details", "mat_iron_dark", (rmx, -33.82, rz + rail_z), (6.3, 0.08, 0.08))
            collector.add_box("Buildings_Details", "mat_iron_dark", (rmx, -30.18, rz + rail_z), (6.3, 0.08, 0.08))
        # Vertical steel truss posts & overhead shading canopy frames along all bays of the ramp
        for sx_p in (-2.8, 0.0, 2.8):
            for sy_p in (-1.75, 1.75):
                collector.add_box("Buildings_Details", "mat_iron_dark", (rmx + sx_p, -32.0 + sy_p, rz + 1.45), (0.10, 0.10, 2.5))
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (rmx, -32.0, rz + 2.75), (6.3, 3.8, 0.16), uv_scale=0.4, bottom_face=True)
        # Twin grounded steel support piers + horizontal cross-beams + diagonal X-bracing struts at BOTH ends of each bay (`g_z .. rz`)
        for px_col in (rx0 + 0.55, rx1 - 0.55):
            for sy_col in (-1.35, 1.35):
                collector.add_box("Buildings_Details", "mat_iron_dark", (px_col, -32.0 + sy_col, g_z + 0.5 * col_h), (0.48, 0.48, col_h), uv_scale=0.3)
            collector.add_box("Buildings_Details", "mat_iron_dark", (px_col, -32.0, rz - 0.35), (0.55, 3.4, 0.42), uv_scale=0.3)
        for sy_br in (-1.35, 1.35):
            collector.add_branch_tube("Buildings_Details", "mat_iron_dark", (rx0 + 0.55, -32.0 + sy_br, g_z + 0.6), (rx1 - 0.55, -32.0 + sy_br, rz - 0.4), 0.07, 0.07, segments=6)
            collector.add_branch_tube("Buildings_Details", "mat_iron_dark", (rx1 - 0.55, -32.0 + sy_br, g_z + 0.6), (rx0 + 0.55, -32.0 + sy_br, rz - 0.4), 0.07, 0.07, segments=6)
    # Solid stone abutment pier where the Mughrabi Bridge meets the Western Wall (`(664.2, -32.0)`)
    collector.add_box("CityWalls", "mat_stone_boss", (664.2, -32.0, pz + 3.5), (3.8, 4.4, 7.6), uv_scale=0.22)

    # Bab al-Maghariba (Mughrabi Gate) portal on the Western Wall at `(665.8, -32.0, pz + 7.2)`
    add_arched_opening_panel(
        collector,
        "Buildings_Openings",
        "mat_wood_doors",
        (wall_x - 0.05, -32.0),
        pz + 7.2,
        pz + 9.8,
        1.2,
        2.8,
        0.02,
        (0.0, -1.0),
        (-1.0, 0.0),
        uv_rect=WOOD_DOOR_UV["door_wood"],
        num_voussoirs=7,
        add_wall_spandrel=False,
    )
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        "mat_stone_boss",
        (wall_x - 0.05, -32.0, pz + 9.8),
        (0.0, -1.0),
        (-1.0, 0.0),
        span_w=2.95,
        arch_h=1.25,
        ring_thickness=0.36,
        offset_n0=-0.02,
        offset_n1=0.22,
        num_voussoirs=7,
    )


def build_region_r06_western_wall_dung_gate(
    collector: MeshCollector,
    osm_data: dict,
    sign_atlas: Dict[str, dict],
) -> int:
    """
    Builds Region 6 (`r06_western_wall_dung_gate`):
      - Continuous stone-paved ground surface across `[555..670] x [-255..68]`
      - SE Ottoman City Wall (`Segment 2` part 2) & Dung Gate (`Bab al-Maghariba` at `(614.4, -196.6)`)
      - Herodian Western Wall (`HaKotel`), Western Wall Plaza, Wilson's Arch, and Mughrabi Bridge
      - All valid `r06_western_wall_dung_gate` OSM buildings, olive trees, and plaza lighting
    """
    build_region_ground_grid(collector, 555.0, 670.0, -255.0, 68.0, step=5.5)

    # 1. SE Ottoman City Wall (`Segment 2` part 2: `(560.0, -224.0)` -> Dung Gate -> Southern Temple Mount corner `(768.1, -84.8)`)
    se_wall_pts = [
        (560.0, -224.0),
        (583.7, -202.2),
        (614.4, -196.6),
        (626.8, -193.9),
        (655.3, -172.7),
        (694.4, -174.1),
        (746.0, -176.1),
        (751.8, -148.8),
        (768.1, -84.8),
    ]
    build_ottoman_wall_polyline(
        collector,
        se_wall_pts,
        wall_height=12.0,
        thickness=3.1,
        gate_gaps=[(614.4, -196.6, 8.5)],
        bastion_every=3,
    )

    # 2. Dung Gate (`Bab al-Maghariba` at `(614.4, -196.6)`, facing North `12 deg` into the Western Wall plaza)
    build_ottoman_gatehouse(
        collector,
        (614.4, -196.6),
        heading_inward_deg=12.0,
        gate_name="Dung Gate (Bab al-Maghariba)",
        width=15.5,
        depth=10.5,
        height=13.5,
    )

    # 3. Western Wall (`HaKotel`), Plaza, Wilson's Arch, Mughrabi Bridge & Dung Gate Ascent Road
    _build_western_wall_and_plaza(collector, sign_atlas)
    _build_souq_street_corridor(collector, (615.0, -178.0), (615.0, -108.0), street_w=8.5, wall_h=7.8, with_arches=False, with_shops=True, seed=601)

    # 4. OSM Buildings in `r06_western_wall_dung_gate`
    clear_scene_building_rings()
    osm_buildings = [
        b for b in osm_data.get("buildings", [])
        if b.get("region") == "r06_western_wall_dung_gate"
    ]
    valid_blds = []
    for b in osm_buildings:
        osm_id = int(b.get("osm_id", 0))
        if osm_id in (5862586, 7880713):
            continue
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        ring = b.get("ring", [])
        if len(ring) < 3:
            continue
        rxs = [float(p[0]) for p in ring]
        rys = [float(p[1]) for p in ring]
        if (max(rxs) - min(rxs)) > 65.0 or (max(rys) - min(rys)) > 65.0:
            continue
        # Keep Western Wall Prayer Plaza & Upper Plaza (`x = 576..668, y = -52..64`) clear
        if _ring_overlaps_rect(ring, 576.0, -52.0, 668.0, 64.0):
            continue
        # Keep Dung Gate approach & inner road (`x = 592..638, y = -225..-95`) clear
        if _ring_overlaps_rect(ring, 592.0, -225.0, 638.0, -95.0):
            continue
        valid_blds.append(b)
        register_scene_building_ring(osm_id, ring)

    built_count = 0
    for idx, b in enumerate(valid_blds):
        ring = b.get("ring", [])
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        bz = street_elevation_z(cx, cy)
        lvls = max(2, min(3, int(b.get("levels", 2))))
        build_building_from_footprint(
            collector,
            ring,
            base_z=bz,
            levels=lvls,
            sign_atlas=sign_atlas,
            building_id=int(b.get("osm_id", idx)),
            is_hero=False,
        )
        built_count += 1

    # Plaza Olive & Cypress Trees + Street Lamps
    for tx, ty, sp, th, seed in (
        (586.0, -12.0, "olive", 6.8, 701),
        (586.0, 14.0, "olive", 6.6, 702),
        (586.0, 34.0, "cypress", 11.0, 703),
        (602.0, -36.0, "olive", 6.4, 704),
        (628.0, -175.0, "cypress", 11.5, 705),
    ):
        tz = street_elevation_z(tx, ty)
        if sp == "cypress":
            build_cypress_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)
        else:
            build_olive_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)

    for lx, ly in ((612.0, 0.0), (612.0, 22.0), (632.0, -5.0), (632.0, 22.0), (614.0, -180.0)):
        build_street_lamp(collector, lx, ly, street_elevation_z(lx, ly), height=5.2)

    return built_count


# ==============================================================================
# 8. REGION 7: TEMPLE MOUNT / HARAM AL-SHARIF, DOME OF THE ROCK, AL-AQSA & GOLDEN GATE (`r07_temple_mount`)
# ==============================================================================

def _build_temple_mount_monuments(collector: MeshCollector, sign_atlas: Dict[str, dict]) -> None:
    """
    Builds the entire Haram al-Sharif / Temple Mount Esplanade (`x = 668..912, y = -85..385`):
      1. Raised Lower Esplanade Deck (`z_esplanade = -12.5m`, walkable `Ground_Paving`) &
         Raised Upper Al-Sakhra Platform (`x = 682..788, y = 92..232`, `z_upper = -8.3m`, walkable `Ground_Paving`)
         with 8 Mamluk *Mawazin* (Arcaded Stairways with walkable steps).
      2. **Dome of the Rock (`Qubbat al-Sakhra` at `(720.7, 155.7)`)**:
         - Octagonal marble/stone lower walls (`mat_stone_mizzi`, `radius = 26.5m`)
         - Upper octagonal Ottoman Iznik cobalt-blue ceramic tile frieze (`mat_tile_iznik`)
         - 4 cardinal columned entrance porches
         - Cylindrical Iznik-tiled drum (`mat_tile_iznik`, `radius = 11.2m`)
         - Radiant **Gilded Gold Dome (`mat_dome_gold`, `radius = 11.1m, height = 12.2m`)** + golden crescent spire (`z_top = +34.5m` above upper platform!).
      3. **Dome of the Chain (`Qubbat al-Silsila` at `(762.4, 162.4)`)**: 11-sided open colonnaded pavilion with Iznik drum and lead dome.
      4. **Al-Aqsa Mosque (`Al-Jami' al-Aqsa` at `(769.5, -52.0)`)**: 7-arched northern Crusader/Fatimid facade porch, basilica hall, clerestory nave, and iconic **Silver-Grey Lead Dome (`mat_dome_lead`)** + **Al-Kas Ablution Fountain**.
      5. **Four Mamluk Minarets** & **Cotton Merchants' Gate (`Bab al-Qattanin`)**.
    """
    z_esp = -12.5
    z_up = -8.3

    # 1. Walkable Lower Esplanade Paving (`x = 668..905, y = -85..385`)
    collector.add_box(
        "Ground_Paving",
        "mat_paving_flagstone",
        (786.5, 150.0, z_esp - 1.5),
        (237.0, 470.0, 3.0),
        uv_scale=0.22,
    )
    # Western & Northern Temple Mount Perimeter Retaining Wall & Cloister Arcade (`Riwaq`)
    collector.add_box("CityWalls", "mat_stone_ashlar_aged", (668.0, 215.0, z_esp + 4.2), (3.6, 340.0, 8.4), uv_scale=0.24)
    collector.add_box("CityWalls", "mat_stone_ashlar_aged", (778.0, 384.0, z_esp + 4.5), (220.0, 3.8, 9.0), uv_scale=0.24)

    # 2. Walkable Upper Al-Sakhra Platform (`x = 682..788, y = 92..232, z = -12.5 .. -8.3m`)
    collector.add_box(
        "Ground_Paving",
        "mat_paving_flagstone",
        (735.0, 162.0, 0.5 * (z_esp + z_up)),
        (106.0, 140.0, z_up - z_esp),
        uv_scale=0.25,
    )
    # Upper platform dressed-stone retaining wall & low parapet balustrade
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (735.0, 162.0, 0.5 * (z_esp + z_up)), (106.4, 140.4, z_up - z_esp - 0.05), uv_scale=0.24)

    # 3. Eight Mamluk *Mawazin* (Walkable Arcaded Stairways leading onto the Upper Platform)
    mawazin_defs = [
        # (stair_base_xy, stair_top_xy, width, arcade_tangent)
        ((720.7, 83.0), (720.7, 92.0), 8.5, (1.0, 0.0)),   # South Mawazin (facing Al-Aqsa)
        ((755.0, 83.0), (755.0, 92.0), 6.5, (1.0, 0.0)),   # SE Mawazin
        ((720.7, 241.0), (720.7, 232.0), 8.5, (1.0, 0.0)), # North Mawazin
        ((673.0, 155.7), (682.0, 155.7), 8.5, (0.0, 1.0)), # West Central Mawazin (Cotton Gate axis)
        ((673.0, 122.0), (682.0, 122.0), 6.5, (0.0, 1.0)), # SW Mawazin
        ((673.0, 195.0), (682.0, 195.0), 6.5, (0.0, 1.0)), # NW Mawazin
        ((797.0, 162.4), (788.0, 162.4), 7.5, (0.0, 1.0)), # East Mawazin (Dome of the Chain axis)
    ]
    for p_bot, p_top, mw_w, (tx, ty) in mawazin_defs:
        # Stop staircase steps `0.58m` in front of `p_top` so the top steps sit cleanly in front of (and never intersect) the 4 column plinths!
        stair_dx = p_top[0] - p_bot[0]
        stair_dy = p_top[1] - p_bot[1]
        stair_len_full = math.hypot(stair_dx, stair_dy)
        ux_s, uy_s = stair_dx / stair_len_full, stair_dy / stair_len_full
        p_top_step = (p_top[0] - ux_s * 0.58, p_top[1] - uy_s * 0.58)
        n_steps = 14
        for s in range(n_steps):
            t0 = s / n_steps
            t1 = (s + 1) / n_steps
            sx = p_bot[0] + (p_top_step[0] - p_bot[0]) * (0.5 * (t0 + t1))
            sy = p_bot[1] + (p_top_step[1] - p_bot[1]) * (0.5 * (t0 + t1))
            sz = z_esp + (z_up - z_esp) * t1
            step_len = math.hypot(p_top_step[0] - p_bot[0], p_top_step[1] - p_bot[1]) / n_steps
            box_sx = mw_w if abs(tx) > 0.5 else step_len + 0.08
            box_sy = step_len + 0.08 if abs(tx) > 0.5 else mw_w
            collector.add_box(
                "Ground_Steps",
                "mat_paving_flagstone",
                (sx, sy, 0.5 * (z_esp + sz)),
                (box_sx, box_sy, sz - z_esp),
                uv_scale=0.30,
            )
            # Stepped stone cheek-wall balustrade flanking both sides of the Mawazin stairs
            for ch_s in (-1.0, 1.0):
                ch_x = sx + tx * (ch_s * (mw_w * 0.5 + 0.22))
                ch_y = sy + ty * (ch_s * (mw_w * 0.5 + 0.22))
                ch_sx = 0.42 if abs(tx) > 0.5 else step_len + 0.12
                ch_sy = step_len + 0.12 if abs(tx) > 0.5 else 0.42
                collector.add_box(
                    "Buildings_Details",
                    "mat_stone_mizzi",
                    (ch_x, ch_y, 0.5 * (z_esp + sz + 0.45)),
                    (ch_sx, ch_sy, (sz - z_esp) + 0.45),
                    uv_scale=0.32,
                )
        # Freestanding Mamluk triple-arched stone colonnade (`Mawazin`) crowning the top of the stairs
        col_offs = (-mw_w * 0.44, -mw_w * 0.15, mw_w * 0.15, mw_w * 0.44)
        for col_off in col_offs:
            cx_m = p_top[0] + tx * col_off
            cy_m = p_top[1] + ty * col_off
            # Square dressed-stone plinth base at the top of the stairs
            collector.add_box(
                "Buildings_Details",
                "mat_stone_boss",
                (cx_m, cy_m, z_up + 0.22),
                (0.72, 0.72, 0.44),
                uv_scale=0.35,
            )
            collector.add_branch_tube(
                "Buildings_Details",
                "mat_stone_mizzi",
                (cx_m, cy_m, z_up + 0.42),
                (cx_m, cy_m, z_up + 4.8),
                0.32,
                0.29,
                segments=10,
                cap_end=True,
            )
            collector.add_box(
                "Buildings_Details",
                "mat_stone_boss",
                (cx_m, cy_m, z_up + 3.45),
                (0.76, 0.76, 0.36),
                uv_scale=0.35,
            )
            # Carved Mamluk circular spandrel medallion (`rosette`) above the two inner columns
            if abs(col_off) < mw_w * 0.30:
                collector.add_branch_tube(
                    "Buildings_Details",
                    "mat_stone_boss",
                    (cx_m - ux_s * 0.44, cy_m - uy_s * 0.44, z_up + 4.42),
                    (cx_m + ux_s * 0.44, cy_m + uy_s * 0.44, z_up + 4.42),
                    0.34,
                    0.34,
                    segments=10,
                    cap_end=True,
                )
        # 3 pointed Mamluk arches spanning between the 4 columns of the Mawazin + hanging brass mosque lamps
        nx_m, ny_m = -ty, tx
        for a_i in range(3):
            c_mid = 0.5 * (col_offs[a_i] + col_offs[a_i + 1])
            span_m = abs(col_offs[a_i + 1] - col_offs[a_i]) - 0.35
            ax_m = p_top[0] + tx * c_mid
            ay_m = p_top[1] + ty * c_mid
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_mizzi",
                (ax_m, ay_m, z_up + 3.55),
                (tx, ty),
                (nx_m, ny_m),
                span_w=span_m,
                arch_h=1.15,
                ring_thickness=0.32,
                offset_n0=-0.42,
                offset_n1=0.42,
                num_voussoirs=7,
            )
            collector.add_box("StreetProps", "mat_dome_gold", (ax_m, ay_m, z_up + 4.25), (0.04, 0.04, 0.75))
            collector.add_box("StreetProps", "mat_lantern_glow", (ax_m, ay_m, z_up + 3.75), (0.26, 0.26, 0.34), bottom_face=True)
        lintel_sx = mw_w + 0.6 if abs(tx) > 0.5 else 0.85
        lintel_sy = 0.85 if abs(tx) > 0.5 else mw_w + 0.6
        collector.add_box(
            "Buildings_Details",
            "mat_stone_mizzi",
            (p_top[0], p_top[1], z_up + 5.05),
            (lintel_sx, lintel_sy, 0.48),
            uv_scale=0.28,
            bottom_face=True,
        )
        # Continuous carved Mamluk cornice frieze & low stepped cresting crowning each Mawazin arcade
        collector.add_box(
            "Buildings_Details",
            "mat_stone_boss",
            (p_top[0], p_top[1], z_up + 5.42),
            (lintel_sx + 0.18, lintel_sy + 0.18, 0.28),
            uv_scale=0.32,
            bottom_face=True,
        )
        for m_off in (-mw_w * 0.42, -mw_w * 0.28, -mw_w * 0.14, 0.0, mw_w * 0.14, mw_w * 0.28, mw_w * 0.42):
            collector.add_box(
                "Buildings_Details",
                "mat_stone_mizzi",
                (p_top[0] + tx * m_off, p_top[1] + ty * m_off, z_up + 5.68),
                (0.42, 0.34, 0.28),
                uv_scale=0.35,
            )

    # 4. DOME OF THE ROCK (`Qubbat al-Sakhra` at `(720.7, 155.7)`, `z_up = -8.3m`)
    dx, dy = 720.7, 155.7
    oct_r = 26.5
    # Build true 8-sided octagonal sanctuary walls (rotated 22.5 deg so flat faces align N, S, E, W!)
    for i in range(8):
        a0 = math.radians(i * 45.0 - 22.5)
        a1 = math.radians((i + 1) * 45.0 - 22.5)
        p0 = (dx + math.cos(a0) * oct_r, dy + math.sin(a0) * oct_r)
        p1 = (dx + math.cos(a1) * oct_r, dy + math.sin(a1) * oct_r)
        am = 0.5 * (a0 + a1)
        nx, ny = math.cos(am), math.sin(am)
        tx, ty = -math.sin(am), math.cos(am)
        mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])

        # Lower white marble octagonal dado (`z = z_up .. z_up + 4.8m`, flush with upper tile wall at `offset_n1 = 0.10`)
        collector.add_oriented_box(
            "Buildings_Walls",
            "mat_stone_mizzi",
            p0,
            p1,
            z_up,
            z_up + 4.8,
            -26.5,
            0.10,
            (nx, ny),
            uv_scale=0.24,
            cap_back=True,
        )
        # Upper Ottoman Iznik cobalt-blue & turquoise ceramic tile course + attica parapet (`z = z_up + 4.8 .. z_up + 10.8m`, flush at `0.10`)
        collector.add_oriented_box(
            "Buildings_Details",
            "mat_tile_iznik",
            p0,
            p1,
            z_up + 4.8,
            z_up + 10.8,
            -26.5,
            0.10,
            (nx, ny),
            uv_scale=0.35,
            cap_back=True,
        )
        # Projecting white marble string course, calligraphic parapet border & upper cornice
        collector.add_oriented_box(
            "Buildings_Details",
            "mat_stone_mizzi",
            p0,
            p1,
            z_up + 4.6,
            z_up + 4.95,
            -0.3,
            0.24,
            (nx, ny),
            uv_scale=0.35,
            cap_back=True,
        )
        collector.add_oriented_box(
            "Buildings_Details",
            "mat_stone_mizzi",
            p0,
            p1,
            z_up + 9.15,
            z_up + 9.42,
            -0.2,
            0.22,
            (nx, ny),
            uv_scale=0.35,
            cap_back=True,
        )
        collector.add_oriented_box(
            "Buildings_Details",
            "mat_stone_mizzi",
            p0,
            p1,
            z_up + 10.6,
            z_up + 11.0,
            -0.3,
            0.30,
            (nx, ny),
            uv_scale=0.35,
            cap_back=True,
        )

        # 5 arched blind bays + windows on each of the 8 octagonal faces (`arch_spring_z = z_up + 6.5`, flush panel + proud white marble arch!)
        for bay_idx in (-2, -1, 0, 1, 2):
            bx = mx + tx * (bay_idx * 3.6) + nx * 0.11
            by = my + ty * (bay_idx * 3.6) + ny * 0.11
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (bx, by),
                z_up + (0.0 if (i % 2 == 0 and bay_idx == 0) else 2.2),
                z_up + 6.5,
                1.08,
                2.25,
                0.01,
                (tx, ty),
                (nx, ny),
                uv_rect=WOOD_DOOR_UV["door_wood" if (i % 2 == 0 and bay_idx == 0) else "window_glass"],
                num_voussoirs=9,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_mizzi",
                (mx + tx * (bay_idx * 3.6) + nx * 0.10, my + ty * (bay_idx * 3.6) + ny * 0.10, z_up + 6.5),
                (tx, ty),
                (nx, ny),
                span_w=2.25,
                arch_h=1.08,
                ring_thickness=0.34,
                offset_n0=-0.04,
                offset_n1=0.28,
                num_voussoirs=9,
            )

        # Cardinal Entrance Porches (E, N, W, S corresponds to `i in (0, 2, 4, 6)`)
        if i % 2 == 0:
            is_south_portico = (i == 6)
            px = mx + nx * (3.2 if is_south_portico else 2.6)
            py = my + ny * (3.2 if is_south_portico else 2.6)
            porch_w = 10.6 if is_south_portico else 6.8
            porch_d = 5.4 if is_south_portico else 4.2
            porch_sx = porch_w if abs(tx) > 0.5 else porch_d
            porch_sy = porch_d if abs(tx) > 0.5 else porch_w
            collector.add_box("Buildings_Details", "mat_stone_mizzi", (px, py, z_up + 4.75), (porch_sx, porch_sy, 0.65), uv_scale=0.30, bottom_face=True)
            collector.add_box("Buildings_Details", "mat_dome_lead", (px, py, z_up + 5.18), (porch_sx + 0.3, porch_sy + 0.3, 0.25), uv_scale=0.30, bottom_face=True)
            c_sides = (-4.4, -1.6, 1.6, 4.4) if is_south_portico else (-2.4, 2.4)
            for c_side in c_sides:
                col_x = mx + nx * (5.2 if is_south_portico else 4.2) + tx * c_side
                col_y = my + ny * (5.2 if is_south_portico else 4.2) + ty * c_side
                collector.add_branch_tube(
                    "Buildings_Details",
                    "mat_stone_mizzi",
                    (col_x, col_y, z_up),
                    (col_x, col_y, z_up + 4.45),
                    0.34,
                    0.30,
                    segments=10,
                    cap_end=True,
                )
                collector.add_box("Buildings_Details", "mat_stone_boss", (col_x, col_y, z_up + 4.55), (0.76, 0.76, 0.32), uv_scale=0.35)

    # Sloped lead-sheathed ambulatory roof between octagon edge (`r=24.5, z=+10.2`) and central drum (`r=11.9, z=+11.6`)
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_dome_lead",
        (dx, dy, z_up + 10.1),
        (dx, dy, z_up + 11.6),
        24.4,
        11.8,
        segments=8,
        cap_end=True,
    )

    # Cylindrical Iznik-Tiled Drum (`radius = 11.85m, z = z_up + 11.2 .. z_up + 20.4m`)
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_tile_iznik",
        (dx, dy, z_up + 11.2),
        (dx, dy, z_up + 20.4),
        11.85,
        11.75,
        segments=24,
        cap_end=True,
    )
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_stone_mizzi",
        (dx, dy, z_up + 20.1),
        (dx, dy, z_up + 20.7),
        12.05,
        12.05,
        segments=24,
        cap_end=True,
    )
    # 16 arched clerestory windows around the drum (`arch_spring_z = z_up + 16.8`) with white stone archivolts
    for w_idx in range(16):
        ang = math.radians(w_idx * 22.5)
        wx = dx + math.cos(ang) * 11.82
        wy = dy + math.sin(ang) * 11.82
        tx, ty = -math.sin(ang), math.cos(ang)
        nx, ny = math.cos(ang), math.sin(ang)
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (wx, wy),
            z_up + 13.0,
            z_up + 16.8,
            0.78,
            1.50,
            0.01,
            (tx, ty),
            (nx, ny),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=7,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_mizzi",
            (wx, wy, z_up + 16.8),
            (tx, ty),
            (nx, ny),
            span_w=1.66,
            arch_h=0.86,
            ring_thickness=0.24,
            offset_n0=-0.04,
            offset_n1=0.22,
            num_voussoirs=7,
        )

    # Radiant Gilded Gold Dome (`mat_dome_gold`, `z = z_up + 20.4 .. z_up + 34.4m`, commanding presence above the octagon!)
    collector.add_dome(
        "Buildings_Details",
        "mat_dome_gold",
        (dx, dy, z_up + 20.4),
        radius=11.90,
        height=14.0,
        rings=12,
        segments=24,
    )
    # Golden Finial Spire & Crescent (`z = z_up + 34.2 .. z_up + 39.0m`)
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_dome_gold",
        (dx, dy, z_up + 34.1),
        (dx, dy, z_up + 38.5),
        0.44,
        0.15,
        segments=10,
        cap_end=True,
    )
    collector.add_dome(
        "Buildings_Details",
        "mat_dome_gold",
        (dx, dy, z_up + 38.3),
        radius=0.70,
        height=1.0,
        rings=5,
        segments=10,
    )

    # 5. DOME OF THE CHAIN (`Qubbat al-Silsila` at `(762.4, 162.4)`)
    #    Two-tier open colonnaded pavilion with outer 11-sided stone arcade, inner hexagonal Iznik-tiled drum & lead dome!
    cx_ch, cy_ch = 762.4, 162.4
    collector.add_box("Ground_Paving", "mat_stone_mizzi", (cx_ch, cy_ch, z_up + 0.15), (14.5, 14.5, 0.3), uv_scale=0.35)
    for c_i in range(11):
        ang = math.radians(c_i * (360.0 / 11.0))
        col_x = cx_ch + math.cos(ang) * 6.4
        col_y = cy_ch + math.sin(ang) * 6.4
        collector.add_branch_tube(
            "Buildings_Details",
            "mat_stone_mizzi",
            (col_x, col_y, z_up + 0.3),
            (col_x, col_y, z_up + 4.4),
            0.28,
            0.25,
            segments=10,
            cap_end=True,
        )
        collector.add_box("Buildings_Details", "mat_stone_boss", (col_x, col_y, z_up + 4.55), (0.68, 0.68, 0.32), uv_scale=0.35)
    # Outer 11-sided stone entablature ring + flat ambulatory roof (`r = 6.6m -> 3.6m`, `z = z_up + 4.7 .. 5.4m`)
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_stone_mizzi",
        (cx_ch, cy_ch, z_up + 4.65),
        (cx_ch, cy_ch, z_up + 5.25),
        6.65,
        6.55,
        segments=11,
        cap_end=True,
    )
    # Inner hexagonal Iznik-tiled drum (`r = 3.55m`, `z = z_up + 5.2 .. 8.5m`) + hemispherical lead dome & gilded finial
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_tile_iznik",
        (cx_ch, cy_ch, z_up + 5.2),
        (cx_ch, cy_ch, z_up + 8.5),
        3.60,
        3.55,
        segments=6,
        cap_end=True,
    )
    collector.add_dome(
        "Buildings_Details",
        "mat_dome_lead",
        (cx_ch, cy_ch, z_up + 8.4),
        radius=3.60,
        height=3.3,
        rings=7,
        segments=14,
    )
    collector.add_branch_tube("Buildings_Details", "mat_dome_gold", (cx_ch, cy_ch, z_up + 11.6), (cx_ch, cy_ch, z_up + 12.8), 0.12, 0.05, segments=8, cap_end=True)

    # 6. AL-KAS ABLUTION FOUNTAIN (`(756.5, 25.0)`) between Dome of the Rock and Al-Aqsa Mosque
    #    Positioned slightly west of the central Al-Aqsa axis so Al-Aqsa's central portal and Rose Window are 100% visible!
    kx, ky = 756.5, 25.0
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_stone_mizzi",
        (kx, ky, z_esp),
        (kx, ky, z_esp + 0.95),
        5.2,
        5.4,
        segments=16,
        cap_end=True,
    )
    # Shimmering blue water surface inside the circular stone pool + outer ring of 12 stone washing seats
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_tile_iznik",
        (kx, ky, z_esp + 0.85),
        (kx, ky, z_esp + 1.02),
        4.75,
        4.70,
        segments=16,
        cap_end=True,
    )
    for st_i in range(12):
        s_rad = math.radians(st_i * 30.0)
        collector.add_branch_tube(
            "StreetProps",
            "mat_stone_mizzi",
            (kx + math.cos(s_rad) * 5.95, ky + math.sin(s_rad) * 5.95, z_esp),
            (kx + math.cos(s_rad) * 5.95, ky + math.sin(s_rad) * 5.95, z_esp + 0.48),
            0.26,
            0.24,
            segments=8,
            cap_end=True,
        )
    # Sculpted central stone chalice basin
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_stone_mizzi",
        (kx, ky, z_esp + 0.95),
        (kx, ky, z_esp + 2.45),
        0.58,
        1.45,
        segments=12,
        cap_end=True,
    )
    for k_i in range(8):
        k_rad = math.radians(k_i * 45.0 + 22.5)
        kcx = kx + math.cos(k_rad) * 4.65
        kcy = ky + math.sin(k_rad) * 4.65
        collector.add_branch_tube(
            "Buildings_Details",
            "mat_stone_mizzi",
            (kcx, kcy, z_esp + 0.95),
            (kcx, kcy, z_esp + 3.85),
            0.20,
            0.18,
            segments=8,
            cap_end=True,
        )
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_dome_lead",
        (kx, ky, z_esp + 3.80),
        (kx, ky, z_esp + 5.65),
        5.05,
        0.16,
        segments=8,
        cap_end=True,
    )

    # 7. AL-AQSA MOSQUE (`Al-Jami' al-Aqsa` centered at `(762.0, -46.0)`, `y = -84..-8`, `z_esp = -12.5m`)
    ax, ay = 762.0, -46.0
    # 7-aisled basilica lower sanctuary (`56m wide x 72m deep x 11.8m tall`)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (ax, ay, z_esp + 5.9), (56.0, 72.0, 11.8), uv_scale=0.22)
    # Raised central nave (`16.8m wide x 72.2m deep x 16.8m tall`) + pitched lead roof & northern clerestory gable
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (ax, ay, z_esp + 8.4), (16.8, 72.2, 16.8), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_dome_lead", (ax, ay, z_esp + 17.2), (17.4, 72.6, 0.9), uv_scale=0.25)

    # Northern Clerestory Gable & Sculpted Circular Crusader Rose Window (`y = -9.8, z = z_esp + 11.2 .. z_esp + 18.8m`)
    for step_i in range(4):
        gw = 16.8 - step_i * 3.4
        gz = z_esp + 16.8 + step_i * 0.55
        collector.add_box("Buildings_Walls", "mat_stone_mizzi", (ax, -9.8, gz), (gw, 2.2, 0.6), uv_scale=0.26)
    # Central Sculpted Circular Crusader Rose Window (`(ax, -8.68, z_esp + 14.6)`)
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_stone_mizzi",
        (ax, -8.85, z_esp + 14.6),
        (ax, -8.45, z_esp + 14.6),
        1.65,
        1.55,
        segments=16,
    )
    collector.add_branch_tube(
        "Buildings_Openings",
        "mat_iron_dark",
        (ax, -8.68, z_esp + 14.6),
        (ax, -8.52, z_esp + 14.6),
        1.28,
        1.25,
        segments=16,
        cap_end=True,
    )
    collector.add_branch_tube(
        "Buildings_Details",
        "mat_stone_boss",
        (ax, -8.55, z_esp + 14.6),
        (ax, -8.38, z_esp + 14.6),
        0.48,
        0.42,
        segments=12,
        cap_end=True,
    )
    for cx_off in (-4.8, 4.8):
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (ax + cx_off, -8.68),
            z_esp + 12.2,
            z_esp + 14.8,
            0.85,
            1.7,
            0.02,
            (-1.0, 0.0),
            (0.0, 1.0),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=7,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (ax + cx_off, -8.68, z_esp + 14.8),
            (-1.0, 0.0),
            (0.0, 1.0),
            span_w=1.85,
            arch_h=0.90,
            ring_thickness=0.32,
            offset_n0=-0.02,
            offset_n1=0.20,
            num_voussoirs=7,
        )

    # Northern 7-Arched Crusader/Fatimid Facade Porch (`y = -7.5`, facing North toward Dome of the Rock)
    py_aqsa = -7.5
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (ax, py_aqsa, z_esp + 5.6), (56.4, 5.2, 11.2), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_stone_boss", (ax, py_aqsa, z_esp + 11.4), (57.2, 5.8, 0.65), uv_scale=0.28)
    # Upper porch stone balustrade cresting + coupled marble colonnettes between the 7 bays
    for bal_x in range(-26, 27, 4):
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (ax + bal_x, py_aqsa + 2.65, z_esp + 12.05), (1.4, 0.42, 0.68), uv_scale=0.35)
    for pier_x in (-19.0, -12.0, -4.6, 4.6, 12.0, 19.0):
        collector.add_branch_tube(
            "Buildings_Details",
            "mat_stone_mizzi",
            (ax + pier_x, py_aqsa + 2.75, z_esp),
            (ax + pier_x, py_aqsa + 2.75, z_esp + 5.2),
            0.26,
            0.23,
            segments=8,
            cap_end=True,
        )
    for arch_idx, (off_x, span_w, arch_h) in enumerate((
        (-22.5, 5.2, 2.2),
        (-15.5, 5.2, 2.2),
        (-8.5, 5.2, 2.2),
        (0.0, 8.2, 3.4),
        (8.5, 5.2, 2.2),
        (15.5, 5.2, 2.2),
        (22.5, 5.2, 2.2),
    )):
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (ax + off_x, py_aqsa + 2.61),
            z_esp,
            z_esp + 5.2,
            arch_h,
            span_w,
            0.01,
            (-1.0, 0.0),
            (0.0, 1.0),
            uv_rect=WOOD_DOOR_UV["door_wood" if off_x == 0.0 else "window_glass"],
            num_voussoirs=9,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (ax + off_x, py_aqsa + 2.61, z_esp + 5.2),
            (-1.0, 0.0),
            (0.0, 1.0),
            span_w=span_w,
            arch_h=arch_h,
            ring_thickness=0.45,
            offset_n0=-0.04,
            offset_n1=0.28,
            num_voussoirs=9,
        )
        # Hanging brass mosque lamp inside each porch arch
        collector.add_box("StreetProps", "mat_dome_gold", (ax + off_x, py_aqsa + 2.82, z_esp + 5.2 + arch_h - 0.65), (0.05, 0.05, 0.95))
        collector.add_box("StreetProps", "mat_lantern_glow", (ax + off_x, py_aqsa + 2.82, z_esp + 5.2 + arch_h - 1.25), (0.32, 0.32, 0.42), bottom_face=True)

    # Iconic Silver-Grey Lead Dome (`(762.0, -28.0)`, positioned so it rises prominently above the northern porch & clerestory!)
    collector.add_branch_tube(
        "Buildings_Walls",
        "mat_stone_mizzi",
        (ax, -28.0, z_esp + 16.5),
        (ax, -28.0, z_esp + 21.6),
        9.6,
        9.4,
        segments=16,
        cap_end=True,
    )
    collector.add_dome(
        "Buildings_Details",
        "mat_dome_lead",
        (ax, -28.0, z_esp + 21.4),
        radius=9.5,
        height=9.6,
        rings=10,
        segments=20,
    )
    collector.add_box("Buildings_Details", "mat_dome_gold", (ax, -28.0, z_esp + 31.5), (0.28, 0.28, 1.9), uv_scale=0.5)

    # 8. Four Mamluk Minarets around the Haram al-Sharif perimeter:
    #    - Bab al-Silsila Minaret `(668.5, 68.0)`
    #    - Al-Fakhariyya Minaret `(672.0, -76.0)`
    #    - Al-Ghawanima Minaret `(675.0, 376.0)`
    #    - Bab al-Asbat Minaret `(855.0, 382.0)`
    for mx_m, my_m in ((668.5, 68.0), (672.0, -76.0), (675.0, 376.0), (855.0, 382.0)):
        collector.add_box("Buildings_Walls", "mat_stone_mizzi", (mx_m, my_m, z_esp + 9.5), (4.6, 4.6, 19.0), uv_scale=0.25)
        collector.add_box("Buildings_Details", "mat_stone_boss", (mx_m, my_m, z_esp + 19.4), (5.6, 5.6, 0.85), uv_scale=0.30)
        collector.add_branch_tube(
            "Buildings_Walls",
            "mat_stone_mizzi",
            (mx_m, my_m, z_esp + 19.8),
            (mx_m, my_m, z_esp + 26.5),
            1.85,
            1.75,
            segments=12,
            cap_end=True,
        )
        collector.add_dome(
            "Buildings_Details",
            "mat_dome_lead",
            (mx_m, my_m, z_esp + 26.4),
            radius=1.8,
            height=2.2,
            rings=6,
            segments=12,
        )

    # 9. Northern Haram al-Sharif Esplanade:
    #    - Raised Stone *Mastaba* Prayer Platform & Domed Mamluk **Sabil of Qaitbay (`(744.5, 277.0)`)**
    #    - Western Mamluk *Riwaq* (Cloister Portico Arcade & Madrasa Facades along `x = 671.5, y = 210..305`)
    #    - Raised Stone-Curbed Cypress & Olive Planter Terrace (`(723.5, 283.0)`)
    ms_x, ms_y = 743.5, 278.0
    collector.add_box("Ground_Steps", "mat_stone_mizzi", (ms_x, ms_y, z_esp + 0.24), (11.5, 8.8, 0.48), uv_scale=0.30)
    collector.add_box("Buildings_Details", "mat_stone_boss", (ms_x - 2.6, ms_y - 4.15, z_esp + 1.35), (1.85, 0.45, 2.2), uv_scale=0.32)
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        "mat_terracotta",
        (ms_x - 2.6, ms_y - 3.90, z_esp + 1.55),
        (1.0, 0.0),
        (0.0, 1.0),
        span_w=1.15,
        arch_h=0.58,
        ring_thickness=0.22,
        offset_n0=-0.04,
        offset_n1=0.14,
        num_voussoirs=5,
    )
    # Domed Mamluk Sabil of Qaitbay (`(745.5, 276.8)`) with alternating red/cream Ablaq courses, bronze grilles & carved stone dome
    sq_x, sq_y = 745.8, 276.8
    sq_z = z_esp + 0.48
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (sq_x, sq_y, sq_z + 2.3), (3.8, 3.8, 4.6), uv_scale=0.26)
    for ab_z in (1.1, 1.9, 2.7, 3.5, 4.2):
        collector.add_box(
            "Buildings_Details",
            "mat_terracotta" if int(ab_z * 2) % 2 == 0 else "mat_stone_boss",
            (sq_x, sq_y, sq_z + ab_z),
            (3.92, 3.92, 0.28),
            uv_scale=0.35,
        )
    collector.add_box("Buildings_Details", "mat_stone_boss", (sq_x, sq_y, sq_z + 4.72), (4.2, 4.2, 0.34), uv_scale=0.32)
    for f_ang in (0.0, 90.0, 180.0, 270.0):
        frad = math.radians(f_ang)
        fnx_q, fny_q = math.cos(frad), math.sin(frad)
        ftx_q, fty_q = -fny_q, fnx_q
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (sq_x + fnx_q * 1.92, sq_y + fny_q * 1.92),
            sq_z + 0.65,
            sq_z + 2.35,
            0.72,
            1.45,
            0.01,
            (ftx_q, fty_q),
            (fnx_q, fny_q),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=5,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (sq_x + fnx_q * 1.92, sq_y + fny_q * 1.92, sq_z + 2.35),
            (ftx_q, fty_q),
            (fnx_q, fny_q),
            span_w=1.45,
            arch_h=0.72,
            ring_thickness=0.26,
            offset_n0=-0.04,
            offset_n1=0.18,
            num_voussoirs=5,
        )
    collector.add_branch_tube("Buildings_Walls", "mat_stone_mizzi", (sq_x, sq_y, sq_z + 4.85), (sq_x, sq_y, sq_z + 6.45), 1.85, 1.75, segments=8, smooth=False, cap_end=True)
    collector.add_dome("Buildings_Details", "mat_stone_mizzi", (sq_x, sq_y, sq_z + 6.38), radius=1.80, height=2.35, rings=7, segments=12)
    collector.add_box("Buildings_Details", "mat_dome_gold", (sq_x, sq_y, sq_z + 9.05), (0.18, 0.18, 0.85), uv_scale=0.5)
    # Stone-curbed cypress & olive planter terrace (`(723.5, 283.0)`) + stone benches
    collector.add_box("StreetProps", "mat_stone_mizzi", (723.5, 283.0, z_esp + 0.32), (5.6, 12.0, 0.64), uv_scale=0.32)
    collector.add_box("Trees_Foliage", "mat_foliage_olive", (723.5, 283.0, z_esp + 0.70), (4.8, 11.2, 0.34), uv_scale=0.6)
    collector.add_box("StreetProps", "mat_stone_boss", (727.2, 283.0, z_esp + 0.28), (0.65, 2.4, 0.56), uv_scale=0.35)
    # Western Mamluk Riwaq Arcade & Madrasa Facades (`x = 672.5, y = 215..300`)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (671.6, 258.0, z_esp + 4.8), (4.2, 86.0, 9.6), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_terracotta", (673.75, 258.0, z_esp + 5.2), (0.16, 86.0, 0.34), uv_scale=0.35)
    collector.add_box("Buildings_Details", "mat_stone_boss", (671.8, 258.0, z_esp + 9.75), (4.6, 86.4, 0.38), uv_scale=0.30)
    for rw_y in range(218, 302, 7):
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (673.72, float(rw_y)),
            z_esp,
            z_esp + 2.8,
            1.35,
            2.70,
            0.01,
            (0.0, 1.0),
            (1.0, 0.0),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=7,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (673.72, float(rw_y), z_esp + 2.8),
            (0.0, 1.0),
            (1.0, 0.0),
            span_w=2.70,
            arch_h=1.35,
            ring_thickness=0.36,
            offset_n0=-0.04,
            offset_n1=0.24,
            num_voussoirs=7,
        )


def build_region_r07_temple_mount(
    collector: MeshCollector,
    osm_data: dict,
    sign_atlas: Dict[str, dict],
) -> int:
    """
    Builds Region 7 (`r07_temple_mount`):
      - Raised Haram al-Sharif Esplanade & Upper Al-Sakhra Platform
      - Southern & Eastern Herodian/Ottoman Retaining Walls (`Segments 3 & 4`)
      - Golden Gate (`Bab al-Rahma` at `(894.6, 260.8)`, `sealed_double_arch=True`)
      - Dome of the Rock (`Qubbat al-Sakhra`), Dome of the Chain, 8 *Mawazin* arcaded stairways,
        Al-Aqsa Mosque, Al-Kas Fountain, 4 Mamluk Minarets, and 36 Cypress & Olive trees!
    """
    # 1. Southern & Eastern Temple Mount Retaining Walls (`Segments 3 & 4` of `5862586`)
    east_wall_pts = [
        (768.1, -84.8),
        (934.6, -58.4),
        (917.4, 111.7),
        (908.3, 187.4),
        (901.7, 252.2),
        (894.6, 260.8),
        (893.7, 344.2),
        (887.3, 407.5),
    ]
    build_ottoman_wall_polyline(
        collector,
        east_wall_pts,
        wall_height=15.5,
        thickness=3.8,
        gate_gaps=[(894.6, 260.8, 7.5)],
        bastion_every=2,
    )

    # 2. Golden Gate (`Bab al-Rahma` at `(894.6, 260.8)`, facing West `265 deg` into the Temple Mount)
    build_ottoman_gatehouse(
        collector,
        (894.6, 260.8),
        heading_inward_deg=265.0,
        gate_name="Golden Gate (Bab al-Rahma)",
        width=18.5,
        depth=14.0,
        height=15.0,
        sealed_double_arch=True,
    )

    # 3. Dome of the Rock, Al-Aqsa Mosque, Dome of the Chain, Mawazin & Minarets
    _build_temple_mount_monuments(collector, sign_atlas)

    # 4. Cypress & Olive Groves across the Haram al-Sharif Esplanade (`z_esp = -12.5m`)
    z_esp = -12.5
    z_up = -8.3
    tm_trees = [
        (695.0, 45.0, "cypress", 13.5),
        (712.0, 58.0, "olive", 7.2),
        (765.0, 52.0, "cypress", 14.0),
        (788.0, 42.0, "olive", 7.5),
        (812.0, 68.0, "cypress", 13.0),
        (825.0, 115.0, "olive", 7.4),
        (818.0, 155.0, "cypress", 13.8),
        (835.0, 195.0, "olive", 7.2),
        (815.0, 245.0, "cypress", 14.2),
        (785.0, 268.0, "olive", 7.6),
        (745.0, 290.0, "cypress", 13.5),
        (723.5, 285.0, "olive", 7.2),
        (705.0, 265.0, "olive", 7.0),
        (695.0, 315.0, "cypress", 13.8),
        (755.0, 325.0, "olive", 7.5),
        (815.0, 315.0, "cypress", 14.0),
        (855.0, 255.0, "olive", 7.2),
        (865.0, 185.0, "cypress", 13.2),
        (855.0, 95.0, "olive", 7.4),
        (702.0, -18.0, "cypress", 13.6),
        (818.0, -18.0, "cypress", 13.8),
    ]
    for idx_t, (tx, ty, sp, th) in enumerate(tm_trees):
        if sp == "cypress":
            build_cypress_tree(collector, tx, ty, z_esp, height=th, seed=800 + idx_t, with_kerb=True)
        else:
            build_olive_tree(collector, tx, ty, z_esp, height=th, seed=800 + idx_t, with_kerb=True)

    for lx, ly in ((720.7, 65.0), (755.0, 65.0), (708.0, 252.0), (734.0, 252.0), (695.0, 155.7), (775.0, 142.0), (746.0, 12.0), (778.0, 12.0)):
        build_street_lamp(collector, lx, ly, z_up if (682.0 <= lx <= 788.0 and 92.0 <= ly <= 232.0) else z_esp, height=4.6)

    return 24


# ==============================================================================
# 9. REGION 8: MUSLIM QUARTER, DAMASCUS GATE, HEROD'S GATE, LIONS' GATE, VIA DOLOROSA & ST. ANNE (`r08_muslim_quarter_north_gates`)
# ==============================================================================

def _build_muslim_quarter_landmarks(collector: MeshCollector, sign_atlas: Dict[str, dict]) -> None:
    """
    Builds the signature landmarks of the Muslim Quarter & Northern/Eastern Gates (`r08`):
      1. **Damascus Gate Amphitheater Steps Plaza (`(254.1, 578.0)`)** & Hadrian's Roman Arch
      2. **Ecce Homo Arch (`(498.7, 413.4)`)** spanning across the Via Dolorosa
      3. **Austrian Hospice (`(445.0, 398.0)`)** at the Via Dolorosa / El-Wad intersection
      4. **Crusader Church of St. Anne (`(835.2, 528.7)`) & Pools of Bethesda (`(786.5, 537.5)`)**
      5. **Suq al-Qattanin (Cotton Merchants' Market, `x = 612..665, y = 152.7`)**
    """
    # 1. Damascus Gate Stepped Amphitheater Forecourt, Bridge Railings & Hadrian's Roman Eastern Arch (`(254.1, 557.2)`)
    dg_x, dg_y = 254.1, 557.2
    dg_z = street_elevation_z(dg_x, dg_y)
    # Stepped semicircular amphitheater tiers flanking the West & East sides of the forecourt (`y = 568..586`)
    for side_s in (-1.0, 1.0):
        for tier_i in range(6):
            tx_step = dg_x + side_s * (10.5 + tier_i * 1.45)
            ty_step = dg_y + 18.5 + tier_i * 0.65
            tz_step = dg_z + (tier_i + 1) * 0.24
            collector.add_box(
                "Ground_Steps",
                "mat_stone_mizzi" if tier_i % 2 == 0 else "mat_paving_flagstone",
                (tx_step, ty_step, dg_z + 0.5 * (tier_i + 1) * 0.24),
                (1.6, 16.5 - tier_i * 0.8, (tier_i + 1) * 0.24),
                rot_z=math.radians(-18.0 * side_s),
                uv_scale=0.30,
            )
    # Black wrought-iron bridge railings flanking the approach bridge into the Damascus Gate portal (`y = 564..576`)
    for br_side in (-3.2, 3.2):
        bx_r = dg_x + br_side
        collector.add_box("StreetProps", "mat_iron_dark", (bx_r, dg_y + 13.5, dg_z + 0.98), (0.08, 12.0, 0.08))
        collector.add_box("StreetProps", "mat_iron_dark", (bx_r, dg_y + 13.5, dg_z + 0.52), (0.06, 12.0, 0.06))
        for p_idx in range(7):
            py_r = dg_y + 8.0 + p_idx * 1.85
            collector.add_box("StreetProps", "mat_iron_dark", (bx_r, py_r, dg_z + 0.50), (0.09, 0.09, 1.0))
    # Hadrian's 2nd-Century Roman Eastern Triumphal Arch (`Aelia Capitolina`) visible at the base of the East Tower (`(261.2, 562.2)`)
    rad_dg = math.radians(90.0 - 162.0)
    ox_dg, oy_dg = -math.cos(rad_dg), -math.sin(rad_dg)
    rx_dg, ry_dg = math.sin(rad_dg), -math.cos(rad_dg)
    had_x = dg_x - rx_dg * 5.8 + ox_dg * 6.85
    had_y = dg_y - ry_dg * 5.8 + oy_dg * 6.85
    add_arched_opening_panel(
        collector,
        "Buildings_Openings",
        "mat_stone_ashlar_aged",
        (had_x, had_y),
        dg_z - 0.5,
        dg_z + 2.2,
        1.35,
        2.7,
        0.02,
        (rx_dg, ry_dg),
        (ox_dg, oy_dg),
        uv_rect=(0.0, 0.0, 1.0, 1.0),
        num_voussoirs=9,
        add_wall_spandrel=False,
    )
    add_arch_voussoirs(
        collector,
        "CityWalls",
        "mat_stone_mizzi",
        (had_x, had_y, dg_z + 2.2),
        (rx_dg, ry_dg),
        (ox_dg, oy_dg),
        span_w=2.9,
        arch_h=1.45,
        ring_thickness=0.44,
        offset_n0=-0.04,
        offset_n1=0.32,
        num_voussoirs=9,
        keystone_boost=0.06,
    )

    # 2. Ecce Homo Roman Triumphal Arch (`(498.7, 413.4)`) spanning across Via Dolorosa
    ex, ey = 498.7, 413.4
    ez = street_elevation_z(ex, ey)
    # North & South stone piers framing the Via Dolorosa street (`y = ey - 4.2 .. ey + 4.2`)
    for side_y in (-3.8, 3.8):
        collector.add_box("Buildings_Walls", "mat_stone_boss", (ex, ey + side_y, ez + 4.2), (2.8, 2.2, 8.4), uv_scale=0.22)
    # 3D Roman semicircular stone arch ring & spandrel wall spanning the street (`ez + 3.2 .. ez + 5.4m`)
    for nx_dir in (-1.0, 1.0):
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (ex + nx_dir * 1.25, ey, ez + 3.2),
            (0.0, nx_dir),
            (nx_dir, 0.0),
            span_w=5.4,
            arch_h=2.0,
            ring_thickness=0.45,
            offset_n0=-0.15,
            offset_n1=0.22,
            num_voussoirs=11,
        )
    add_arch_spandrel_wall(
        collector,
        "Buildings_Walls",
        "mat_stone_mizzi",
        (ex, ey, ez + 3.2),
        (0.0, 1.0),
        (1.0, 0.0),
        span_w=5.5,
        arch_h=2.0,
        z_top=ez + 5.4,
        offset_n0=-1.3,
        offset_n1=1.3,
        num_voussoirs=10,
    )
    # Upper gallery bridge spanning across the street (`z = ez + 5.3 .. ez + 9.6m`)
    collector.add_box(
        "Buildings_Walls",
        "mat_stone_mizzi",
        (ex, ey, ez + 7.45),
        (2.6, 9.8, 4.3),
        uv_scale=0.24,
        bottom_face=True,
    )
    collector.add_box(
        "Buildings_Details",
        "mat_stone_boss",
        (ex, ey, ez + 9.75),
        (2.9, 10.2, 0.40),
        uv_scale=0.28,
        bottom_face=True,
    )
    # Projecting Roman entablature cornice below the upper gallery windows + engaged Corinthian pilasters
    collector.add_box(
        "Buildings_Details",
        "mat_stone_boss",
        (ex, ey, ez + 5.55),
        (2.95, 10.1, 0.32),
        uv_scale=0.30,
        bottom_face=True,
    )
    for pil_y in (-2.95, 2.95):
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (ex - 1.38, ey + pil_y, ez + 2.65), (0.32, 0.58, 5.3), uv_scale=0.32)
    # Smaller Northern Roman Sidewalk Arch (`(ex - 1.32, ey + 3.85)`) from Hadrian's triple triumphal arch
    add_arched_opening_panel(
        collector,
        "Buildings_Openings",
        "mat_wood_doors",
        (ex - 1.42, ey + 3.85),
        ez,
        ez + 2.2,
        0.95,
        1.90,
        0.01,
        (0.0, -1.0),
        (-1.0, 0.0),
        uv_rect=WOOD_DOOR_UV["door_wood"],
        num_voussoirs=7,
        add_wall_spandrel=False,
    )
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        "mat_stone_boss",
        (ex - 1.42, ey + 3.85, ez + 2.2),
        (0.0, -1.0),
        (-1.0, 0.0),
        span_w=1.90,
        arch_h=0.95,
        ring_thickness=0.32,
        offset_n0=-0.04,
        offset_n1=0.22,
        num_voussoirs=7,
    )
    # Twin arched windows on both the East and West faces of the Ecce Homo bridge (`arch_spring_z = ez + 7.9`, zero-gap + paired louvred shutters!)
    for nx_dir in (-1.0, 1.0):
        for wy_off in (-1.4, 1.4):
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (ex + nx_dir * 1.31, ey + wy_off),
                ez + 5.8,
                ez + 7.9,
                0.65,
                1.35,
                0.01,
                (0.0, nx_dir),
                (nx_dir, 0.0),
                uv_rect=WOOD_DOOR_UV["window_glass"],
                num_voussoirs=5,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (ex + nx_dir * 1.31, ey + wy_off, ez + 7.9),
                (0.0, nx_dir),
                (nx_dir, 0.0),
                span_w=1.35,
                arch_h=0.65,
                ring_thickness=0.26,
                offset_n0=-0.04,
                offset_n1=0.18,
                num_voussoirs=5,
            )
            for sh_s in (-0.86, 0.86):
                add_arched_opening_panel(
                    collector,
                    "Buildings_Openings",
                    "mat_wood_doors",
                    (ex + nx_dir * 1.35, ey + wy_off + sh_s),
                    ez + 5.82,
                    ez + 7.85,
                    0.02,
                    0.32,
                    0.02,
                    (0.0, nx_dir),
                    (nx_dir, 0.0),
                    uv_rect=WOOD_DOOR_UV["shutter_louvre"],
                    num_voussoirs=2,
                    add_wall_spandrel=False,
                )
    # Flanking Convent of the Sisters of Zion (`(492.0, 422.5)`) & southern street canyon wall (`(492.0, 404.3)`)
    # Articulated with string courses, roof cornices, recessed convent portals, shuttered upper windows, wall lanterns & Via Dolorosa plaques!
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (ex - 5.0, ey + 9.2, ez + 5.8), (28.0, 9.0, 11.6), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_stone_boss", (ex - 5.0, ey + 9.2, ez + 4.4), (28.3, 9.3, 0.28), uv_scale=0.30)
    collector.add_box("Buildings_Details", "mat_stone_boss", (ex - 5.0, ey + 9.2, ez + 11.65), (28.5, 9.5, 0.38), uv_scale=0.30)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (ex - 5.0, ey - 9.2, ez + 5.2), (28.0, 9.0, 10.4), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_stone_boss", (ex - 5.0, ey - 9.2, ez + 4.2), (28.3, 9.3, 0.28), uv_scale=0.30)
    collector.add_box("Buildings_Details", "mat_stone_boss", (ex - 5.0, ey - 9.2, ez + 10.45), (28.5, 9.5, 0.38), uv_scale=0.30)
    for side_v, vy_face, v_ny, v_tx in ((1.0, ey + 4.69, -1.0, -1.0), (-1.0, ey - 4.69, 1.0, 1.0)):
        for vx_off in (-14.5, -8.5, -3.2):
            wx_v = ex + vx_off
            wz_v = street_elevation_z(wx_v, ey)
            # Ground-level arched convent doorway / shop niche
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (wx_v, vy_face),
                wz_v,
                wz_v + 2.5,
                1.05,
                2.10,
                0.01,
                (v_tx, 0.0),
                (0.0, v_ny),
                uv_rect=WOOD_DOOR_UV["door_wood" if vx_off == -8.5 else "window_glass"],
                num_voussoirs=7,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (wx_v, vy_face, wz_v + 2.5),
                (v_tx, 0.0),
                (0.0, v_ny),
                span_w=2.10,
                arch_h=1.05,
                ring_thickness=0.32,
                offset_n0=-0.04,
                offset_n1=0.20,
                num_voussoirs=7,
            )
            # Upper-floor arched window with stone sill & paired louvred shutters
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (wx_v, vy_face),
                wz_v + 5.2,
                wz_v + 7.4,
                0.78,
                1.55,
                0.01,
                (v_tx, 0.0),
                (0.0, v_ny),
                uv_rect=WOOD_DOOR_UV["window_glass"],
                num_voussoirs=7,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (wx_v, vy_face, wz_v + 7.4),
                (v_tx, 0.0),
                (0.0, v_ny),
                span_w=1.55,
                arch_h=0.78,
                ring_thickness=0.28,
                offset_n0=-0.04,
                offset_n1=0.18,
                num_voussoirs=7,
            )
            for sh_v in (-0.98, 0.98):
                add_arched_opening_panel(
                    collector,
                    "Buildings_Openings",
                    "mat_wood_doors",
                    (wx_v + sh_v, vy_face + v_ny * 0.05),
                    wz_v + 5.22,
                    wz_v + 7.35,
                    0.02,
                    0.36,
                    0.02,
                    (v_tx, 0.0),
                    (0.0, v_ny),
                    uv_rect=WOOD_DOOR_UV["shutter_louvre"],
                    num_voussoirs=2,
                    add_wall_spandrel=False,
                )
        # Wall-mounted wrought-iron Via Dolorosa lantern & cobalt-blue Station II plaque
        build_wall_lantern(collector, (ex - 6.2, vy_face + v_ny * 0.08, ez + 3.85), (v_tx, 0.0), (0.0, v_ny))
        collector.add_box("Buildings_Details", "mat_tile_iznik", (ex - 11.2, vy_face + v_ny * 0.08, ez + 3.2), (0.75, 0.08, 0.48), uv_scale=0.5)

    # 3. Austrian Hospice (`(448.0, 412.0)`) & Via Dolorosa Station III at the Via Dolorosa / El-Wad Corner
    ah_x, ah_y = 448.0, 412.0
    ah_z = street_elevation_z(ah_x, ah_y)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (ah_x, ah_y, ah_z + 6.2), (24.0, 18.0, 12.4), uv_scale=0.22)
    collector.add_box("Buildings_Details", "mat_stone_boss", (ah_x, ah_y, ah_z + 5.05), (24.3, 18.3, 0.26), uv_scale=0.30)
    collector.add_box("Buildings_Details", "mat_terracotta", (ah_x, ah_y, ah_z + 12.5), (24.6, 18.6, 0.45), uv_scale=0.30)
    # Western & Southern arched windows + wrought-iron balcony + Station III Chapel Portal facing `(422, 398)`
    for wy_off in (-5.2, 0.0, 5.2):
        for wz_lvl, is_portal in ((0.0, wy_off == 0.0), (5.8, False)):
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (ah_x - 12.01, ah_y + wy_off),
                ah_z + (0.0 if is_portal else wz_lvl + 1.2),
                ah_z + (3.2 if is_portal else wz_lvl + 3.8),
                1.10,
                2.20,
                0.01,
                (0.0, -1.0),
                (-1.0, 0.0),
                uv_rect=WOOD_DOOR_UV["door_wood" if is_portal else "window_glass"],
                num_voussoirs=7,
                add_wall_spandrel=False,
            )
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_boss",
                (ah_x - 12.01, ah_y + wy_off, ah_z + (3.2 if is_portal else wz_lvl + 3.8)),
                (0.0, -1.0),
                (-1.0, 0.0),
                span_w=2.20,
                arch_h=1.10,
                ring_thickness=0.34,
                offset_n0=-0.04,
                offset_n1=0.20,
                num_voussoirs=7,
            )
            if not is_portal:
                collector.add_box(
                    "Buildings_Details",
                    "mat_stone_mizzi",
                    (ah_x - 12.14, ah_y + wy_off, ah_z + wz_lvl + 1.12),
                    (0.24, 2.45, 0.16),
                    uv_scale=0.35,
                )
                for sh_s in (-1.34, 1.34):
                    add_arched_opening_panel(
                        collector,
                        "Buildings_Openings",
                        "mat_wood_doors",
                        (ah_x - 12.08, ah_y + wy_off + sh_s),
                        ah_z + wz_lvl + 1.22,
                        ah_z + wz_lvl + 3.75,
                        0.02,
                        0.44,
                        0.03,
                        (0.0, -1.0),
                        (-1.0, 0.0),
                        uv_rect=WOOD_DOOR_UV["shutter_louvre"],
                        num_voussoirs=2,
                        add_wall_spandrel=False,
                    )
    # Moulded stone balcony slab (`ah_z + 5.05m`) on 4 stone corbels + openwork wrought-iron railing + Austrian flag
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (ah_x - 12.62, ah_y, ah_z + 5.05), (1.25, 6.4, 0.26), uv_scale=0.35, bottom_face=True)
    for cb_y in (-2.6, -0.9, 0.9, 2.6):
        collector.add_box("Buildings_Details", "mat_stone_boss", (ah_x - 12.42, ah_y + cb_y, ah_z + 4.72), (0.82, 0.28, 0.42), uv_scale=0.35, bottom_face=True)
    collector.add_box("StreetProps", "mat_iron_dark", (ah_x - 13.18, ah_y, ah_z + 5.98), (0.08, 6.3, 0.08))
    collector.add_box("StreetProps", "mat_iron_dark", (ah_x - 13.18, ah_y, ah_z + 5.55), (0.06, 6.3, 0.06))
    for rp_y in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
        collector.add_box("StreetProps", "mat_iron_dark", (ah_x - 13.18, ah_y + rp_y, ah_z + 5.56), (0.07, 0.07, 0.88))
    # Red-white-red Austrian Flag flying above the balcony
    collector.add_branch_tube("StreetProps", "mat_iron_dark", (ah_x - 12.1, ah_y, ah_z + 6.2), (ah_x - 13.8, ah_y, ah_z + 7.6), 0.05, 0.04, segments=6)
    collector.add_box("StreetProps", "mat_awning_red", (ah_x - 13.5, ah_y + 0.05, ah_z + 7.05), (0.95, 0.06, 0.66), uv_scale=0.5)
    collector.add_box("StreetProps", "mat_stone_mizzi", (ah_x - 13.5, ah_y + 0.05, ah_z + 7.05), (0.96, 0.08, 0.22), uv_scale=0.5)
    # Double exterior entrance steps & wrought-iron handrails at the Austrian Hospice portal (`(435.5, 412.0)`)
    collector.add_box("Ground_Steps", "mat_stone_mizzi", (ah_x - 12.65, ah_y, ah_z + 0.16), (1.3, 3.6, 0.32), uv_scale=0.35)
    collector.add_box("Ground_Steps", "mat_stone_mizzi", (ah_x - 13.15, ah_y, ah_z + 0.08), (2.2, 4.2, 0.16), uv_scale=0.35)
    # Station III Polish Chapel Arched Relief Portal adjacent to the Austrian Hospice (`(436.0, 402.2)`, facing SW)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (436.0, 401.8, ah_z + 3.4), (5.2, 4.8, 6.8), uv_scale=0.25)
    collector.add_box("Buildings_Details", "mat_stone_boss", (436.0, 401.8, ah_z + 6.95), (5.6, 5.2, 0.35), uv_scale=0.30)
    add_arched_opening_panel(
        collector,
        "Buildings_Openings",
        "mat_wood_doors",
        (433.39, 401.8),
        ah_z,
        ah_z + 2.6,
        1.05,
        2.10,
        0.01,
        (0.0, -1.0),
        (-1.0, 0.0),
        uv_rect=WOOD_DOOR_UV["door_wood"],
        num_voussoirs=7,
        add_wall_spandrel=False,
    )
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        "mat_stone_boss",
        (433.39, 401.8, ah_z + 2.6),
        (0.0, -1.0),
        (-1.0, 0.0),
        span_w=2.10,
        arch_h=1.05,
        ring_thickness=0.34,
        offset_n0=-0.04,
        offset_n1=0.22,
        num_voussoirs=7,
    )
    # Carved semicircular stone tympanum relief plaque above Station III portal
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (433.30, 401.8, ah_z + 4.35), (0.18, 1.85, 0.85), uv_scale=0.35)

    # Via Dolorosa Eastern Street Corridor (`(650.0, 462.0) -> (750.0, 462.0)`)
    _build_souq_street_corridor(collector, (650.0, 462.0), (750.0, 462.0), street_w=6.2, wall_h=8.4, with_arches=True, with_shops=True, seed=801)

    # NE Corner Citadel Bastion: Burj al-Laqlaq (`Stork Tower` at `(815.0, 768.0)`)
    st_x, st_y = 815.0, 768.0
    st_z = street_elevation_z(st_x, st_y)
    # Grounded 3.0m deep into the sloped NE corner terrain (`st_z - 3.0 .. st_z + 15.2m`) + battered bossed lower plinth skirt (zero base seam!)
    collector.add_box("CityWalls", "mat_stone_boss", (st_x, st_y, st_z + 0.4), (15.6, 15.6, 6.8), uv_scale=0.20)
    collector.add_box("CityWalls", "mat_stone_boss", (st_x, st_y, st_z + 6.1), (15.0, 15.0, 18.2), uv_scale=0.22)
    collector.add_box("CityWalls", "mat_stone_mizzi", (st_x, st_y, st_z + 8.35), (15.4, 15.4, 0.32), uv_scale=0.30)
    collector.add_box("CityWalls", "mat_stone_mizzi", (st_x, st_y, st_z + 15.4), (15.6, 15.6, 0.45), uv_scale=0.30)
    for mx_off in (-5.6, -2.8, 0.0, 2.8, 5.6):
        collector.add_box("CityWalls", "mat_stone_boss", (st_x + mx_off, st_y - 7.1, st_z + 16.2), (1.4, 0.8, 1.3), uv_scale=0.30)
        collector.add_box("CityWalls", "mat_stone_mizzi", (st_x + mx_off, st_y - 7.1, st_z + 16.92), (1.5, 0.9, 0.22), uv_scale=0.32)
        collector.add_box("CityWalls", "mat_stone_boss", (st_x - 7.1, st_y + mx_off, st_z + 16.2), (0.8, 1.4, 1.3), uv_scale=0.30)
        collector.add_box("CityWalls", "mat_stone_mizzi", (st_x - 7.1, st_y + mx_off, st_z + 16.92), (0.9, 1.5, 0.22), uv_scale=0.32)
    # Splayed vertical arrow-slit loopholes (`embrasures`), carved Ottoman circular stone rosettes & corbelled machicolation boxes on both faces
    for sl_off in (-4.2, 0.0, 4.2):
        for sl_z in (5.6, 10.6):
            # South face (`y = st_y - 7.52`)
            collector.add_box("CityWalls", "mat_stone_mizzi", (st_x + sl_off, st_y - 7.52, st_z + sl_z), (0.58, 0.18, 1.95), uv_scale=0.32)
            collector.add_box("Buildings_Openings", "mat_iron_dark", (st_x + sl_off, st_y - 7.58, st_z + sl_z), (0.20, 0.10, 1.68))
            # West face (`x = st_x - 7.52`)
            collector.add_box("CityWalls", "mat_stone_mizzi", (st_x - 7.52, st_y + sl_off, st_z + sl_z), (0.18, 0.58, 1.95), uv_scale=0.32)
            collector.add_box("Buildings_Openings", "mat_iron_dark", (st_x - 7.58, st_y + sl_off, st_z + sl_z), (0.10, 0.20, 1.68))
        if sl_off != 0.0:
            collector.add_branch_tube("CityWalls", "mat_stone_mizzi", (st_x + sl_off, st_y - 7.45, st_z + 12.8), (st_x + sl_off, st_y - 7.68, st_z + 12.8), 0.56, 0.52, segments=12, cap_end=True)
            collector.add_branch_tube("CityWalls", "mat_stone_mizzi", (st_x - 7.45, st_y + sl_off, st_z + 12.8), (st_x - 7.68, st_y + sl_off, st_z + 12.8), 0.56, 0.52, segments=12, cap_end=True)
    # Corbelled stone machicolation boxes centered on South and West faces of Stork Tower
    collector.add_box("CityWalls", "mat_stone_boss", (st_x, st_y - 7.85, st_z + 13.4), (3.4, 0.82, 1.95), uv_scale=0.25, bottom_face=True)
    collector.add_box("CityWalls", "mat_stone_boss", (st_x - 7.85, st_y, st_z + 13.4), (0.82, 3.4, 1.95), uv_scale=0.25, bottom_face=True)
    # Exterior Stone Rampart Staircase ascending the Eastern Curtain Wall (`x = 821.2, y = 746.5..759.5`) + Foreground Olive Terrace & Benches
    for st_i in range(12):
        sy_st = 746.5 + st_i * 1.12
        sx_st = 822.2 - st_i * 0.24
        sz_st = street_elevation_z(sx_st, sy_st)
        sh_st = (st_i + 1) * 0.62
        collector.add_box("Ground_Steps", "mat_stone_boss", (sx_st, sy_st, sz_st + 0.5 * sh_st), (2.1, 1.18, sh_st), uv_scale=0.26)
        collector.add_box("CityWalls", "mat_stone_mizzi", (sx_st - 0.92, sy_st, sz_st + sh_st + 0.42), (0.34, 1.18, 0.84), uv_scale=0.32)
    ter_x, ter_y = 805.5, 755.5
    ter_z = street_elevation_z(ter_x, ter_y)
    collector.add_box("StreetProps", "mat_stone_mizzi", (ter_x, ter_y, ter_z + 0.34), (5.8, 7.6, 0.68), uv_scale=0.32)
    collector.add_box("Trees_Foliage", "mat_foliage_olive", (ter_x, ter_y, ter_z + 0.72), (5.0, 6.8, 0.34), uv_scale=0.6)
    build_olive_tree(collector, ter_x, ter_y + 1.2, ter_z + 0.65, height=6.4, seed=915, with_kerb=False)
    collector.add_box("StreetProps", "mat_stone_boss", (ter_x - 3.4, ter_y - 1.2, ter_z + 0.28), (0.65, 2.2, 0.56), uv_scale=0.35)
    build_street_lamp(collector, ter_x - 3.6, ter_y + 2.8, ter_z, height=4.3)

    # 4. Crusader Church of St. Anne (`(835.2, 528.7)`) & Pools of Bethesda (`(786.5, 537.5)`)
    sa_x, sa_y = 835.2, 528.7
    sa_z = street_elevation_z(sa_x, sa_y)
    # Stepped Tripartite Romanesque Limestone Basilica:
    # Lower side aisles (`26m x 17.6m x 8.8m`) + taller raised central nave (`26m x 8.4m x 12.6m`)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (sa_x, sa_y, sa_z + 4.4), (26.0, 17.6, 8.8), uv_scale=0.22)
    # Side-aisle roof cornices ONLY on the North and South side aisles (never crossing the central nave window!)
    for aisle_s in (-6.55, 6.55):
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (sa_x, sa_y + aisle_s, sa_z + 8.95), (26.4, 4.9, 0.35), uv_scale=0.30)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (sa_x, sa_y, sa_z + 6.3), (26.0, 8.4, 12.6), uv_scale=0.22)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (sa_x, sa_y, sa_z + 12.75), (26.5, 8.9, 0.40), uv_scale=0.30)
    # Side-aisle arched Romanesque windows on the West facade (`sa_y - 6.4` and `sa_y + 6.4`)
    for aw_y in (-6.4, 6.4):
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (sa_x - 13.04, sa_y + aw_y),
            sa_z + 3.2,
            sa_z + 5.6,
            0.75,
            1.5,
            0.02,
            (0.0, -1.0),
            (-1.0, 0.0),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=7,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (sa_x - 13.04, sa_y + aw_y, sa_z + 5.6),
            (0.0, -1.0),
            (-1.0, 0.0),
            span_w=1.65,
            arch_h=0.82,
            ring_thickness=0.28,
            offset_n0=-0.02,
            offset_n1=0.18,
            num_voussoirs=7,
        )
    # 4 vertical Romanesque pilaster buttresses dividing the western facade into 3 bays
    for by_off, bh in ((-8.4, 8.8), (-4.1, 12.6), (4.1, 12.6), (8.4, 8.8)):
        collector.add_box(
            "Buildings_Details",
            "mat_stone_boss",
            (sa_x - 13.15, sa_y + by_off, sa_z + 0.5 * bh),
            (0.65, 0.85, bh),
            uv_scale=0.28,
        )
    # Western Crusader pointed portal & upper nave window (`sa_x - 13.01`, zero-gap + 2-order stepped Crusader archivolts!)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (sa_x - 13.10, sa_y, sa_z + 5.15), (0.28, 8.4, 0.24), uv_scale=0.35)
    for z0, arch_spring_z, arch_h, span_w, uv_k in (
        (sa_z, sa_z + 2.9, 1.50, 3.20, "door_wood"),
        (sa_z + 5.8, sa_z + 8.5, 1.30, 2.50, "window_glass"),
    ):
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (sa_x - 13.01, sa_y),
            z0,
            arch_spring_z,
            arch_h,
            span_w,
            0.01,
            (0.0, -1.0),
            (-1.0, 0.0),
            uv_rect=WOOD_DOOR_UV[uv_k],
            num_voussoirs=9,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (sa_x - 13.01, sa_y, arch_spring_z),
            (0.0, -1.0),
            (-1.0, 0.0),
            span_w=span_w,
            arch_h=arch_h,
            ring_thickness=0.38,
            offset_n0=-0.04,
            offset_n1=0.24,
            num_voussoirs=9,
        )
        if uv_k == "door_wood":
            # Outer stepped Crusader compound archivolt molding around the main West portal
            add_arch_voussoirs(
                collector,
                "Buildings_Details",
                "mat_stone_mizzi",
                (sa_x - 13.01, sa_y, arch_spring_z + 0.15),
                (0.0, -1.0),
                (-1.0, 0.0),
                span_w=span_w + 0.72,
                arch_h=arch_h + 0.42,
                ring_thickness=0.30,
                offset_n0=-0.02,
                offset_n1=0.34,
                num_voussoirs=11,
            )
    # Crossing Drum + Cupola Dome (`(sa_x + 4.5, sa_y)`) & Bended Bell Tower with belfry louvre arch & pyramidal cap
    collector.add_branch_tube(
        "Buildings_Walls",
        "mat_stone_mizzi",
        (sa_x + 4.5, sa_y, sa_z + 12.6),
        (sa_x + 4.5, sa_y, sa_z + 15.8),
        4.2,
        4.0,
        segments=12,
        cap_end=True,
    )
    collector.add_dome(
        "Buildings_Details",
        "mat_dome_lead",
        (sa_x + 4.5, sa_y, sa_z + 15.7),
        radius=4.05,
        height=3.8,
        rings=7,
        segments=14,
    )
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (sa_x - 10.5, sa_y - 6.5, sa_z + 11.4), (4.2, 4.2, 5.2), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (sa_x - 10.5, sa_y - 6.5, sa_z + 14.15), (4.6, 4.6, 0.35), uv_scale=0.30)
    collector.add_branch_tube("Buildings_Details", "mat_dome_lead", (sa_x - 10.5, sa_y - 6.5, sa_z + 14.3), (sa_x - 10.5, sa_y - 6.5, sa_z + 16.8), 2.4, 0.12, segments=4, smooth=False, cap_end=True)

    # Pools of Bethesda Archaeological Basin (`(786.5, 537.5)`)
    pb_x, pb_y = 786.5, 537.5
    pb_z = street_elevation_z(pb_x, pb_y)
    for wall_dx, wall_dy, sx, sy in (
        (0.0, -11.0, 26.0, 1.4),
        (0.0, 11.0, 26.0, 1.4),
        (-13.0, 0.0, 1.4, 22.0),
        (13.0, 0.0, 1.4, 22.0),
        (0.0, 0.0, 25.0, 1.6),
    ):
        collector.add_box("Buildings_Walls", "mat_stone_rubble", (pb_x + wall_dx, pb_y + wall_dy, pb_z + 0.9), (sx, sy, 1.8), uv_scale=0.25)


def build_region_r08_muslim_quarter_north_gates(
    collector: MeshCollector,
    osm_data: dict,
    sign_atlas: Dict[str, dict],
) -> int:
    """
    Builds Region 8 (`r08_muslim_quarter_north_gates`):
      - Continuous stone-paved ground surface across `[235..905] x [255..785]` (excluding Temple Mount `x >= 666 and y <= 385`)
      - Eastern & Northern Ottoman City Wall (`Segment 5`: NE Temple Mount -> Lions' Gate -> Stork Tower -> Herod's Gate -> Damascus Gate)
      - Three Monumental Ottoman Gates:
        1. **Damascus Gate (`Bab al-Amud` at `(254.1, 557.2)`, `twin_towers=True`)**
        2. **Herod's Gate (`Bab al-Zahra` at `(568.0, 702.3)`)**
        3. **Lions' Gate (`Bab al-Asbat` at `(875.8, 467.0)`)**
      - Via Dolorosa, Ecce Homo Arch, Austrian Hospice, Crusader Church of St. Anne & Pools of Bethesda
      - All valid `r08_muslim_quarter_north_gates` OSM buildings, trees, and street props
    """
    def r08_exclude(x: float, y: float) -> bool:
        # Exclude Temple Mount esplanade (`r07`) and Christian Quarter (`r02`)
        if x >= 666.0 and y <= 385.0:
            return True
        if x < 292.0 and y < 305.0:
            return True
        return False

    build_region_ground_grid(collector, 235.0, 905.0, 255.0, 785.0, step=6.5, exclude_fn=r08_exclude)

    # 1. Eastern & Northern Ottoman City Wall (`Segment 5` of relation `5862586`, decimated to key vertices)
    ne_wall_pts = [
        (887.3, 407.5),
        (878.8, 456.6),
        (875.8, 467.0), # Lions' Gate
        (871.9, 491.3),
        (863.2, 540.5),
        (850.5, 616.7),
        (838.6, 681.9),
        (824.9, 756.3),
        (817.5, 775.9), # NE Stork Tower (Burj al-Laqlaq)
        (781.9, 761.0),
        (736.7, 751.4),
        (687.2, 738.3),
        (648.9, 725.6),
        (600.2, 712.6),
        (568.0, 702.3), # Herod's Gate
        (553.0, 697.8),
        (519.1, 696.9),
        (482.5, 718.6),
        (450.2, 708.1),
        (423.4, 667.1),
        (395.9, 646.9),
        (356.7, 622.6),
        (320.2, 597.2),
        (281.5, 562.4),
        (254.1, 557.2), # Damascus Gate
        (242.0, 551.5),
        (236.5, 540.5),
    ]
    build_ottoman_wall_polyline(
        collector,
        ne_wall_pts,
        wall_height=13.2,
        thickness=3.5,
        gate_gaps=[
            (875.8, 467.0, 10.2),
            (568.0, 702.3, 10.8),
            (254.1, 557.2, 13.5),
        ],
        bastion_every=3,
    )

    # 2. Damascus Gate (`Bab al-Amud` at `(254.1, 557.2)`, facing SSE `162 deg` into Khan el-Zeit / El-Wad fork)
    build_ottoman_gatehouse(
        collector,
        (254.1, 557.2),
        heading_inward_deg=162.0,
        gate_name="Damascus Gate (Bab al-Amud)",
        width=22.0,
        depth=13.5,
        height=16.8,
        twin_towers=True,
    )

    # 3. Herod's Gate (`Bab al-Zahra` at `(568.0, 702.3)`, facing SSW `198 deg` into the Muslim Quarter)
    build_ottoman_gatehouse(
        collector,
        (568.0, 702.3),
        heading_inward_deg=198.0,
        gate_name="Herod's Gate (Bab al-Zahra)",
        width=15.5,
        depth=11.0,
        height=14.0,
    )

    # 4. Lions' Gate (`Bab al-Asbat` at `(875.8, 467.0)`, facing WSW `258 deg` down the Via Dolorosa)
    build_ottoman_gatehouse(
        collector,
        (875.8, 467.0),
        heading_inward_deg=258.0,
        gate_name="Lions' Gate (Bab al-Asbat)",
        width=16.5,
        depth=11.5,
        height=14.5,
    )

    # 5. Ecce Homo Arch, Damascus Gate Amphitheater, Austrian Hospice & St. Anne Church
    _build_muslim_quarter_landmarks(collector, sign_atlas)

    # 6. OSM Buildings in `r08_muslim_quarter_north_gates`
    clear_scene_building_rings()
    osm_buildings = [
        b for b in osm_data.get("buildings", [])
        if b.get("region") == "r08_muslim_quarter_north_gates"
    ]
    valid_blds = []
    for b in osm_buildings:
        osm_id = int(b.get("osm_id", 0))
        if osm_id in (5862586, 7880713, 126864879, 126864878):
            continue
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        ring = b.get("ring", [])
        if len(ring) < 3:
            continue
        rxs = [float(p[0]) for p in ring]
        rys = [float(p[1]) for p in ring]
        if (max(rxs) - min(rxs)) > 65.0 or (max(rys) - min(rys)) > 65.0:
            continue
        # Keep Damascus Gate plaza & inner fork (`x = 225..295, y = 510..595`) clear
        if _ring_overlaps_rect(ring, 225.0, 510.0, 295.0, 595.0):
            continue
        # Keep Herod's Gate portal & inner street (`x = 548..588, y = 672..725`) clear
        if _ring_overlaps_rect(ring, 548.0, 672.0, 588.0, 725.0):
            continue
        # Keep Lions' Gate & Via Dolorosa eastern corridor (`x = 640..895, y = 450..555`) clear
        if _ring_overlaps_rect(ring, 640.0, 450.0, 895.0, 555.0):
            continue
        # Keep Ecce Homo Arch & Austrian Hospice Via Dolorosa corridor (`x = 410..525, y = 382..438`) clear
        if _ring_overlaps_rect(ring, 410.0, 382.0, 525.0, 438.0):
            continue
        # Keep Stork Tower / Burj al-Laqlaq NE Bastion Plaza (`x = 780..832, y = 730..782`) clear
        if _ring_overlaps_rect(ring, 780.0, 730.0, 832.0, 782.0):
            continue
        # Keep Suq al-Qattanin Vaulted Cotton Market (`x = 558..670, y = 132..158`) clear of `r08` buildings!
        if _ring_overlaps_rect(ring, 558.0, 132.0, 670.0, 158.0):
            continue
        valid_blds.append(b)
        register_scene_building_ring(osm_id, ring)

    built_count = 0
    for idx, b in enumerate(valid_blds):
        ring = b.get("ring", [])
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        bz = street_elevation_z(cx, cy)
        lvls = max(2, min(3, int(b.get("levels", 2))))
        build_building_from_footprint(
            collector,
            ring,
            base_z=bz,
            levels=lvls,
            sign_atlas=sign_atlas,
            building_id=int(b.get("osm_id", idx)),
            is_hero=False,
        )
        built_count += 1

    # 7. Trees, Damascus Gate Inner Fork Market Stalls & Street Lamps across Damascus Gate, St. Anne, and Via Dolorosa
    mq_trees = [
        (232.0, 568.0, "olive", 6.8, 901),
        (278.0, 574.0, "cypress", 11.5, 902),
        (432.0, 424.0, "cypress", 11.0, 903),
        (555.0, 715.0, "olive", 6.5, 904),
        (812.0, 522.0, "cypress", 12.0, 905),
        (818.0, 538.0, "olive", 6.8, 906),
        (855.0, 482.0, "olive", 6.6, 907),
        (798.0, 755.0, "cypress", 11.8, 908),
    ]
    for tx, ty, sp, th, seed in mq_trees:
        tz = street_elevation_z(tx, ty)
        if sp == "cypress":
            build_cypress_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)
        else:
            build_olive_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)

    # Damascus Gate Inner Fork Market Stalls, Spice Displays & Bollards (`(252..270, 518..540)`)
    for st_idx, (st_x, st_y, st_rot) in enumerate((
        (253.5, 528.0, 15.0),
        (267.5, 526.0, -25.0),
        (262.0, 518.5, 90.0),
    )):
        st_z = street_elevation_z(st_x, st_y)
        rad_st = math.radians(st_rot)
        ca_s, sa_s = math.cos(rad_st), math.sin(rad_st)
        build_souq_display(
            collector,
            (st_x, st_y, st_z),
            (ca_s, sa_s),
            (-sa_s, ca_s),
            width=2.1,
            seed=920 + st_idx,
        )
        collector.add_box(
            "StreetProps",
            ("mat_awning_striped", "mat_awning_red", "mat_awning_green")[st_idx % 3],
            (st_x, st_y, st_z + 2.45),
            (2.4, 1.8, 0.14),
            rot_z=rad_st,
            uv_scale=0.45,
            bottom_face=True,
        )
        # 4 grounded corner iron posts supporting each market stall canopy
        for px_o in (-1.08, 1.08):
            for py_o in (-0.78, 0.78):
                wx_p = st_x + px_o * ca_s - py_o * sa_s
                wy_p = st_y + px_o * sa_s + py_o * ca_s
                collector.add_box("StreetProps", "mat_iron_dark", (wx_p, wy_p, st_z + 1.22), (0.06, 0.06, 2.44))
    for b_dg_x in (256.0, 260.0, 264.0, 268.0):
        build_bollard(collector, b_dg_x, 542.5, street_elevation_z(b_dg_x, 542.5))

    for lx, ly in ((254.0, 538.0), (268.0, 534.0), (266.0, 574.0), (428.0, 398.0), (568.0, 688.0), (858.0, 466.0), (818.0, 521.0)):
        build_street_lamp(collector, lx, ly, street_elevation_z(lx, ly), height=4.4)

    return built_count


def build_additional_region(
    region_id: str,
    collector: MeshCollector,
    osm_data: dict,
    sign_atlas: Dict[str, dict],
) -> int:
    if region_id == "r03_central_souqs":
        return build_region_r03_central_souqs(collector, osm_data, sign_atlas)
    if region_id == "r04_armenian_quarter":
        return build_region_r04_armenian_quarter(collector, osm_data, sign_atlas)
    if region_id == "r05_jewish_quarter":
        return build_region_r05_jewish_quarter(collector, osm_data, sign_atlas)
    if region_id == "r06_western_wall_dung_gate":
        return build_region_r06_western_wall_dung_gate(collector, osm_data, sign_atlas)
    if region_id == "r07_temple_mount":
        return build_region_r07_temple_mount(collector, osm_data, sign_atlas)
    if region_id == "r08_muslim_quarter_north_gates":
        return build_region_r08_muslim_quarter_north_gates(collector, osm_data, sign_atlas)
    raise ValueError(f"Unknown region_id: {region_id}")

