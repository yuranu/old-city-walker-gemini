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

        # Subdivide long wall segments into <= 18m sub-segments so the wall hugs topography smoothly
        n_sub = max(1, int(math.ceil(seg_len / 18.0)))
        nx, ny = dy / seg_len, -dx / seg_len

        for s in range(n_sub):
            t0 = s / n_sub
            t1 = (s + 1) / n_sub
            sp0 = (p0[0] + dx * t0, p0[1] + dy * t0)
            sp1 = (p0[0] + dx * t1, p0[1] + dy * t1)
            mx, my = 0.5 * (sp0[0] + sp1[0]), 0.5 * (sp0[1] + sp1[1])

            if any(math.hypot(mx - gx, my - gy) < grad for gx, gy, grad in gate_gaps):
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

    if sealed_double_arch:
        # Golden Gate (Bab al-Rahma): Monumental Byzantine/Umayyad/Ottoman double-arched sealed gatehouse
        collector.add_box(
            "CityWalls",
            "mat_stone_ashlar",
            (cx, cy, cz + height * 0.5 - 0.5),
            (width, depth, height + 1.0),
            rot_z=rot_z,
            uv_scale=0.24,
        )
        for sign_s in (-1.0, 1.0):
            ax = cx + rx * (sign_s * 3.1) + ox * (depth * 0.5 + 0.05)
            ay = cy + ry * (sign_s * 3.1) + oy * (depth * 0.5 + 0.05)
            add_arch_voussoirs(
                collector,
                "CityWalls",
                "mat_stone_boss",
                (ax, ay, cz + 4.2),
                (rx, ry),
                (ox, oy),
                span_w=4.4,
                arch_h=2.1,
                ring_thickness=0.45,
                offset_n0=-0.05,
                offset_n1=0.32,
                num_voussoirs=11,
                keystone_boost=0.06,
            )
            add_arched_opening_panel(
                collector,
                "CityWalls",
                "mat_stone_mizzi",
                (ax, ay),
                cz,
                cz + 4.2,
                2.0,
                4.3,
                0.02,
                (rx, ry),
                (ox, oy),
                uv_rect=(0.0, 0.0, 1.0, 1.0),
                num_voussoirs=9,
                add_wall_spandrel=False,
            )
        # Twin low lead domes on the roof of Golden Gate
        for sign_s in (-1.0, 1.0):
            collector.add_dome(
                "CityWalls",
                "mat_dome_lead",
                (cx + rx * (sign_s * 3.1), cy + ry * (sign_s * 3.1), cz + height + 0.4),
                radius=2.6,
                height=2.2,
                rings=5,
                segments=12,
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
            collector.add_box(
                "CityWalls",
                "mat_stone_boss",
                (tx_b, ty_b, cz + (height + 2.4) * 0.5 - 0.5),
                (6.8, depth + 4.4, height + 3.4),
                rot_z=rot_z,
                uv_scale=0.22,
            )

    # 2. Upper Gatehouse Chamber above the arched portal (`z = cz + arch_h .. cz + height`)
    collector.add_box(
        "CityWalls",
        "mat_stone_ashlar_aged",
        (cx, cy, cz + arch_h + (height - arch_h) * 0.5),
        (arch_w + 0.6, depth, height - arch_h),
        rot_z=rot_z,
        uv_scale=0.24,
        bottom_face=True,
    )

    # 3. Outer & Inner Monumental Pointed Stone Arch Voussoirs
    for face_sign, n_vec in ((1.0, (ox, oy)), (-1.0, (fx, fy))):
        portal_x = cx + n_vec[0] * (depth * 0.5)
        portal_y = cy + n_vec[1] * (depth * 0.5)
        t_vec = (rx * face_sign, ry * face_sign)
        add_arch_voussoirs(
            collector,
            "CityWalls",
            "mat_stone_boss",
            (portal_x, portal_y, cz + 3.4),
            t_vec,
            n_vec,
            span_w=arch_w,
            arch_h=2.0,
            ring_thickness=0.48,
            offset_n0=-0.05,
            offset_n1=0.36,
            num_voussoirs=11,
            keystone_boost=0.07,
        )

    # 4. Projecting Corbelled Machicolation Box above Outer Portal
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

    # 5. Crenellated Merlons atop the Gatehouse Crown
    n_merlons = 5
    for m_idx in range(n_merlons):
        ms = -width * 0.42 + (m_idx / (n_merlons - 1)) * (width * 0.84)
        mx = cx + rx * ms + ox * (depth * 0.38)
        my = cy + ry * ms + oy * (depth * 0.38)
        collector.add_box(
            "CityWalls",
            "mat_stone_boss",
            (mx, my, cz + height + 0.75),
            (1.35, 0.85, 1.5),
            rot_z=rot_z,
            uv_scale=0.26,
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

    # 2. Main Basilica Nave & Transept Mass (`x = 166..218, y = 178..214`)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar_aged", (196.0, 196.0, parvis_z + 7.2), (44.0, 34.0, 14.4), uv_scale=0.24)
    # Upper Clerestory & Roof Cornice
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (196.0, 196.0, parvis_z + 14.5), (44.6, 34.6, 0.45), uv_scale=0.32)

    # 3. Crusader Double-Arched Southern Facade (`y = 178.5, x = 183..211`, facing South `(0, -1)`)
    facade_y = 178.6
    # Lower and upper horizontal carved string courses across the Crusader facade
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (197.0, facade_y - 0.12, parvis_z + 6.8), (26.0, 0.35, 0.32), uv_scale=0.35)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (197.0, facade_y - 0.14, parvis_z + 13.8), (26.4, 0.42, 0.38), uv_scale=0.35)

    # Twin Crusader Portals: West (`x = 193.2`, open entrance) & East (`x = 200.8`, walled-up in 1187)
    for idx_p, px in enumerate((193.2, 200.8)):
        # Double-order cushion voussoir archivolts
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (px, facade_y, parvis_z + 3.8),
            (1.0, 0.0),
            (0.0, -1.0),
            span_w=4.4,
            arch_h=2.2,
            ring_thickness=0.52,
            offset_n0=-0.04,
            offset_n1=0.36,
            num_voussoirs=11,
            keystone_boost=0.06,
        )
        # Clustered marble colonnettes flanking the portal jambs
        for side_c in (-2.35, -2.05, 2.05, 2.35):
            collector.add_branch_tube(
                "Buildings_Details",
                "mat_stone_mizzi",
                (px + side_c, facade_y - 0.22, parvis_z),
                (px + side_c, facade_y - 0.22, parvis_z + 3.8),
                0.14,
                0.13,
                segments=8,
            )
        if idx_p == 0:
            # West Portal: Open dark arched entrance with monumental timber doors
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (px, facade_y - 0.04),
                parvis_z,
                parvis_z + 3.8,
                2.0,
                4.1,
                0.02,
                (1.0, 0.0),
                (0.0, -1.0),
                uv_rect=WOOD_DOOR_UV["door_wood"],
                num_voussoirs=9,
                add_wall_spandrel=False,
            )
        else:
            # East Portal: Walled-up Crusader archway
            add_arched_opening_panel(
                collector,
                "Buildings_Details",
                "mat_stone_ashlar",
                (px, facade_y - 0.04),
                parvis_z,
                parvis_z + 3.8,
                2.0,
                4.1,
                0.02,
                (1.0, 0.0),
                (0.0, -1.0),
                uv_rect=(0.0, 0.0, 1.0, 1.0),
                num_voussoirs=9,
                add_wall_spandrel=False,
            )

        # Upper Story Twin Crusader Pointed Windows (`z = parvis_z + 7.6 .. 12.2`)
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (px, facade_y, parvis_z + 10.2),
            (1.0, 0.0),
            (0.0, -1.0),
            span_w=3.6,
            arch_h=1.8,
            ring_thickness=0.42,
            offset_n0=-0.04,
            offset_n1=0.30,
            num_voussoirs=9,
            keystone_boost=0.05,
        )
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (px, facade_y - 0.04),
            parvis_z + 7.4,
            parvis_z + 10.2,
            1.65,
            3.3,
            0.02,
            (1.0, 0.0),
            (0.0, -1.0),
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=9,
            add_wall_spandrel=False,
        )

    # Iconic Wooden "Immovable Ladder" resting on the upper ledge below the right window (`x = 200.8, z = parvis_z + 7.1`)
    collector.add_box(
        "Buildings_Details",
        "mat_bark",
        (201.2, facade_y - 0.28, parvis_z + 8.1),
        (0.45, 0.12, 1.85),
    )

    # 4. Chapel of the Franks (East Side of Southern Facade, `x = 208.5..213.5, y = 173..178.6`)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (210.8, 175.8, parvis_z + 4.2), (5.2, 5.4, 8.4), uv_scale=0.25)
    collector.add_dome("Buildings_Details", "mat_dome_lead", (210.8, 175.8, parvis_z + 8.4), radius=2.1, height=1.8, rings=5, segments=12)
    for st_i in range(10):
        collector.add_box(
            "Buildings_Details",
            "mat_stone_mizzi",
            (207.2 + st_i * 0.35, 174.2, parvis_z + (st_i + 1) * 0.18),
            (0.42, 1.8, (st_i + 1) * 0.36),
        )

    # 5. 12th-Century Crusader Bell Tower (`x = 185.5, y = 180.5`, rising to `parvis_z + 26.5m`)
    bt_x, bt_y = 185.2, 180.2
    collector.add_box("Buildings_Walls", "mat_stone_boss", (bt_x, bt_y, parvis_z + 7.5), (8.6, 8.6, 15.0), uv_scale=0.22)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (bt_x, bt_y, parvis_z + 18.0), (8.0, 8.0, 6.0), uv_scale=0.24)
    collector.add_box("Buildings_Walls", "mat_stone_ashlar", (bt_x, bt_y, parvis_z + 23.5), (7.4, 7.4, 5.0), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_stone_mizzi", (bt_x, bt_y, parvis_z + 26.2), (7.9, 7.9, 0.45), uv_scale=0.32)
    # Coupled Gothic belfry arches on South & East faces of the Bell Tower
    for b_z in (parvis_z + 16.5, parvis_z + 22.0):
        for off_x in (-1.6, 1.6):
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (bt_x + off_x, bt_y - 4.04),
                b_z,
                b_z + 2.2,
                0.95,
                1.8,
                0.02,
                (1.0, 0.0),
                (0.0, -1.0),
                uv_rect=WOOD_DOOR_UV["shutter_louvre"],
                num_voussoirs=7,
                add_wall_spandrel=False,
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
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (red_x, red_y, red_z + 6.8), (24.0, 16.5, 13.6), uv_scale=0.24)
    # Pitched terracotta gable roof
    collector.add_box("Buildings_Details", "mat_roof_tile", (red_x, red_y, red_z + 14.4), (24.4, 15.0, 1.8), uv_scale=0.30)
    # Soaring 38m White Bell Tower at the West front (`(266.8, 141.5)`)
    tw_x, tw_y = 266.8, 141.5
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (tw_x, tw_y, red_z + 14.0), (6.6, 6.6, 28.0), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_stone_boss", (tw_x, tw_y, red_z + 28.2), (7.2, 7.2, 0.65), uv_scale=0.28)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (tw_x, tw_y, red_z + 31.2), (5.8, 5.8, 5.4), uv_scale=0.24)
    collector.add_dome("Buildings_Details", "mat_dome_lead", (tw_x, tw_y, red_z + 33.9), radius=3.4, height=4.8, rings=6, segments=8, smooth=False)

    # Belfry arched windows on all 4 faces of the Redeemer Bell Tower
    for face_ang in (0.0, 90.0, 180.0, 270.0):
        rad = math.radians(face_ang)
        nx, ny = math.cos(rad), math.sin(rad)
        tx, ty = -ny, nx
        for off_t in (-1.2, 1.2):
            wx = tw_x + nx * 2.92 + tx * off_t
            wy = tw_y + ny * 2.92 + ty * off_t
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (wx, wy),
                red_z + 29.2,
                red_z + 32.0,
                0.95,
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
    # Octagonal stepped fountain basin & ornate Victorian/Ottoman baroque stone fountain centerpiece
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
        2.6,
        2.4,
        segments=8,
        smooth=False,
        cap_end=True,
    )
    collector.add_branch_tube(
        "StreetProps",
        "mat_stone_mizzi",
        (mur_x, mur_y, mur_z + 1.35),
        (mur_x, mur_y, mur_z + 4.2),
        0.75,
        0.55,
        segments=8,
        smooth=True,
        cap_end=True,
    )
    collector.add_dome("StreetProps", "mat_stone_mizzi", (mur_x, mur_y, mur_z + 4.2), radius=1.1, height=1.2, rings=5, segments=8)

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
        gate_gaps=[(new_gate_xy[0], new_gate_xy[1], 5.5)],
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

    # 3. Hero Landmarks: Church of the Holy Sepulchre, Redeemer, Muristan Fountain
    _build_church_of_the_holy_sepulchre(collector, sign_atlas)
    _build_redeemer_and_muristan(collector, sign_atlas)

    # 4. Christian Quarter OSM Buildings (filtered to inside Old City Walls & non-overlapping with hero sites)
    clear_scene_building_rings()
    osm_buildings = [
        b for b in osm_data.get("buildings", [])
        if b.get("region") == "r02_christian_quarter"
    ]

    valid_blds = []
    for b in osm_buildings:
        osm_id = int(b.get("osm_id", 0))
        if osm_id in (5862586, 7880713, 126086870, 158063844, 158063849, 1022672085, 1022663529, 29268902, 690881495, 690894839, 290821793):
            continue
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        if not is_inside_old_city_walls(cx, cy, margin=-4.0):
            continue
        # Keep Holy Sepulchre + Southern Parvis (`164..220, 152..218`) clear
        if 164.0 <= cx <= 220.0 and 152.0 <= cy <= 218.0:
            continue
        # Keep Redeemer + Muristan Fountain Plaza (`218..294, 114..158`) clear
        if 218.0 <= cx <= 294.0 and 114.0 <= cy <= 158.0:
            continue
        # Keep New Gate portal road (`-152..-125, 282..312`) clear
        if -152.0 <= cx <= -125.0 and 282.0 <= cy <= 312.0:
            continue
        ring = b.get("ring", [])
        if len(ring) < 3:
            continue
        rxs = [float(p[0]) for p in ring]
        rys = [float(p[1]) for p in ring]
        if (max(rxs) - min(rxs)) > 68.0 or (max(rys) - min(rys)) > 68.0:
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
        # Leave cross-street intersection open where David Street / Street of the Chain crosses (`y in [-2..18]`)
        if -2.0 <= ym <= 18.0:
            continue
        bz = street_elevation_z(313.0, ym)

        # Build the 3 parallel market lane walls & arched shopfronts
        for lane_idx, lx in enumerate((302.0, 313.0, 324.0)):
            w_left = lx - 2.8
            w_right = lx + 2.8
            # Side walls (2 storeys, 7.8m tall)
            collector.add_box("Buildings_Walls", "mat_stone_ashlar_aged", (w_left - 1.2, ym, bz + 3.9), (2.4, 11.6, 7.8), uv_scale=0.25)
            collector.add_box("Buildings_Walls", "mat_stone_ashlar_aged", (w_right + 1.2, ym, bz + 3.9), (2.4, 11.6, 7.8), uv_scale=0.25)

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
                        bz + 3.1,
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

            # High Crusader vaulted stone roof bay above the souq lane (leaving every 3rd bay open as a skylight shaft!)
            if (j + lane_idx) % 3 != 1:
                collector.add_box(
                    "Buildings_Walls",
                    "mat_stone_ashlar",
                    (lx, ym, bz + 8.1),
                    (6.2, 11.2, 0.65),
                    uv_scale=0.26,
                    bottom_face=True,
                )
            # Transverse stone vault arch ribs + hanging glowing bazaar lantern
            collector.add_box(
                "Buildings_Details",
                "mat_stone_boss",
                (lx, ym, bz + 7.4),
                (5.6, 0.65, 0.85),
                uv_scale=0.28,
                bottom_face=True,
            )
            collector.add_box(
                "StreetProps",
                "mat_lantern_glow",
                (lx, ym, bz + 6.5),
                (0.28, 0.28, 0.42),
                bottom_face=True,
            )

            # Spice sacks & produce crates along the souq edges
            if j % 2 == 0:
                build_souq_display(collector, (w_left + 0.55, ym, bz), (0.0, 1.0), (1.0, 0.0), seed=500 + j * 10 + lane_idx)

    # 2. Street of the Chain (`Tariq Bab al-Silsila`) Arched Portals & Mamluk Ablaq Turbas
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
                (cx, cy + side_y, cz + 3.1),
                (1.35, 1.15, 6.2),
                uv_scale=0.26,
            )
        # Stone arch beam & cornice spanning between the piers
        collector.add_box(
            "Buildings_Details",
            "mat_stone_boss",
            (cx, cy, cz + 5.8),
            (1.15, 7.5, 0.95),
            uv_scale=0.28,
            bottom_face=True,
        )
        collector.add_box(
            "Buildings_Details",
            "mat_stone_mizzi",
            (cx, cy, cz + 6.4),
            (1.28, 7.6, 0.24),
            uv_scale=0.32,
            bottom_face=True,
        )
        build_street_lamp(collector, cx + 2.2, cy + 2.5, cz, height=4.2)

    # 3. Mamluk *Turbas* & *Madrasas* on Street of the Chain (`Turbat Baraka Khan` at `(445, 46)`, `Madrasa al-Tashtamuriyya` at `(525, 64)`)
    for t_idx, (tx, ty) in enumerate(((445.0, 46.0), (525.0, 64.0))):
        tz = street_elevation_z(tx, ty)
        # Main masonry block with alternating Mamluk Ablaq red/cream limestone bands (`mat_stone_mizzi` + `mat_terracotta`)
        collector.add_box("Buildings_Walls", "mat_stone_ashlar", (tx, ty, tz + 5.2), (16.0, 11.0, 10.4), uv_scale=0.24)
        for band_z in (1.6, 2.8, 4.0, 5.2):
            collector.add_box(
                "Buildings_Details",
                "mat_terracotta" if int(band_z * 2) % 2 == 0 else "mat_stone_mizzi",
                (tx, ty - 5.56, tz + band_z),
                (16.1, 0.16, 0.34),
                uv_scale=0.35,
            )
        # Mamluk street facade arched grille windows & portal
        for wx_off in (-4.8, 0.0, 4.8):
            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                "mat_wood_doors",
                (tx + wx_off, ty - 5.53),
                tz + (0.0 if wx_off == 0.0 else 0.9),
                tz + 3.6,
                1.1,
                2.2,
                0.02,
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
                (tx + wx_off, ty - 5.53, tz + 2.5),
                (1.0, 0.0),
                (0.0, -1.0),
                span_w=2.4,
                arch_h=1.15,
                ring_thickness=0.34,
                offset_n0=-0.02,
                offset_n1=0.22,
                num_voussoirs=7,
            )
        # Twin Mamluk mausoleum stone domes on drum
        for off_x in (-3.8, 3.8):
            collector.add_branch_tube(
                "Buildings_Walls",
                "mat_stone_mizzi",
                (tx + off_x, ty, tz + 10.4),
                (tx + off_x, ty, tz + 12.6),
                2.8,
                2.7,
                segments=12,
                cap_end=True,
            )
            collector.add_dome(
                "Buildings_Details",
                "mat_stone_ashlar",
                (tx + off_x, ty, tz + 12.5),
                radius=2.7,
                height=2.8,
                rings=6,
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

    build_region_ground_grid(collector, 165.0, 605.0, -45.0, 260.0, step=5.5, exclude_fn=r03_exclude)
    _build_vaulted_triple_souq_and_chain_street(collector, sign_atlas)

    clear_scene_building_rings()
    osm_buildings = [
        b for b in osm_data.get("buildings", [])
        if b.get("region") == "r03_central_souqs"
    ]

    valid_blds = []
    for b in osm_buildings:
        osm_id = int(b.get("osm_id", 0))
        if osm_id in (5862586, 7880713):
            continue
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        # Keep Triple Souq corridor (`x = 292..334, y = -42..162`) clear
        if 292.0 <= cx <= 334.0 and -42.0 <= cy <= 162.0:
            continue
        # Keep Street of the Chain & Mamluk Turba corridor clear
        chain_y = 4.0 + max(0.0, (cx - 175.0) / 420.0) * 62.0
        if abs(cy - chain_y) < 9.5:
            continue
        # Keep Khan el-Zeit corridor (`x = 296..314, y = 160..260`) clear
        if 296.0 <= cx <= 314.0 and 160.0 <= cy <= 260.0:
            continue
        ring = b.get("ring", [])
        if len(ring) < 3:
            continue
        rxs = [float(p[0]) for p in ring]
        rys = [float(p[1]) for p in ring]
        if (max(rxs) - min(rxs)) > 65.0 or (max(rys) - min(rys)) > 65.0:
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


def build_additional_region(
    region_id: str,
    collector: MeshCollector,
    osm_data: dict,
    sign_atlas: Dict[str, dict],
) -> int:
    if region_id == "r03_central_souqs":
        return build_region_r03_central_souqs(collector, osm_data, sign_atlas)
    raise ValueError(f"Unknown region_id: {region_id}")

