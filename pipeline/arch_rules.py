#!/usr/bin/env python3
"""
Clean-Room 3D Architectural Grammar, Multi-Stone Recessed Masonry Facades,
Graffitied Steel Shutters, Tiled Street Name Plates & Landmark Builders
for the Jaffa Gate / Old City Jerusalem 3D Pipeline (`pipeline/arch_rules.py`).

Pass 2 Upgrades:
  - Multi-texture Jerusalem limestone variety (`mat_stone_ashlar`, `mat_stone_mizzi`,
    `mat_stone_ashlar_aged`, `mat_stone_boss`, `mat_stone_rubble`) across buildings
  - Authentic Jerusalem ground-floor bay variety:
    * Graffitied & weathered corrugated steel rolling shutters (`mat_steel_shutters`)
      with 3D galvanized roller-hood housings (`mat_corrugated_metal`)
    * Tagged steel double service doors (`steel_door_tagged`)
    * Dark mullioned café / shopfront glazing (`window_glass` with 3D timber muntins)
    * Open bazaar arches (`souq_open`) and panelled timber portals (`door_wood`)
  - Sealed Jaffa Gate vaulted L-passage ceiling (`bottom_face=True` + portal spandrels)
  - Widened 13.5m Sunken Citadel Dry Moat (`y = -29.0..-15.5, z = -6.0m`) with low 0.42m
    counterscarp curb + 0.78m open wrought-iron railing and exposed Date Palm trunk
  - Chamfered SW corner, full upper-floor wrought-iron balconies, pilasters, and rooftop
    balustrades on `New Imperial Hotel` (`7002`) and `Petra Hotel` (`7003`)
  - Western Mamilla horizon embankment & buildings (`x = -125m`) eliminating horizon gaps
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from props_lib import (
    MeshCollector,
    UVRect,
    Vec2,
    Vec3,
    build_blade_sign,
    build_cypress_tree,
    build_fascia_sign,
    build_iron_railing,
    build_olive_tree,
    build_palm_tree,
    build_rooftop_clutter,
    build_shop_awning,
    build_souq_display,
    build_street_name_plate,
    build_wall_lantern,
    build_wall_utility_box,
)


# Exact 2x2 quadrant UV coordinates matching `gen_wood_doors_atlas` in `gen_pbr_textures.py`
WOOD_DOOR_UV: Dict[str, UVRect] = {
    "window_glass": (0.02, 0.52, 0.48, 0.98),    # Top-left quadrant
    "souq_open": (0.52, 0.52, 0.98, 0.98),       # Top-right quadrant
    "door_wood": (0.02, 0.02, 0.48, 0.48),       # Bottom-left quadrant
    "shutter_louvre": (0.52, 0.02, 0.98, 0.48),  # Bottom-right quadrant
}

# Exact 2x2 quadrant UV coordinates matching `steel_shutters_{diff,norm,rough}.jpg`
STEEL_SHUTTER_UV: Dict[str, UVRect] = {
    "shutter_graffiti_1": (0.02, 0.02, 0.48, 0.48),
    "shutter_graffiti_2": (0.52, 0.02, 0.98, 0.48),
    "shutter_rusty_green": (0.02, 0.52, 0.48, 0.98),
    "steel_door_tagged": (0.52, 0.52, 0.98, 0.98),
}

UPPER_STONE_MATS: List[str] = [
    "mat_stone_ashlar",       # Warm golden-cream Meleke limestone
    "mat_stone_mizzi",        # Pale cool ivory-grey Mizzi Hilu limestone
    "mat_stone_ashlar_aged",  # Weathered ochre/patinated historic courses
]

GROUND_PIER_MATS: List[str] = [
    "mat_stone_boss",
    "mat_stone_ashlar_aged",
    "mat_stone_rubble",
]

SHOP_SIGN_KEYS: List[str] = [
    "old_city_bazaar",
    "jerusalem_pottery",
    "money_exchange",
    "falafel_cafe",
    "st_james_souvenirs",
    "holy_land_rugs",
    "pilgrim_books",
    "swedish_study",
    "christian_info",
    "new_imperial_hotel",
    "petra_hotel",
]

AWNING_MATS: List[str] = [
    "mat_awning_red",
    "mat_awning_striped",
    "mat_awning_green",
    "mat_corrugated_metal",
]

# Active walkable player/camera corridors for 3-Tier Architectural LOD calculation
WALKABLE_CORRIDORS: List[Tuple[Vec2, Vec2]] = [
    ((-55.0, 4.0), (15.0, 2.0)),    # West Plaza & Jaffa Gate
    ((4.0, -4.0), (52.0, -45.0)),   # 1898 Breach & Citadel Moat Road
    ((12.0, -16.0), (78.0, 12.0)),  # Omar Ibn El-Khattab Square
    ((26.0, 8.0), (28.0, 52.0)),    # Latin Patriarchate Road
    ((75.0, 2.0), (135.0, 2.0)),    # David Street Souq
]


def _point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-8:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def corridor_distance(cx: float, cy: float) -> float:
    """Returns the minimum 2D Euclidean distance from (cx, cy) to any active walkable corridor."""
    return min(
        _point_to_segment_dist(cx, cy, a[0], a[1], b[0], b[1])
        for a, b in WALKABLE_CORRIDORS
    )


def get_sign_uv(sign_atlas: Dict[str, dict], key_or_idx: str | int) -> UVRect:
    """Returns (u0, v0, u1, v1) from sign_atlas.json for a named key or integer index."""
    if isinstance(key_or_idx, int):
        key = SHOP_SIGN_KEYS[key_or_idx % len(SHOP_SIGN_KEYS)]
    else:
        key = key_or_idx
    entry = sign_atlas.get(key)
    if entry and "u0" in entry:
        return (float(entry["u0"]), float(entry["v0"]), float(entry["u1"]), float(entry["v1"]))
    idx = abs(hash(str(key_or_idx))) % 16
    col = idx % 4
    row = idx // 4
    u0 = col * 0.25 + 0.01
    u1 = (col + 1) * 0.25 - 0.01
    v0 = (3 - row) * 0.25 + 0.01
    v1 = (4 - row) * 0.25 - 0.01
    return (u0, v0, u1, v1)


def ensure_ccw_2d(ring: Sequence[Sequence[float]]) -> List[Vec2]:
    """Removes duplicate closing vertex and ensures counter-clockwise 2D winding."""
    pts: List[Vec2] = [(float(p[0]), float(p[1])) for p in ring]
    if len(pts) >= 2 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-4:
        pts.pop()
    if len(pts) < 3:
        return pts
    area2 = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        area2 += x0 * y1 - x1 * y0
    if area2 < 0.0:
        pts.reverse()
    return pts


# ==============================================================================
# 1. 3D ARCH VOUSSOIRS, RECESSED WINDOWS & MULTI-MODE GROUND SHOPFRONT BAYS
# ==============================================================================

def add_arch_voussoirs(
    collector: MeshCollector,
    group: str,
    mat_name: str,
    center_xyz: Vec3,
    tangent_xy: Vec2,
    normal_xy: Vec2,
    span_w: float,
    arch_h: float,
    ring_thickness: float = 0.24,
    offset_n0: float = -0.04,
    offset_n1: float = 0.08,
    num_voussoirs: int = 5,
    keystone_boost: float = 0.06,
    back_face: bool = False,
) -> None:
    """
    Constructs a 3D radial stone arch ring (voussoirs) with a proud projecting center
    keystone above an arched doorway, shopfront, or window.
    Optimized to 5 segments and omits hidden rear faces (`back_face=False`).
    """
    cx, cy, cz = center_xyz
    tx, ty = tangent_xy
    nx, ny = normal_xy
    rx_in = span_w * 0.5
    rz_in = arch_h

    for i in range(num_voussoirs):
        a0 = math.pi * (1.0 - i / num_voussoirs)
        a1 = math.pi * (1.0 - (i + 1) / num_voussoirs)
        is_keystone = (i == num_voussoirs // 2)
        rt = ring_thickness + (keystone_boost if is_keystone else 0.0)
        n1 = offset_n1 + (keystone_boost * 0.65 if is_keystone else 0.0)

        point0 = math.sin(a0) ** 2 * 0.14 * arch_h
        point1 = math.sin(a1) ** 2 * 0.14 * arch_h

        u_in0 = math.cos(a0) * rx_in
        z_in0 = math.sin(a0) * rz_in + point0
        u_in1 = math.cos(a1) * rx_in
        z_in1 = math.sin(a1) * rz_in + point1

        u_out0 = math.cos(a0) * (rx_in + rt)
        z_out0 = math.sin(a0) * (rz_in + rt) + point0
        u_out1 = math.cos(a1) * (rx_in + rt)
        z_out1 = math.sin(a1) * (rz_in + rt) + point1

        def wp(u: float, z_off: float, d_n: float) -> Vec3:
            return (cx + tx * u + nx * d_n, cy + ty * u + ny * d_n, cz + z_off)

        f0 = wp(u_in0, z_in0, n1)
        f1 = wp(u_in1, z_in1, n1)
        f2 = wp(u_out1, z_out1, n1)
        f3 = wp(u_out0, z_out0, n1)
        collector.add_quad(group, mat_name, f0, f1, f2, f3, uv_scale=0.35, expected_normal=(nx, ny, 0.0))

        b0 = wp(u_in0, z_in0, offset_n0)
        b1 = wp(u_in1, z_in1, offset_n0)
        collector.add_quad(group, mat_name, b0, b1, f1, f0, uv_scale=0.35, expected_normal=(0.0, 0.0, -1.0))

        if back_face:
            bo0 = wp(u_out0, z_out0, offset_n0)
            bo1 = wp(u_out1, z_out1, offset_n0)
            collector.add_quad(group, mat_name, f3, f2, bo1, bo0, uv_scale=0.35, expected_normal=(0.0, 0.0, 1.0))


def build_upper_window_bay(
    collector: MeshCollector,
    p0_xy: Vec2,
    p1_xy: Vec2,
    z0: float,
    z1: float,
    normal_xy: Vec2,
    wall_mat: str = "mat_stone_ashlar",
    trim_mat: str = "mat_stone_mizzi",
    recess: float = 0.28,
    has_shutters: bool = True,
    has_corbels: bool = True,
    has_grille: bool = False,
    has_balcony: bool = False,
) -> None:
    """
    Builds a single LOD 0 upper-storey masonry facade bay (`p0_xy -> p1_xy`, `z0 .. z1`) with:
    - Outer masonry wall quads + 0.28m deeply recessed window reveal (stone jambs & soffit)
    - Projecting dressed-stone window surround (`trim_mat`, 0.14m wide, 0.07m proud) + 5-seg arched voussoir hood
    - Projecting stone sill (0.15m deep, 0.09m thick) + optional twin stone corbels
    - Recessed `window_glass` pane + 2 thin single front mullion quads (2 faces instead of 18!)
    - 3D open angled louvred wooden shutters (`shutter_louvre`) or wrought-iron window grille
    """
    dx = p1_xy[0] - p0_xy[0]
    dy = p1_xy[1] - p0_xy[1]
    bay_w = math.hypot(dx, dy)
    bay_h = z1 - z0
    nx, ny = normal_xy
    if bay_w < 1.3 or bay_h < 2.0:
        collector.add_quad(
            "Buildings_Walls",
            wall_mat,
            (p0_xy[0], p0_xy[1], z0),
            (p1_xy[0], p1_xy[1], z0),
            (p1_xy[0], p1_xy[1], z1),
            (p0_xy[0], p0_xy[1], z1),
            expected_normal=(nx, ny, 0.0),
        )
        return

    tx, ty = dx / bay_w, dy / bay_w

    win_w = min(1.18, bay_w * 0.44)
    win_h = min(1.72, bay_h * 0.54)
    sill_z = z0 + bay_h * 0.24
    head_z = sill_z + win_h

    u_left = (bay_w - win_w) * 0.5
    u_right = u_left + win_w

    def pt_at(u: float) -> Vec2:
        return (p0_xy[0] + tx * u, p0_xy[1] + ty * u)

    w_p0 = pt_at(u_left)
    w_p1 = pt_at(u_right)

    # 1. Surrounding outer masonry wall planes (4 single-sided exterior quads, zero interior duplicates)
    collector.add_quad(
        "Buildings_Walls", wall_mat,
        (p0_xy[0], p0_xy[1], z0), (w_p0[0], w_p0[1], z0), (w_p0[0], w_p0[1], z1), (p0_xy[0], p0_xy[1], z1),
        expected_normal=(nx, ny, 0.0),
    )
    collector.add_quad(
        "Buildings_Walls", wall_mat,
        (w_p1[0], w_p1[1], z0), (p1_xy[0], p1_xy[1], z0), (p1_xy[0], p1_xy[1], z1), (w_p1[0], w_p1[1], z1),
        expected_normal=(nx, ny, 0.0),
    )
    collector.add_quad(
        "Buildings_Walls", wall_mat,
        (w_p0[0], w_p0[1], z0), (w_p1[0], w_p1[1], z0), (w_p1[0], w_p1[1], sill_z), (w_p0[0], w_p0[1], sill_z),
        expected_normal=(nx, ny, 0.0),
    )
    collector.add_quad(
        "Buildings_Walls", wall_mat,
        (w_p0[0], w_p0[1], head_z), (w_p1[0], w_p1[1], head_z), (w_p1[0], w_p1[1], z1), (w_p0[0], w_p0[1], z1),
        expected_normal=(nx, ny, 0.0),
    )

    # Explicit 0.28m recessed stone jamb & lintel soffit reveals
    jamb_l0 = (w_p0[0] - nx * recess, w_p0[1] - ny * recess, sill_z)
    jamb_l1 = (w_p0[0], w_p0[1], sill_z)
    jamb_l2 = (w_p0[0], w_p0[1], head_z)
    jamb_l3 = (w_p0[0] - nx * recess, w_p0[1] - ny * recess, head_z)
    collector.add_quad("Buildings_Walls", wall_mat, jamb_l0, jamb_l1, jamb_l2, jamb_l3, expected_normal=(tx, ty, 0.0))

    jamb_r0 = (w_p1[0], w_p1[1], sill_z)
    jamb_r1 = (w_p1[0] - nx * recess, w_p1[1] - ny * recess, sill_z)
    jamb_r2 = (w_p1[0] - nx * recess, w_p1[1] - ny * recess, head_z)
    jamb_r3 = (w_p1[0], w_p1[1], head_z)
    collector.add_quad("Buildings_Walls", wall_mat, jamb_r0, jamb_r1, jamb_r2, jamb_r3, expected_normal=(-tx, -ty, 0.0))

    collector.add_quad("Buildings_Walls", wall_mat, jamb_l2, jamb_r3, jamb_r2, jamb_l3, expected_normal=(0.0, 0.0, -1.0))

    # 2. Projecting dressed-stone window surround frame (`cap_back=False`)
    frame_w = 0.14
    frame_proud = 0.07
    collector.add_oriented_box(
        "Buildings_Details",
        trim_mat,
        pt_at(u_left - frame_w),
        w_p0,
        sill_z,
        head_z + 0.06,
        0.0,
        frame_proud,
        normal_xy,
        cap_ends=False,
    )
    collector.add_oriented_box(
        "Buildings_Details",
        trim_mat,
        w_p1,
        pt_at(u_right + frame_w),
        sill_z,
        head_z + 0.06,
        0.0,
        frame_proud,
        normal_xy,
        cap_ends=False,
    )

    # 3. 3D Arched stone voussoir hood mould + keystone (`num_voussoirs=5`)
    win_cx = 0.5 * (w_p0[0] + w_p1[0])
    win_cy = 0.5 * (w_p0[1] + w_p1[1])
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        trim_mat,
        (win_cx, win_cy, head_z - 0.02),
        (tx, ty),
        normal_xy,
        span_w=win_w,
        arch_h=0.28,
        ring_thickness=0.18,
        offset_n0=-recess * 0.5,
        offset_n1=frame_proud + 0.02,
        num_voussoirs=5,
        keystone_boost=0.05,
    )

    # 4. Projecting stone window sill + optional twin stone corbel brackets
    sill_p0 = pt_at(u_left - frame_w - 0.04)
    sill_p1 = pt_at(u_right + frame_w + 0.04)
    collector.add_oriented_box(
        "Buildings_Details",
        trim_mat,
        sill_p0,
        sill_p1,
        sill_z - 0.09,
        sill_z,
        -recess,
        0.15,
        normal_xy,
        cap_bottom=True,
    )
    if has_corbels:
        for corbel_u in (u_left - 0.02, u_right - 0.08):
            collector.add_oriented_box(
                "Buildings_Details",
                trim_mat,
                pt_at(corbel_u),
                pt_at(corbel_u + 0.10),
                sill_z - 0.24,
                sill_z - 0.09,
                0.0,
                0.11,
                normal_xy,
                cap_top=False,
                cap_bottom=True,
            )

    # 5. Recessed Window Glass Pane (`window_glass` quadrant) + 2 Thin Single Front Mullion Quads (2 faces)
    glass_d = -recess + 0.04
    g0 = (w_p0[0] + nx * glass_d, w_p0[1] + ny * glass_d, sill_z)
    g1 = (w_p1[0] + nx * glass_d, w_p1[1] + ny * glass_d, sill_z)
    g2 = (w_p1[0] + nx * glass_d, w_p1[1] + ny * glass_d, head_z)
    g3 = (w_p0[0] + nx * glass_d, w_p0[1] + ny * glass_d, head_z)
    collector.add_quad("Buildings_Openings", "mat_wood_doors", g0, g1, g2, g3, uv_rect=WOOD_DOOR_UV["window_glass"], expected_normal=(nx, ny, 0.0))

    m_d = glass_d + 0.025
    mid_u = 0.5 * (u_left + u_right)
    mv0 = pt_at(mid_u - 0.022)
    mv1 = pt_at(mid_u + 0.022)
    collector.add_quad(
        "Buildings_Details",
        "mat_iron_dark",
        (mv0[0] + nx * m_d, mv0[1] + ny * m_d, sill_z),
        (mv1[0] + nx * m_d, mv1[1] + ny * m_d, sill_z),
        (mv1[0] + nx * m_d, mv1[1] + ny * m_d, head_z),
        (mv0[0] + nx * m_d, mv0[1] + ny * m_d, head_z),
        expected_normal=(nx, ny, 0.0),
    )
    bar_z = sill_z + win_h * 0.54
    collector.add_quad(
        "Buildings_Details",
        "mat_iron_dark",
        (w_p0[0] + nx * m_d, w_p0[1] + ny * m_d, bar_z - 0.022),
        (w_p1[0] + nx * m_d, w_p1[1] + ny * m_d, bar_z - 0.022),
        (w_p1[0] + nx * m_d, w_p1[1] + ny * m_d, bar_z + 0.022),
        (w_p0[0] + nx * m_d, w_p0[1] + ny * m_d, bar_z + 0.022),
        expected_normal=(nx, ny, 0.0),
    )

    # 6. 3D Open Angled Louvred Wooden Shutters (`shutter_louvre` quadrant) or Iron Grille
    if has_shutters:
        shutter_leaf_w = win_w * 0.48
        open_ang = 0.52
        for side_sign, hinge_xy in ((-1.0, w_p0), (1.0, w_p1)):
            hx = hinge_xy[0] + tx * (side_sign * 0.04) + nx * (frame_proud + 0.01)
            hy = hinge_xy[1] + ty * (side_sign * 0.04) + ny * (frame_proud + 0.01)
            ca, sa = math.cos(open_ang), math.sin(open_ang)
            ex = hx + (tx * side_sign * ca + nx * sa) * shutter_leaf_w
            ey = hy + (ty * side_sign * ca + ny * sa) * shutter_leaf_w
            if side_sign < 0:
                s0, s1 = (ex, ey, sill_z + 0.02), (hx, hy, sill_z + 0.02)
                s2, s3 = (hx, hy, head_z - 0.02), (ex, ey, head_z - 0.02)
            else:
                s0, s1 = (hx, hy, sill_z + 0.02), (ex, ey, sill_z + 0.02)
                s2, s3 = (ex, ey, head_z - 0.02), (hx, hy, head_z - 0.02)
            collector.add_quad(
                "Buildings_Openings",
                "mat_wood_doors",
                s0,
                s1,
                s2,
                s3,
                uv_rect=WOOD_DOOR_UV["shutter_louvre"],
                double_sided=True,
            )

    if has_grille:
        gr_d = -0.04
        gr0 = (w_p0[0] + nx * gr_d, w_p0[1] + ny * gr_d, sill_z)
        gr1 = (w_p1[0] + nx * gr_d, w_p1[1] + ny * gr_d, sill_z)
        gr2 = (w_p1[0] + nx * gr_d, w_p1[1] + ny * gr_d, head_z)
        gr3 = (w_p0[0] + nx * gr_d, w_p0[1] + ny * gr_d, head_z)
        collector.add_quad("Buildings_Details", "mat_iron_grille", gr0, gr1, gr2, gr3, uv_rect=(0.0, 0.0, 1.0, 1.5), double_sided=True)

    if has_balcony:
        bal_p0 = pt_at(u_left - 0.28)
        bal_p1 = pt_at(u_right + 0.28)
        collector.add_oriented_box("Buildings_Details", trim_mat, bal_p0, bal_p1, sill_z - 0.14, sill_z, 0.0, 0.72, normal_xy, cap_bottom=True)
        b_c0 = (bal_p0[0] + nx * 0.68, bal_p0[1] + ny * 0.68, sill_z)
        b_c1 = (bal_p1[0] + nx * 0.68, bal_p1[1] + ny * 0.68, sill_z)
        build_iron_railing(collector, b_c0, b_c1, height=0.88, picket_spacing=0.30)


def build_ground_shop_bay(
    collector: MeshCollector,
    p0_xy: Vec2,
    p1_xy: Vec2,
    z0: float,
    z1: float,
    normal_xy: Vec2,
    pier_mat: str = "mat_stone_boss",
    opening_mode: str = "souq_open",
    shutter_variant: str = "shutter_graffiti_1",
    has_awning: bool = True,
    awning_mat: str = "mat_awning_red",
    sign_uv: Optional[UVRect] = None,
    has_blade_sign: bool = False,
    has_display: bool = False,
    seed: int = 0,
) -> None:
    """
    Builds a LOD 0 ground-floor rusticated stone bay (`pier_mat`, `z0 .. z1`) with 4 realistic modes:
      - `opening_mode == "steel_shutter"`: Graffitied or weathered green corrugated steel
        rolling shutter (`mat_steel_shutters`) + 3D galvanized steel roller-hood housing box
      - `opening_mode == "steel_door"`: Tagged steel double service door (`steel_door_tagged`)
      - `opening_mode == "cafe_glass"`: Dark glazed hotel/café shopfront (`window_glass`) with
        stone stall-riser plinth + divided-light mullions
      - `opening_mode == "souq_open"` / `"wood_door"`: Active open souq arch or panelled wood door
    """
    dx = p1_xy[0] - p0_xy[0]
    dy = p1_xy[1] - p0_xy[1]
    bay_w = math.hypot(dx, dy)
    bay_h = z1 - z0
    recess = 0.35
    nx, ny = normal_xy

    if bay_w < 1.6 or bay_h < 2.6:
        collector.add_quad(
            "Buildings_Walls", pier_mat,
            (p0_xy[0], p0_xy[1], z0), (p1_xy[0], p1_xy[1], z0),
            (p1_xy[0], p1_xy[1], z1), (p0_xy[0], p0_xy[1], z1),
            expected_normal=(nx, ny, 0.0),
        )
        return

    tx, ty = dx / bay_w, dy / bay_w
    rx, ry = -ny, nx

    is_wide_opening = opening_mode in ("souq_open", "steel_shutter", "cafe_glass")
    open_w = min(2.35 if is_wide_opening else 1.50, bay_w * (0.66 if is_wide_opening else 0.48))
    open_h = min(2.95, bay_h * 0.74)
    u_left = (bay_w - open_w) * 0.5
    u_right = u_left + open_w

    def pt_at(u: float) -> Vec2:
        return (p0_xy[0] + tx * u, p0_xy[1] + ty * u)

    s_p0 = pt_at(u_left)
    s_p1 = pt_at(u_right)
    head_z = z0 + open_h

    # 1. Left & right outer stone piers + upper spandrel wall above arch
    collector.add_quad(
        "Buildings_Walls", pier_mat,
        (p0_xy[0], p0_xy[1], z0), (s_p0[0], s_p0[1], z0), (s_p0[0], s_p0[1], z1), (p0_xy[0], p0_xy[1], z1),
        expected_normal=(nx, ny, 0.0),
    )
    collector.add_quad(
        "Buildings_Walls", pier_mat,
        (s_p1[0], s_p1[1], z0), (p1_xy[0], p1_xy[1], z0), (p1_xy[0], p1_xy[1], z1), (s_p1[0], s_p1[1], z1),
        expected_normal=(nx, ny, 0.0),
    )
    collector.add_quad(
        "Buildings_Walls", pier_mat,
        (s_p0[0], s_p0[1], head_z), (s_p1[0], s_p1[1], head_z), (s_p1[0], s_p1[1], z1), (s_p0[0], s_p0[1], z1),
        expected_normal=(nx, ny, 0.0),
    )

    # Explicit 0.35m recessed stone jamb & lintel soffit reveals
    jl0 = (s_p0[0] - nx * recess, s_p0[1] - ny * recess, z0)
    jl1 = (s_p0[0], s_p0[1], z0)
    jl2 = (s_p0[0], s_p0[1], head_z)
    jl3 = (s_p0[0] - nx * recess, s_p0[1] - ny * recess, head_z)
    collector.add_quad("Buildings_Walls", pier_mat, jl0, jl1, jl2, jl3, expected_normal=(tx, ty, 0.0))

    jr0 = (s_p1[0], s_p1[1], z0)
    jr1 = (s_p1[0] - nx * recess, s_p1[1] - ny * recess, z0)
    jr2 = (s_p1[0] - nx * recess, s_p1[1] - ny * recess, head_z)
    jr3 = (s_p1[0], s_p1[1], head_z)
    collector.add_quad("Buildings_Walls", pier_mat, jr0, jr1, jr2, jr3, expected_normal=(-tx, -ty, 0.0))
    collector.add_quad("Buildings_Walls", pier_mat, jl2, jr3, jr2, jl3, expected_normal=(0.0, 0.0, -1.0))

    # 2. Stone threshold step at ground level
    collector.add_oriented_box("Buildings_Details", "mat_stone_ashlar_aged", s_p0, s_p1, z0 - 0.35, z0 + 0.14, -recess, 0.16, normal_xy)

    # 3. 3D Radial Stone Voussoir Arch Surround + Projecting Keystone (`num_voussoirs=5`)
    mid_x = 0.5 * (s_p0[0] + s_p1[0])
    mid_y = 0.5 * (s_p0[1] + s_p1[1])
    arch_spring_z = head_z - min(0.68, open_w * 0.28)
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        "mat_stone_ashlar",
        (mid_x, mid_y, arch_spring_z),
        (tx, ty),
        normal_xy,
        span_w=open_w,
        arch_h=head_z - arch_spring_z,
        ring_thickness=0.26,
        offset_n0=-recess * 0.65,
        offset_n1=0.09,
        num_voussoirs=5,
        keystone_boost=0.07,
    )

    # 4. Recessed Bay Interior Panel by `opening_mode`
    panel_d = -recess + 0.035
    hw_open = open_w * 0.5
    v0 = (mid_x - rx * hw_open + nx * panel_d, mid_y - ry * hw_open + ny * panel_d, z0 + 0.14)
    v1 = (mid_x + rx * hw_open + nx * panel_d, mid_y + ry * hw_open + ny * panel_d, z0 + 0.14)
    v2 = (mid_x + rx * hw_open + nx * panel_d, mid_y + ry * hw_open + ny * panel_d, head_z)
    v3 = (mid_x - rx * hw_open + nx * panel_d, mid_y - ry * hw_open + ny * panel_d, head_z)

    if opening_mode in ("steel_shutter", "steel_door"):
        uv_q = STEEL_SHUTTER_UV.get(shutter_variant, STEEL_SHUTTER_UV["shutter_graffiti_1"])
        collector.add_quad("Buildings_Openings", "mat_steel_shutters", v0, v1, v2, v3, uv_rect=uv_q, expected_normal=(nx, ny, 0.0))
        collector.add_oriented_box(
            "Buildings_Details",
            "mat_corrugated_metal",
            s_p0,
            s_p1,
            arch_spring_z - 0.18,
            arch_spring_z + 0.08,
            -recess + 0.02,
            0.05,
            normal_xy,
            cap_bottom=True,
        )

    elif opening_mode == "cafe_glass":
        riser_z = z0 + 0.68
        collector.add_oriented_box("Buildings_Details", "mat_stone_mizzi", s_p0, s_p1, z0 + 0.14, riser_z, -recess, -recess + 0.12, normal_xy)
        vg0 = (mid_x - rx * hw_open + nx * panel_d, mid_y - ry * hw_open + ny * panel_d, riser_z)
        vg1 = (mid_x + rx * hw_open + nx * panel_d, mid_y + ry * hw_open + ny * panel_d, riser_z)
        collector.add_quad("Buildings_Openings", "mat_wood_doors", vg0, vg1, v2, v3, uv_rect=WOOD_DOOR_UV["window_glass"], expected_normal=(nx, ny, 0.0))
        m_d = panel_d + 0.025
        for m_frac in (0.33, 0.67):
            mu = u_left + open_w * m_frac
            mp0, mp1 = pt_at(mu - 0.022), pt_at(mu + 0.022)
            collector.add_quad(
                "Buildings_Details", "mat_iron_dark",
                (mp0[0] + nx * m_d, mp0[1] + ny * m_d, riser_z),
                (mp1[0] + nx * m_d, mp1[1] + ny * m_d, riser_z),
                (mp1[0] + nx * m_d, mp1[1] + ny * m_d, head_z),
                (mp0[0] + nx * m_d, mp0[1] + ny * m_d, head_z),
                expected_normal=(nx, ny, 0.0),
            )
        collector.add_quad(
            "Buildings_Details", "mat_iron_dark",
            (s_p0[0] + nx * m_d, s_p0[1] + ny * m_d, z0 + 1.84),
            (s_p1[0] + nx * m_d, s_p1[1] + ny * m_d, z0 + 1.84),
            (s_p1[0] + nx * m_d, s_p1[1] + ny * m_d, z0 + 1.89),
            (s_p0[0] + nx * m_d, s_p0[1] + ny * m_d, z0 + 1.89),
            expected_normal=(nx, ny, 0.0),
        )

    else:
        uv_q = WOOD_DOOR_UV["souq_open"] if opening_mode == "souq_open" else WOOD_DOOR_UV["door_wood"]
        collector.add_quad("Buildings_Openings", "mat_wood_doors", v0, v1, v2, v3, uv_rect=uv_q, expected_normal=(nx, ny, 0.0))

    # 5. Un-mirrored Framed Shop Fascia Sign &/or Perpendicular Hanging Blade Sign
    if sign_uv is not None:
        fascia_w = min(open_w * 0.90, 1.90)
        fascia_h = min(0.44, (z1 - head_z) * 0.56)
        if fascia_h >= 0.25:
            build_fascia_sign(
                collector,
                (mid_x, mid_y, head_z + fascia_h * 0.62 + 0.10),
                (tx, ty),
                normal_xy,
                width=fascia_w,
                height=fascia_h,
                sign_uv_rect=sign_uv,
            )
        if has_blade_sign:
            pier_x = s_p1[0] + tx * 0.22
            pier_y = s_p1[1] + ty * 0.22
            build_blade_sign(
                collector,
                (pier_x, pier_y, z0 + 3.05),
                (tx, ty),
                normal_xy,
                width=0.82,
                height=0.46,
                sign_uv_rect=sign_uv,
            )

    # 6. 3D Sloped/Curved Awning on Wrought-Iron Brackets
    if has_awning and opening_mode in ("souq_open", "cafe_glass"):
        build_shop_awning(
            collector,
            (mid_x, mid_y, head_z + 0.08),
            (tx, ty),
            normal_xy,
            width=open_w + 0.26,
            projection=1.08,
            drop=0.40,
            mat_name=awning_mat,
        )

    # 7. Active Bazaar Sidewalk Display (only outside open souq bays)
    if has_display and opening_mode == "souq_open":
        build_souq_display(
            collector,
            (mid_x, mid_y, z0),
            (tx, ty),
            normal_xy,
            width=open_w,
            seed=seed,
        )


def _build_lod1_wall(
    collector: MeshCollector,
    p0: Vec2,
    p1: Vec2,
    base_z: float,
    ground_h: float,
    upper_floor_h: float,
    levels: int,
    normal_xy: Vec2,
    ground_pier_mat: str,
    upper_wall_mat: str,
    trim_stone_mat: str,
    building_id: int,
    edge_idx: int,
) -> None:
    """
    Builds a medium-detail LOD 1 facade wall (`22m < corridor_dist <= 55m` or rear walls of LOD 0):
    - Ground-floor stone wall quad + shopfront/door quad + 3-segment stone arch strip
    - Upper-floor stone wall quad + simple window quads (`window_glass` / `shutter_louvre` in a 1-quad frame)
    - Projecting cornice strip + roof parapet
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    edge_len = math.hypot(dx, dy)
    tx, ty = dx / edge_len, dy / edge_len
    nx, ny = normal_xy
    rx, ry = -ny, nx
    roof_z = base_z + ground_h + (levels - 1) * upper_floor_h

    # Ground floor continuous stone wall quad (from base_z - 1.2 to base_z + ground_h)
    collector.add_quad(
        "Buildings_Walls", ground_pier_mat,
        (p0[0], p0[1], base_z - 1.2), (p1[0], p1[1], base_z - 1.2),
        (p1[0], p1[1], base_z + ground_h), (p0[0], p0[1], base_z + ground_h),
        expected_normal=(nx, ny, 0.0),
    )
    # Upper floors continuous stone wall quad (from base_z + ground_h to roof_z + 0.70)
    collector.add_quad(
        "Buildings_Walls", upper_wall_mat,
        (p0[0], p0[1], base_z + ground_h), (p1[0], p1[1], base_z + ground_h),
        (p1[0], p1[1], roof_z + 0.70), (p0[0], p0[1], roof_z + 0.70),
        expected_normal=(nx, ny, 0.0),
    )

    num_bays = max(1, int(round(edge_len / 4.2)))
    bay_w = edge_len / num_bays
    shutter_keys = ["shutter_graffiti_1", "shutter_graffiti_2", "shutter_rusty_green", "steel_door_tagged"]

    for b_idx in range(num_bays):
        if bay_w < 2.2:
            continue
        mid_u = (b_idx + 0.5) * bay_w
        mx = p0[0] + tx * mid_u
        my = p0[1] + ty * mid_u

        # Ground floor opening quad (1 quad + 3-segment arch front strip = 4 faces)
        open_hw = min(0.95, bay_w * 0.28)
        open_h = 2.65
        d_g = 0.03
        gv0 = (mx - rx * open_hw + nx * d_g, my - ry * open_hw + ny * d_g, base_z + 0.12)
        gv1 = (mx + rx * open_hw + nx * d_g, my + ry * open_hw + ny * d_g, base_z + 0.12)
        gv2 = (mx + rx * open_hw + nx * d_g, my + ry * open_hw + ny * d_g, base_z + open_h)
        gv3 = (mx - rx * open_hw + nx * d_g, my - ry * open_hw + ny * d_g, base_z + open_h)
        if (building_id + edge_idx + b_idx) % 2 == 0:
            sh_k = shutter_keys[(building_id + edge_idx + b_idx) % len(shutter_keys)]
            collector.add_quad("Buildings_Openings", "mat_steel_shutters", gv0, gv1, gv2, gv3, uv_rect=STEEL_SHUTTER_UV[sh_k], expected_normal=(nx, ny, 0.0))
        else:
            collector.add_quad("Buildings_Openings", "mat_wood_doors", gv0, gv1, gv2, gv3, uv_rect=WOOD_DOOR_UV["door_wood"], expected_normal=(nx, ny, 0.0))

        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            trim_stone_mat,
            (mx, my, base_z + open_h - 0.35),
            (tx, ty),
            normal_xy,
            span_w=open_hw * 2.0,
            arch_h=0.42,
            ring_thickness=0.22,
            offset_n0=0.0,
            offset_n1=0.06,
            num_voussoirs=3,
            keystone_boost=0.04,
        )

        # Upper floor window quads (1 stone frame quad + 1 window/shutter quad = 2 faces per window)
        win_hw = 0.54
        win_h = 1.55
        for lvl in range(1, levels):
            fl_z0 = base_z + ground_h + (lvl - 1) * upper_floor_h
            sill_z = fl_z0 + 0.75
            head_z = sill_z + win_h
            d_f = 0.03
            d_w = 0.045
            fw_hw = win_hw + 0.12
            f0 = (mx - rx * fw_hw + nx * d_f, my - ry * fw_hw + ny * d_f, sill_z - 0.08)
            f1 = (mx + rx * fw_hw + nx * d_f, my + ry * fw_hw + ny * d_f, sill_z - 0.08)
            f2 = (mx + rx * fw_hw + nx * d_f, my + ry * fw_hw + ny * d_f, head_z + 0.12)
            f3 = (mx - rx * fw_hw + nx * d_f, my - ry * fw_hw + ny * d_f, head_z + 0.12)
            collector.add_quad("Buildings_Details", trim_stone_mat, f0, f1, f2, f3, expected_normal=(nx, ny, 0.0))

            w0 = (mx - rx * win_hw + nx * d_w, my - ry * win_hw + ny * d_w, sill_z)
            w1 = (mx + rx * win_hw + nx * d_w, my + ry * win_hw + ny * d_w, sill_z)
            w2 = (mx + rx * win_hw + nx * d_w, my + ry * win_hw + ny * d_w, head_z)
            w3 = (mx - rx * win_hw + nx * d_w, my - ry * win_hw + ny * d_w, head_z)
            uv_w = WOOD_DOOR_UV["shutter_louvre"] if (b_idx + lvl) % 2 == 0 else WOOD_DOOR_UV["window_glass"]
            collector.add_quad("Buildings_Openings", "mat_wood_doors", w0, w1, w2, w3, uv_rect=uv_w, expected_normal=(nx, ny, 0.0))

    # Projecting roof cornice strip (2 faces)
    collector.add_oriented_box(
        "Buildings_Details",
        trim_stone_mat,
        p0,
        p1,
        roof_z,
        roof_z + 0.18,
        0.0,
        0.16,
        normal_xy,
        cap_ends=False,
    )


def _build_lod2_building(
    collector: MeshCollector,
    pts: List[Vec2],
    base_z: float,
    ground_h: float,
    upper_floor_h: float,
    levels: int,
    ground_pier_mat: str,
    upper_wall_mat: str,
    building_id: int,
    has_dome: bool = False,
) -> None:
    """
    Builds a low-poly LOD 2 background OSM stone massing prism (`55m < corridor_dist <= 95m`):
    - Split at `base_z + 2.2m` so vertex color `COLOR_0` rising-damp dirt gradient bakes cleanly
    - At most 1 simple window quad per floor on walls longer than 5.0m
    - Total ~20–36 faces per entire building instead of 3,500!
    """
    roof_z = base_z + ground_h + (levels - 1) * upper_floor_h
    n_edges = len(pts)

    for i in range(n_edges):
        p0 = pts[i]
        p1 = pts[(i + 1) % n_edges]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        edge_len = math.hypot(dx, dy)
        if edge_len < 0.8:
            continue
        tx, ty = dx / edge_len, dy / edge_len
        nx, ny = ty, -tx
        rx, ry = -ny, nx

        z_split = base_z + 2.2
        collector.add_quad(
            "Buildings_Walls", ground_pier_mat,
            (p0[0], p0[1], base_z - 1.0), (p1[0], p1[1], base_z - 1.0),
            (p1[0], p1[1], z_split), (p0[0], p0[1], z_split),
            expected_normal=(nx, ny, 0.0),
        )
        collector.add_quad(
            "Buildings_Walls", upper_wall_mat,
            (p0[0], p0[1], z_split), (p1[0], p1[1], z_split),
            (p1[0], p1[1], roof_z + 0.55), (p0[0], p0[1], roof_z + 0.55),
            expected_normal=(nx, ny, 0.0),
        )

        # At most 1 simple window quad per upper floor on outward walls >= 5.0m
        if edge_len >= 5.0:
            mx = 0.5 * (p0[0] + p1[0]) + nx * 0.04
            my = 0.5 * (p0[1] + p1[1]) + ny * 0.04
            win_hw = 0.60
            for lvl in range(1, levels):
                sill_z = base_z + ground_h + (lvl - 1) * upper_floor_h + 0.75
                head_z = sill_z + 1.50
                w0 = (mx - rx * win_hw, my - ry * win_hw, sill_z)
                w1 = (mx + rx * win_hw, my + ry * win_hw, sill_z)
                w2 = (mx + rx * win_hw, my + ry * win_hw, head_z)
                w3 = (mx - rx * win_hw, my - ry * win_hw, head_z)
                collector.add_quad("Buildings_Openings", "mat_wood_doors", w0, w1, w2, w3, uv_rect=WOOD_DOOR_UV["window_glass"], expected_normal=(nx, ny, 0.0))

    roof_pts = [(p[0], p[1], roof_z + 0.55) for p in pts]
    collector.add_polygon_3d("Buildings_Walls", "mat_plaster", roof_pts, uv_scale=0.25)

    if has_dome:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        collector.add_dome("Buildings_Details", "mat_plaster", (cx, cy, roof_z + 0.55), radius=2.1, height=1.6, rings=3, segments=8)


# ==============================================================================
# 2. FULL BUILDING FOOTPRINT BUILDER (3-Tier Architectural LOD)
# ==============================================================================

def build_building_from_footprint(
    collector: MeshCollector,
    ring_xy: Sequence[Sequence[float]],
    base_z: float,
    levels: int,
    sign_atlas: Dict[str, dict],
    building_id: int = 0,
    is_hero: bool = False,
    force_souq_ground: bool = False,
    has_dome: bool = False,
    forced_sign_key: Optional[str] = None,
    all_upper_balconies: bool = False,
    rooftop_balustrade: bool = False,
) -> None:
    """
    Constructs a multi-storey Jerusalem Old City building using a strict 3-Tier Architectural LOD:
      - LOD 0 (`is_hero` or `corridor_dist <= 22.0m`): Full street-level detail on street-facing
        walls (`wall_corridor_dist <= 26.0m`), and LOD 1 on rear/courtyard walls!
      - LOD 1 (`22.0m < corridor_dist <= 55.0m`): Recessed/framed window & shop quads + 3-segment
        arch strip + cornice & parapet.
      - LOD 2 (`55.0m < corridor_dist <= 95.0m`): Low-poly stone massing prism with baked vertex
        AO/dirt (`COLOR_0`) and at most 1 window quad per floor (~24–40 faces per building).
    """
    pts = ensure_ccw_2d(ring_xy)
    if len(pts) < 3:
        return

    levels = max(2, min(4, levels))
    ground_h = 4.0
    upper_floor_h = 3.3
    total_wall_h = ground_h + (levels - 1) * upper_floor_h
    roof_z = base_z + total_wall_h

    upper_wall_mat = UPPER_STONE_MATS[building_id % len(UPPER_STONE_MATS)]
    trim_stone_mat = UPPER_STONE_MATS[(building_id + 1) % len(UPPER_STONE_MATS)]
    ground_pier_mat = "mat_stone_boss" if is_hero else GROUND_PIER_MATS[building_id % len(GROUND_PIER_MATS)]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    corridor_dist = corridor_distance(cx, cy)
    if is_hero or corridor_dist <= 22.0:
        lod_tier = 0
    elif corridor_dist <= 55.0:
        lod_tier = 1
    else:
        lod_tier = 2

    if lod_tier == 2:
        _build_lod2_building(
            collector,
            pts,
            base_z,
            ground_h,
            upper_floor_h,
            levels,
            ground_pier_mat,
            upper_wall_mat,
            building_id,
            has_dome=has_dome,
        )
        return

    shutter_keys = ["shutter_graffiti_1", "shutter_graffiti_2", "shutter_rusty_green", "steel_door_tagged"]

    n_edges = len(pts)
    for i in range(n_edges):
        p0 = pts[i]
        p1 = pts[(i + 1) % n_edges]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        edge_len = math.hypot(dx, dy)
        if edge_len < 0.8:
            continue

        tx, ty = dx / edge_len, dy / edge_len
        nx, ny = ty, -tx

        wall_mx = 0.5 * (p0[0] + p1[0])
        wall_my = 0.5 * (p0[1] + p1[1])
        d_mid = corridor_distance(wall_mx, wall_my)
        d_out = corridor_distance(wall_mx + nx * 3.5, wall_my + ny * 3.5)
        d_in = corridor_distance(wall_mx - nx * 2.0, wall_my - ny * 2.0)
        wall_corridor_dist = min(d_mid, d_out)
        faces_away = (d_out > d_in + 1.4) and (wall_corridor_dist > 14.0)

        # Even on a LOD 0 building, rear/courtyard walls that face away from the street use LOD 1!
        if lod_tier == 1 or wall_corridor_dist > 26.0 or faces_away:
            _build_lod1_wall(
                collector,
                p0,
                p1,
                base_z,
                ground_h,
                upper_floor_h,
                levels,
                (nx, ny),
                ground_pier_mat,
                upper_wall_mat,
                trim_stone_mat,
                building_id,
                i,
            )
            continue

        # --- LOD 0 Street-Facing Facade Wall ---
        # 0. Sub-grade foundation plinth skirt (`base_z - 1.5m .. base_z`)
        collector.add_oriented_box(
            "Buildings_Walls",
            ground_pier_mat,
            p0,
            p1,
            base_z - 1.5,
            base_z,
            0.0,
            0.03,
            (nx, ny),
            cap_top=True,
            cap_ends=False,
        )

        target_bay_w = 3.4
        num_bays = max(1, int(round(edge_len / target_bay_w)))
        bay_w = edge_len / num_bays

        # 1. Ground Floor (`base_z .. base_z + ground_h`)
        for b_idx in range(num_bays):
            b_p0 = (p0[0] + tx * (b_idx * bay_w), p0[1] + ty * (b_idx * bay_w))
            b_p1 = (p0[0] + tx * ((b_idx + 1) * bay_w), p0[1] + ty * ((b_idx + 1) * bay_w))

            if bay_w >= 2.1:
                h_val = (building_id * 7 + i * 13 + b_idx * 5) % 10
                if all_upper_balconies:
                    if b_idx % 4 == 1:
                        mode = "cafe_glass"
                    elif b_idx % 4 == 3:
                        mode = "steel_shutter"
                    else:
                        mode = "souq_open"
                elif force_souq_ground:
                    if (b_idx + i + building_id) % 3 == 1:
                        mode = "steel_shutter"
                    elif (b_idx + i) % 5 == 4:
                        mode = "cafe_glass"
                    else:
                        mode = "souq_open"
                else:
                    if h_val < 4:
                        mode = "steel_shutter"
                    elif h_val < 6:
                        mode = "steel_door"
                    elif h_val < 8:
                        mode = "wood_door"
                    else:
                        mode = "souq_open"

                sh_var = shutter_keys[(building_id + i * 3 + b_idx) % len(shutter_keys)]
                show_sign = (mode in ("souq_open", "cafe_glass", "steel_shutter")) and (b_idx % 2 == 0) and not all_upper_balconies
                s_key = forced_sign_key if (forced_sign_key and b_idx == num_bays // 2) else SHOP_SIGN_KEYS[(building_id + i + b_idx) % len(SHOP_SIGN_KEYS)]
                s_uv = get_sign_uv(sign_atlas, s_key) if show_sign else None
                awn_mat = AWNING_MATS[(building_id + i + b_idx) % len(AWNING_MATS)]

                build_ground_shop_bay(
                    collector,
                    b_p0,
                    b_p1,
                    base_z,
                    base_z + ground_h,
                    (nx, ny),
                    pier_mat=ground_pier_mat,
                    opening_mode=mode,
                    shutter_variant=sh_var,
                    has_awning=(mode in ("souq_open", "cafe_glass") and (b_idx + i) % 2 == 0),
                    awning_mat=awn_mat,
                    sign_uv=s_uv,
                    has_blade_sign=is_hero and (b_idx == 0) and not all_upper_balconies,
                    has_display=(mode == "souq_open" and force_souq_ground and b_idx % 2 == 0),
                    seed=building_id + i * 17 + b_idx,
                )

                if b_idx == 0 and (building_id + i) % 2 == 0 and wall_corridor_dist <= 18.0:
                    ux = b_p0[0] + tx * 0.28
                    uy = b_p0[1] + ty * 0.28
                    build_wall_utility_box(collector, (ux, uy, base_z + 1.65), (tx, ty), (nx, ny), seed=building_id + i)
            else:
                collector.add_quad(
                    "Buildings_Walls", ground_pier_mat,
                    (b_p0[0], b_p0[1], base_z), (b_p1[0], b_p1[1], base_z),
                    (b_p1[0], b_p1[1], base_z + ground_h), (b_p0[0], b_p0[1], base_z + ground_h),
                    expected_normal=(nx, ny, 0.0),
                )

        # 2. String Course above Ground Floor + Upper Storeys
        for lvl in range(1, levels):
            fl_z0 = base_z + ground_h + (lvl - 1) * upper_floor_h
            fl_z1 = fl_z0 + upper_floor_h

            collector.add_oriented_box(
                "Buildings_Details",
                trim_stone_mat,
                p0,
                p1,
                fl_z0 - 0.07,
                fl_z0 + 0.08,
                0.0,
                0.085,
                (nx, ny),
                cap_ends=False,
            )

            add_3d_win_extras = (lvl <= 2) and (wall_corridor_dist <= 22.0)
            for b_idx in range(num_bays):
                b_p0 = (p0[0] + tx * (b_idx * bay_w), p0[1] + ty * (b_idx * bay_w))
                b_p1 = (p0[0] + tx * ((b_idx + 1) * bay_w), p0[1] + ty * ((b_idx + 1) * bay_w))
                if bay_w >= 1.85:
                    use_balcony = all_upper_balconies or (is_hero and lvl == 1 and b_idx == num_bays // 2)
                    build_upper_window_bay(
                        collector,
                        b_p0,
                        b_p1,
                        fl_z0 + 0.08,
                        fl_z1,
                        (nx, ny),
                        wall_mat=upper_wall_mat,
                        trim_mat=trim_stone_mat,
                        recess=0.28,
                        has_shutters=add_3d_win_extras,
                        has_corbels=add_3d_win_extras,
                        has_grille=(lvl == 1 and (b_idx + building_id) % 3 == 0 and not use_balcony),
                        has_balcony=use_balcony,
                    )
                else:
                    collector.add_quad(
                        "Buildings_Walls", upper_wall_mat,
                        (b_p0[0], b_p0[1], fl_z0 + 0.08), (b_p1[0], b_p1[1], fl_z0 + 0.08),
                        (b_p1[0], b_p1[1], fl_z1), (b_p0[0], b_p0[1], fl_z1),
                        expected_normal=(nx, ny, 0.0),
                    )

                if is_hero and b_idx < num_bays:
                    pil_u = b_idx * bay_w
                    pil_p0 = (p0[0] + tx * max(0.0, pil_u - 0.11), p0[1] + ty * max(0.0, pil_u - 0.11))
                    pil_p1 = (p0[0] + tx * min(edge_len, pil_u + 0.11), p0[1] + ty * min(edge_len, pil_u + 0.11))
                    collector.add_oriented_box(
                        "Buildings_Details",
                        trim_stone_mat,
                        pil_p0,
                        pil_p1,
                        fl_z0 + 0.08,
                        fl_z1,
                        0.0,
                        0.085,
                        (nx, ny),
                        cap_top=False,
                    )

        # 3. Projecting Moulded Stone Cornice + Roof Parapet Wall
        collector.add_oriented_box(
            "Buildings_Details",
            trim_stone_mat,
            p0,
            p1,
            roof_z,
            roof_z + 0.22,
            0.0,
            0.20,
            (nx, ny),
            cap_bottom=True,
            cap_ends=False,
        )
        if rooftop_balustrade:
            collector.add_oriented_box(
                "Buildings_Walls",
                upper_wall_mat,
                p0,
                p1,
                roof_z + 0.22,
                roof_z + 0.52,
                -0.26,
                0.0,
                (nx, ny),
                cap_ends=False,
            )
            build_iron_railing(
                collector,
                (p0[0], p0[1], roof_z + 0.52),
                (p1[0], p1[1], roof_z + 0.52),
                height=0.58,
                picket_spacing=0.32,
            )
        else:
            collector.add_oriented_box(
                "Buildings_Walls",
                upper_wall_mat,
                p0,
                p1,
                roof_z + 0.22,
                roof_z + 0.85,
                -0.26,
                0.0,
                (nx, ny),
                cap_ends=False,
            )

    # 4. Recessed Flat Stone Roof Deck (`roof_z + 0.15`)
    roof_pts = [(p[0], p[1], roof_z + 0.15) for p in pts]
    collector.add_polygon_3d("Buildings_Walls", "mat_plaster", roof_pts, uv_scale=0.25)

    # 5. Ottoman Roof Dome or Rooftop Solar Water Heater (`Dud Shemesh`) + AC Clutter
    if min(span_x, span_y) >= 6.0 and lod_tier <= 1:
        if has_dome or (building_id % 4 == 0):
            dome_r = min(2.6, min(span_x, span_y) * 0.24)
            collector.add_cylinder(
                "Buildings_Details",
                trim_stone_mat,
                (cx, cy, roof_z + 0.15),
                dome_r * 1.04,
                dome_r,
                0.55,
                segments=8,
                smooth=False,
            )
            collector.add_dome(
                "Buildings_Details",
                "mat_plaster",
                (cx, cy, roof_z + 0.70),
                radius=dome_r,
                height=dome_r * 0.78,
                rings=3,
                segments=8,
                smooth=True,
            )
        if lod_tier == 0 and (building_id % 2 == 0 or is_hero):
            build_rooftop_clutter(collector, cx, cy, roof_z + 0.15, span_x, span_y, seed=building_id)


# ==============================================================================
# 3. MONUMENTAL LANDMARK BUILDERS
# ==============================================================================

def build_jaffa_gate_monument(
    collector: MeshCollector,
    sign_atlas: Dict[str, dict],
) -> None:
    """
    Constructs the 1538 Ottoman Jaffa Gate (`Bab el-Khalil`) at `(0, 0)` with:
    - Walkable Groin-Vaulted L-Turn Interior Passage with a 100% SEALED stone ceiling
      (`bottom_face=True` + portal spandrel walls) so ZERO sky leaks above the arch ribs.
    - West Outer Facade (`x = -8.5`): pointed outer arch, inner portal, Suleiman inscription
      tympanum, projecting corbelled machicolation box, arrow slits, rosettes, and merlons.
    - Interior Chamber: open iron-strapped wooden gate leaves, un-mirrored Karakashian
      ceramic street name plaques (`jaffa_gate_plaque`), and glowing wall lanterns.
    """
    # --- A. Structural Piers framing the Walkable L-Turn Passage ---
    # 1. North Solid Wall Wing (`x = -8.5 .. 5.8, y = 5.2 .. 8.8, z = -1.0 .. 15.8`)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-1.35, 7.0, 2.6), (14.3, 3.6, 7.2), bottom_face=True)
    collector.add_box("JaffaGate_Monument", "mat_stone_ashlar_aged", (-1.35, 7.0, 11.0), (14.3, 3.6, 9.6))

    # 2. South-West Bastion Pier (`x = -8.5 .. -2.2, y = -5.5 .. 0.2, z = -1.0 .. 15.8`)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-5.35, -2.65, 2.6), (6.3, 5.7, 7.2), bottom_face=True)
    collector.add_box("JaffaGate_Monument", "mat_stone_ashlar_aged", (-5.35, -2.65, 11.0), (6.3, 5.7, 9.6))

    # 3. East Blind Curtain Wall of the L-Turn (`x = 4.0 .. 6.2, y = -5.5 .. 8.8, z = -1.0 .. 15.8`)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (5.1, 1.65, 2.6), (2.2, 14.3, 7.2), bottom_face=True)
    collector.add_box("JaffaGate_Monument", "mat_stone_ashlar_aged", (5.1, 1.65, 11.0), (2.2, 14.3, 9.6))

    # 4. Sealed Vaulted Ceiling Slab (`z = 5.25 .. 5.75`) + Upper Gatehouse Tower (`z = 5.4 .. 15.8`)
    collector.add_box(
        "JaffaGate_Monument",
        "mat_plaster",
        (-1.6, 1.65, 5.50),
        (14.2, 14.2, 0.55),
        top_face=True,
        bottom_face=True,
    )
    collector.add_box(
        "JaffaGate_Monument",
        "mat_stone_ashlar",
        (-1.15, 1.65, 10.6),
        (14.7, 14.3, 10.4),
        top_face=True,
        bottom_face=True,
    )

    # Solid Portal Spandrel Walls above West Portal & South Exit Portal so arch corners have zero sky gaps
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-7.4, 2.6, 5.15), (2.2, 5.4, 1.3), bottom_face=True)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (0.9, -4.85, 5.05), (6.8, 1.5, 1.5), bottom_face=True)

    # 5. Groin/Barrel Vaulted Ribs inside the L-Passage (`z = 4.1 .. 5.5`)
    for arch_x in (-7.2, -5.0, -2.8, 1.0):
        add_arch_voussoirs(
            collector,
            "JaffaGate_Monument",
            "mat_stone_ashlar_aged",
            (arch_x, 2.6, 4.1),
            (0.0, 1.0),
            (-1.0, 0.0),
            span_w=4.8,
            arch_h=1.25,
            ring_thickness=0.38,
            offset_n0=-0.45,
            offset_n1=0.45,
            num_voussoirs=7,
        )
    # South Exit Portal Arch (`y = -5.5, x = -2.2 .. 4.0`)
    add_arch_voussoirs(
        collector,
        "JaffaGate_Monument",
        "mat_stone_boss",
        (0.9, -5.3, 3.65),
        (1.0, 0.0),
        (0.0, -1.0),
        span_w=5.2,
        arch_h=1.55,
        ring_thickness=0.45,
        offset_n0=-0.65,
        offset_n1=0.25,
        num_voussoirs=9,
        keystone_boost=0.09,
    )

    # --- B. West Outer Monumental Facade (`x = -8.5`) ---
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-7.4, 0.35, 2.6), (2.2, 0.7, 5.4))
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-7.4, 4.85, 2.6), (2.2, 0.7, 5.4))

    add_arch_voussoirs(
        collector,
        "JaffaGate_Monument",
        "mat_stone_boss",
        (-8.1, 2.6, 3.55),
        (0.0, 1.0),
        (-1.0, 0.0),
        span_w=3.8,
        arch_h=1.45,
        ring_thickness=0.38,
        offset_n0=-0.45,
        offset_n1=0.18,
        num_voussoirs=9,
        keystone_boost=0.08,
    )

    add_arch_voussoirs(
        collector,
        "JaffaGate_Monument",
        "mat_stone_boss",
        (-8.45, 2.6, 5.15),
        (0.0, 1.0),
        (-1.0, 0.0),
        span_w=6.0,
        arch_h=2.45,
        ring_thickness=0.52,
        offset_n0=-0.15,
        offset_n1=0.42,
        num_voussoirs=11,
        keystone_boost=0.12,
    )

    # Sultan Suleiman Inscription Tympanum Plaque (`x = -8.35, y = 1.1 .. 4.1, z = 5.3 .. 6.5`)
    collector.add_box("JaffaGate_Monument", "mat_stone_ashlar_aged", (-8.38, 2.6, 5.9), (0.24, 3.1, 1.25))
    collector.add_box("JaffaGate_Monument", "mat_stone_mizzi", (-8.48, 2.6, 5.9), (0.08, 2.7, 0.92))

    # Projecting Corbelled Stone Machicolation Box high on the West Facade (`z = 11.4 .. 13.7`)
    for c_y in (0.7, 1.65, 2.6, 3.55, 4.5):
        collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-8.72, c_y, 10.95), (0.55, 0.34, 0.45), bottom_face=True)
        collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-8.92, c_y, 11.35), (0.85, 0.36, 0.42), bottom_face=True)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-8.98, 2.6, 12.65), (0.95, 4.3, 2.25), bottom_face=True)

    for slit_y in (-2.6, 7.1):
        collector.add_box("JaffaGate_Monument", "mat_iron_dark", (-8.52, slit_y, 9.2), (0.12, 0.22, 1.95))
        collector.add_cylinder(
            "JaffaGate_Monument",
            "mat_stone_boss",
            (-8.58, slit_y, 12.2),
            0.42,
            0.38,
            0.14,
            segments=10,
            smooth=False,
        )

    # Crenellated Ottoman Bevelled Merlons (`z = 15.8 .. 17.2`)
    collector.add_box("JaffaGate_Monument", "mat_stone_ashlar_aged", (-1.15, 1.65, 15.65), (15.1, 14.7, 0.28))
    for mx in (-7.6, -5.2, -2.8, -0.4, 2.0, 4.4):
        for my in (-5.0, 8.3):
            collector.add_box("JaffaGate_Monument", "mat_stone_ashlar_aged", (mx, my, 16.5), (1.35, 0.65, 1.4))
    for my in (-4.4, -2.0, 0.4, 2.8, 5.2, 7.6):
        for mx in (-8.15, 5.85):
            collector.add_box("JaffaGate_Monument", "mat_stone_ashlar_aged", (mx, my, 16.5), (0.65, 1.35, 1.4))

    # --- C. Interior L-Passage Doors, Tiled Street Name Plates & Lanterns ---
    collector.add_box("JaffaGate_Monument", "mat_wood_doors", (-4.8, 5.02, 2.35), (1.85, 0.16, 4.5), uv_rect=WOOD_DOOR_UV["door_wood"])
    collector.add_box("JaffaGate_Monument", "mat_wood_doors", (-4.8, 0.32, 2.35), (1.85, 0.16, 4.5), uv_rect=WOOD_DOOR_UV["door_wood"])
    for strap_z in (0.65, 1.75, 2.85, 3.95):
        collector.add_box("JaffaGate_Monument", "mat_iron_dark", (-4.8, 4.92, strap_z), (1.88, 0.04, 0.11))
        collector.add_box("JaffaGate_Monument", "mat_iron_dark", (-4.8, 0.42, strap_z), (1.88, 0.04, 0.11))

    # Trilingual Karakashian ceramic street name plaques ("JAFFA GATE / BAB EL-KHALIL")
    jg_uv = get_sign_uv(sign_atlas, "jaffa_gate_plaque")
    build_street_name_plate(collector, (3.92, 2.6, 2.45), (-1.0, 0.0), jg_uv, width=1.25, height=0.62)
    build_street_name_plate(collector, (-8.52, -1.5, 2.45), (-1.0, 0.0), jg_uv, width=1.15, height=0.58)
    build_street_name_plate(collector, (4.85, -5.52, 2.45), (0.0, -1.0), jg_uv, width=1.05, height=0.54)

    build_wall_lantern(collector, (-6.2, 4.95, 3.6), (1.0, 0.0), (0.0, -1.0))
    build_wall_lantern(collector, (3.85, 0.5, 3.6), (0.0, 1.0), (-1.0, 0.0))

    # --- D. Northern & Southern Ottoman City Wall Segments ---
    nw_wall_pts = [
        (-3.5, 8.5),
        (-14.0, 26.0),
        (-28.0, 50.0),
        (-48.0, 82.0),
        (-74.0, 122.0),
    ]
    sw_wall_pts = [
        (13.5, -78.0),
        (16.0, -115.0),
        (20.0, -155.0),
    ]
    for wall_chain in (nw_wall_pts, sw_wall_pts):
        for i in range(len(wall_chain) - 1):
            wp0 = wall_chain[i]
            wp1 = wall_chain[i + 1]
            dx = wp1[0] - wp0[0]
            dy = wp1[1] - wp0[1]
            seg_l = math.hypot(dx, dy)
            nx, ny = -dy / seg_l, dx / seg_l
            collector.add_oriented_box("CityWalls", "mat_stone_boss", wp0, wp1, -2.5, 5.5, -1.4, 1.4, (nx, ny), cap_back=True)
            collector.add_oriented_box("CityWalls", "mat_stone_ashlar_aged", wp0, wp1, 5.5, 12.8, -1.2, 1.2, (nx, ny), cap_back=True)
            num_m = int(seg_l / 2.4)
            tx, ty = dx / seg_l, dy / seg_l
            for m_i in range(num_m):
                u0 = m_i * 2.4 + 0.3
                u1 = u0 + 1.35
                mp0 = (wp0[0] + tx * u0, wp0[1] + ty * u0)
                mp1 = (wp0[0] + tx * u1, wp0[1] + ty * u1)
                collector.add_oriented_box("CityWalls", "mat_stone_ashlar_aged", mp0, mp1, 12.8, 14.1, 0.4, 1.2, (nx, ny), cap_back=True)


def build_citadel_and_tower_of_david(
    collector: MeshCollector,
    sign_atlas: Dict[str, dict],
) -> None:
    """
    Constructs the Citadel (`El-Qal'a`) and Tower of David (`Phasael Tower`) complex:
    - Widened 13.5m Sunken Dry Moat (`x = 10..54, y = -29.0..-15.5, z = -6.0m`) with low
      `0.42m` stone counterscarp curb + `0.78m` open wrought-iron railing so the full
      moat basin, tumbled ruins, and Date Palm trunk at `(20.5, -21.5, -6.0)` are clearly
      visible from Omar Square and the 1898 Breach!
    - Continuous Battered Stone Glacis (`z = -6.0m .. +1.4m`) rising seamlessly into the
      unified `mat_stone_boss` Citadel Curtain Wall (`z = +1.4m .. +14.6m`).
    - Phasael Tower (`Tower of David`) at `(24.5, -34.5)` rising to `z = +27.5m` (`33.5m` above moat).
    - Arched Stone Moat Bridge (`x = 42..47, y = -29.0..-15.2`) & Museum Gatehouse (`y = -30.5`).
    - Cylindrical Mamluk Minaret at `(18.0, -66.0)` with triple muqarnas balconies.
    """
    moat_z = -6.0
    street_z = 0.35

    # --- 1. Widened Sunken Dry Moat Floor & Low 0.42m Counterscarp Curb + 0.78m Open Iron Railing ---
    moat_floor = [
        (9.5, -15.0, moat_z),
        (54.5, -15.0, moat_z),
        (54.5, -58.0, moat_z),
        (9.5, -58.0, moat_z),
    ]
    collector.add_polygon_3d("Citadel_TowerOfDavid", "mat_stone_rubble", moat_floor, uv_scale=0.22)

    counterscarp_segs = [
        ((10.0, -15.5), (42.0, -15.5), (0.0, -1.0)),  # North moat edge along Breach / Omar Sq
        ((47.0, -15.5), (54.0, -15.5), (0.0, -1.0)),  # East of Moat Bridge
        ((10.0, -56.0), (10.0, -15.5), (1.0, 0.0)),   # West moat edge along Breach road
        ((54.0, -56.0), (54.0, -15.5), (-1.0, 0.0)),  # East moat edge along Omar Sq south
    ]
    for p0, p1, n_in in counterscarp_segs:
        # Sub-street counterscarp retaining wall from moat_z (-6.0) up to low 0.36m curb
        collector.add_oriented_box(
            "Citadel_TowerOfDavid",
            "mat_stone_ashlar_aged",
            p0,
            p1,
            moat_z,
            street_z + 0.36,
            -0.52,
            0.0,
            n_in,
        )
        # Low coped stone curb cap (`street_z + 0.36 .. street_z + 0.44m`)
        collector.add_oriented_box(
            "Citadel_TowerOfDavid",
            "mat_stone_mizzi",
            p0,
            p1,
            street_z + 0.36,
            street_z + 0.44,
            -0.58,
            0.05,
            n_in,
        )
        # Tall open wrought-iron safety railing (`0.78m`) so camera sees right through into the moat
        build_iron_railing(
            collector,
            (p0[0], p0[1], street_z + 0.44),
            (p1[0], p1[1], street_z + 0.44),
            height=0.78,
            picket_spacing=0.24,
        )

    # --- 2. Exposed Date Palm Trees & Archaeological Ruin Blocks inside the Widened Moat ---
    build_palm_tree(collector, 20.5, -21.2, moat_z, height=13.2, seed=101)
    build_palm_tree(collector, 35.5, -21.5, moat_z, height=11.5, seed=106)
    build_olive_tree(collector, 29.0, -20.5, moat_z, height=6.6, seed=102, with_kerb=False)
    build_cypress_tree(collector, 12.2, -38.0, moat_z, height=9.2, seed=103, with_kerb=False)

    ruin_blocks = [
        (16.5, -19.5, 2.4, 1.5, 1.4, 0.15),
        (25.2, -19.8, 3.2, 1.3, 1.2, -0.08),
        (32.5, -22.2, 2.0, 2.0, 1.5, 0.42),
        (39.0, -20.0, 2.6, 1.4, 1.2, -0.22),
        (12.5, -27.5, 2.5, 1.2, 1.3, 0.05),
    ]
    for rx, ry, sx, sy, sz, rot in ruin_blocks:
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (rx, ry, moat_z + sz * 0.5), (sx, sy, sz), rot_z=rot)

    # --- 3. Continuous Battered Stone Glacis + Unified Citadel Curtain Wall (`mat_stone_boss`) ---
    citadel_ring = [
        (15.0, -29.0),  # NW corner (Phasael Tower base)
        (14.0, -54.0),  # West curtain wall
        (13.5, -78.0),  # SW Minaret Bastion corner
        (46.0, -82.0),  # South curtain wall
        (56.0, -54.0),  # SE Bastion
        (50.0, -29.0),  # NE Gatehouse corner
    ]
    n_cit = len(citadel_ring)
    glacis_top_z = 1.4
    curtain_top_z = 14.6

    for i in range(n_cit):
        cp0 = citadel_ring[i]
        cp1 = citadel_ring[(i + 1) % n_cit]
        dx = cp1[0] - cp0[0]
        dy = cp1[1] - cp0[1]
        seg_l = math.hypot(dx, dy)
        tx, ty = dx / seg_l, dy / seg_l
        nx, ny = ty, -tx

        # 3a. Sloped Battered Stone Glacis (`z = -6.0 .. +1.4m`, flaring 3.4m outward into the moat)
        glacis_flare = 3.4
        gb0 = (cp0[0] + nx * glacis_flare, cp0[1] + ny * glacis_flare, moat_z)
        gb1 = (cp1[0] + nx * glacis_flare, cp1[1] + ny * glacis_flare, moat_z)
        gt1 = (cp1[0], cp1[1], glacis_top_z)
        gt0 = (cp0[0], cp0[1], glacis_top_z)
        collector.add_quad(
            "Citadel_TowerOfDavid",
            "mat_stone_boss",
            gb0,
            gb1,
            gt1,
            gt0,
            uv_scale=0.24,
            expected_normal=(nx, ny, 0.5),
        )

        # 3b. Unified Vertical Citadel Curtain Wall (`z = +1.4 .. +14.6m`, `mat_stone_boss`)
        collector.add_oriented_box(
            "Citadel_TowerOfDavid",
            "mat_stone_boss",
            cp0,
            cp1,
            glacis_top_z,
            curtain_top_z,
            -2.2,
            0.0,
            (nx, ny),
            cap_back=True,
        )
        collector.add_oriented_box(
            "Citadel_TowerOfDavid",
            "mat_stone_boss",
            cp0,
            cp1,
            glacis_top_z - 0.12,
            glacis_top_z + 0.16,
            -0.2,
            0.18,
            (nx, ny),
            cap_ends=False,
        )
        collector.add_oriented_box(
            "Citadel_TowerOfDavid",
            "mat_stone_boss",
            cp0,
            cp1,
            curtain_top_z - 0.22,
            curtain_top_z + 0.08,
            -2.3,
            0.18,
            (nx, ny),
            cap_ends=False,
        )

        # 3c. Crenellated Battlement Merlons (`z = 14.6 .. 15.95m`)
        num_merlons = max(2, int(seg_l / 2.3))
        for m_i in range(num_merlons):
            u0 = m_i * 2.3 + 0.25
            u1 = min(seg_l - 0.2, u0 + 1.35)
            if u1 <= u0:
                continue
            mp0 = (cp0[0] + tx * u0, cp0[1] + ty * u0)
            mp1 = (cp0[0] + tx * u1, cp0[1] + ty * u1)
            collector.add_oriented_box(
                "Citadel_TowerOfDavid",
                "mat_stone_boss",
                mp0,
                mp1,
                curtain_top_z,
                curtain_top_z + 1.35,
                -0.65,
                0.0,
                (nx, ny),
                cap_back=True,
            )

    courtyard_pts = [(p[0], p[1], 3.5) for p in citadel_ring]
    collector.add_polygon_3d("Citadel_TowerOfDavid", "mat_paving_flagstone", courtyard_pts, uv_scale=0.30)
    build_cypress_tree(collector, 32.0, -44.0, 3.5, height=11.0, seed=104, with_kerb=False)
    build_olive_tree(collector, 36.0, -54.0, 3.5, height=7.8, seed=105, with_kerb=False)

    # --- 4. Phasael Tower (`Tower of David`, NW Bastion at `x = 24.5, y = -34.5`) ---
    px, py = 24.5, -34.5
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px, py, 6.4), (19.5, 16.0, 16.8))
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px, py, 15.3), (18.6, 15.2, 1.0))
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px, py, 20.8), (17.8, 14.4, 10.0))
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px, py, 26.05), (18.5, 15.1, 0.5))

    for m_x in (px - 4.2, px, px + 4.2):
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (m_x, py + 7.65, 23.0), (2.3, 0.95, 2.1), bottom_face=True)
        collector.add_box("Citadel_TowerOfDavid", "mat_iron_dark", (m_x, py + 8.15, 19.6), (0.24, 0.12, 1.9))
    for m_y in (py - 3.2, py + 3.2):
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px - 9.25, m_y, 23.0), (0.95, 2.3, 2.1), bottom_face=True)
        collector.add_box("Citadel_TowerOfDavid", "mat_iron_dark", (px - 9.75, m_y, 19.6), (0.12, 0.24, 1.9))

    for mx in range(-8, 9, 3):
        for side_y in (py - 7.1, py + 7.1):
            collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px + mx, side_y, 27.0), (1.55, 0.68, 1.4))
    for my in range(-5, 6, 3):
        for side_x in (px - 8.8, px + 8.8):
            collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (side_x, py + my, 27.0), (0.68, 1.55, 1.4))

    collector.add_box("Citadel_TowerOfDavid", "mat_stone_ashlar_aged", (px + 3.2, py + 1.8, 27.7), (4.4, 4.0, 2.8))
    collector.add_dome("Citadel_TowerOfDavid", "mat_stone_ashlar_aged", (px + 3.2, py + 1.8, 29.1), radius=1.8, height=1.2)

    # --- 5. Arched Stone Citadel Moat Bridge (`x = 42..47, y = -29.0..-15.2`) & Museum Entrance ---
    bx = 44.5
    for pier_y in (-19.8, -24.5):
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (bx, pier_y, -2.8), (4.8, 1.4, 6.4))
    collector.add_box("Citadel_TowerOfDavid", "mat_paving_flagstone", (bx, -22.1, 0.28), (4.6, 14.0, 0.55), bottom_face=True)
    for side_x in (bx - 2.15, bx + 2.15):
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_ashlar_aged", (side_x, -22.1, 0.72), (0.38, 14.0, 0.42))
        build_iron_railing(collector, (side_x, -15.2, 0.94), (side_x, -28.8, 0.94), height=0.68)

    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (bx, -30.5, 5.2), (7.4, 3.8, 10.4))
    add_arch_voussoirs(
        collector,
        "Citadel_TowerOfDavid",
        "mat_stone_boss",
        (bx, -28.55, 2.8),
        (1.0, 0.0),
        (0.0, 1.0),
        span_w=3.2,
        arch_h=1.45,
        ring_thickness=0.35,
        offset_n0=-0.4,
        offset_n1=0.18,
        num_voussoirs=7,
    )
    collector.add_quad(
        "Buildings_Openings",
        "mat_wood_doors",
        (bx + 1.5, -28.65, 0.45),
        (bx - 1.5, -28.65, 0.45),
        (bx - 1.5, -28.65, 3.95),
        (bx + 1.5, -28.65, 3.95),
        uv_rect=WOOD_DOOR_UV["door_wood"],
        expected_normal=(0.0, 1.0, 0.0),
    )
    tod_uv = get_sign_uv(sign_atlas, "tower_of_david_museum")
    build_fascia_sign(collector, (bx, -28.52, 4.95), (1.0, 0.0), (0.0, 1.0), width=2.8, height=1.1, sign_uv_rect=tod_uv)
    build_street_name_plate(collector, (41.6, -15.35, 1.25), (0.0, 1.0), tod_uv, width=1.45, height=0.68)

    # --- 6. Mamluk Citadel Minaret (`x = 18.0, y = -66.0`) ---
    mx, my = 18.0, -66.0
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (mx, my, 6.5), (11.2, 11.2, 17.0))
    collector.add_cylinder("Citadel_TowerOfDavid", "mat_stone_ashlar_aged", (mx, my, 15.0), 3.3, 2.8, 3.2, segments=8, smooth=False)
    collector.add_cylinder("Citadel_TowerOfDavid", "mat_stone_ashlar", (mx, my, 18.2), 2.48, 2.28, 10.8, segments=18, smooth=True)
    collector.add_cylinder("Citadel_TowerOfDavid", "mat_stone_boss", (mx, my, 26.4), 2.38, 2.85, 0.55, segments=16, smooth=False)
    collector.add_cylinder("Citadel_TowerOfDavid", "mat_stone_mizzi", (mx, my, 26.95), 2.85, 3.35, 0.65, segments=16, smooth=False)
    collector.add_cylinder("Citadel_TowerOfDavid", "mat_stone_mizzi", (mx, my, 27.60), 3.35, 3.35, 1.15, segments=16, smooth=False)
    collector.add_cylinder("Citadel_TowerOfDavid", "mat_stone_ashlar", (mx, my, 29.0), 1.75, 1.70, 2.8, segments=14, smooth=True)
    collector.add_dome("Citadel_TowerOfDavid", "mat_iron_dark", (mx, my, 31.8), radius=1.82, height=2.1, rings=5, segments=14, smooth=True)
    collector.add_cylinder("Citadel_TowerOfDavid", "mat_iron_dark", (mx, my, 33.9), 0.08, 0.03, 1.2, segments=6, smooth=True)


# ==============================================================================
# 4. HERO OMAR SQUARE, LATIN PATRIARCHATE RD, DAVID ST & WESTERN HORIZON
# ==============================================================================

def build_hero_omar_square_and_david_street(
    collector: MeshCollector,
    sign_atlas: Dict[str, dict],
    elev_fn: Callable[[float, float], float],
) -> None:
    """
    Constructs the curated hero buildings, street name plates, stepped bazaar corridors,
    and western Mamilla horizon embankment:
    1. Omar ibn al-Khattab Square:
       - `Tourist Information / Ministry of Tourism` (`x = 5.5..18.5, y = 10..20.5`)
       - `New Imperial Hotel` (`7002`) with chamfered SW corner, full upper-storey wrought-iron
         balconies, stone pilasters, rooftop iron balustrade, and clean spandrel fascia plaques
       - `Petra Hotel` (`7003`) at NE corner of Omar Square with full upper balconies & balustrade
       - `Christian Information Centre` (`7004`) shifted south to `y = -34.0..-18.5, x = 58..82`
         so it never obstructs the Citadel curtain wall from Omar Square!
    2. Latin Patriarchate Road (`x = 8..48, y = 25..56`) with flying-buttress arches (`qantara`)
    3. David Street (`Suq el-Bazar`) Stepped Bazaar & Vaulted East Terminus (`x = 78..168`)
    4. Western Mamilla Horizon Embankment & Terraced Buildings (`x = -132..-118, y = -95..95`)
    """
    # --- 1A. Tourist Information / Ministry of Tourism (`7001`, `x = 5.5..18.5, y = 10.0..20.5`) ---
    bz_7001 = elev_fn(12.0, 10.0)
    build_building_from_footprint(
        collector,
        [(5.5, 10.0), (18.5, 10.0), (18.5, 20.5), (5.5, 20.5)],
        base_z=bz_7001,
        levels=2,
        sign_atlas=sign_atlas,
        building_id=7001,
        is_hero=True,
        force_souq_ground=True,
        has_dome=True,
    )
    omar_uv = get_sign_uv(sign_atlas, "omar_sq_plaque")
    build_street_name_plate(collector, (12.0, 9.96, bz_7001 + 2.35), (0.0, -1.0), omar_uv, width=0.95, height=0.55)
    build_street_name_plate(collector, (18.54, 12.2, bz_7001 + 2.35), (1.0, 0.0), omar_uv, width=0.92, height=0.52)

    # --- 1B. New Imperial Hotel (`7002`) with Chamfered SW Corner, Full Balconies & Rooftop Balustrade ---
    bz_7002 = elev_fn(36.0, 14.5)
    build_building_from_footprint(
        collector,
        [
            (25.2, 14.5),
            (50.0, 14.5),
            (50.0, 26.5),
            (22.0, 26.5),
            (22.0, 17.5),
            (23.3, 15.4),
        ],
        base_z=bz_7002,
        levels=3,
        sign_atlas=sign_atlas,
        building_id=7002,
        is_hero=True,
        force_souq_ground=True,
        has_dome=True,
        all_upper_balconies=True,
        rooftop_balustrade=True,
    )
    nih_uv = get_sign_uv(sign_atlas, "new_imperial_hotel")
    # Cleanly mounted in the ground-floor spandrel band (`z = bz + 3.68m, height = 0.58m`) below the 1st-floor balcony
    build_fascia_sign(collector, (36.5, 14.46, bz_7002 + 3.68), (1.0, 0.0), (0.0, -1.0), width=2.85, height=0.58, sign_uv_rect=nih_uv)
    build_blade_sign(collector, (23.2, 15.3, bz_7002 + 3.55), (0.0, 1.0), (-0.707, -0.707), width=0.92, height=0.52, sign_uv_rect=nih_uv)

    # Trilingual Street Name Plate on New Imperial Hotel SW corner facing Latin Patriarchate Rd / Omar Sq
    latin_uv = get_sign_uv(sign_atlas, "latin_rd_plaque")
    build_street_name_plate(collector, (21.96, 19.2, bz_7002 + 2.35), (-1.0, 0.0), latin_uv, width=0.92, height=0.54)

    # --- 1C. Petra Hotel (`7003`, `x = 64.0..86.0, y = 11.5..27.5`) at NE Corner of Omar Square ---
    bz_7003 = elev_fn(74.0, 11.5)
    build_building_from_footprint(
        collector,
        [(64.0, 11.5), (86.0, 11.5), (86.0, 27.5), (64.0, 27.5)],
        base_z=bz_7003,
        levels=3,
        sign_atlas=sign_atlas,
        building_id=7003,
        is_hero=True,
        force_souq_ground=True,
        has_dome=True,
        all_upper_balconies=True,
        rooftop_balustrade=True,
    )
    petra_uv = get_sign_uv(sign_atlas, "petra_hotel")
    build_fascia_sign(collector, (73.5, 11.46, bz_7003 + 3.68), (1.0, 0.0), (0.0, -1.0), width=2.75, height=0.58, sign_uv_rect=petra_uv)
    build_blade_sign(collector, (63.96, 13.0, bz_7003 + 3.55), (0.0, 1.0), (-1.0, 0.0), width=0.92, height=0.52, sign_uv_rect=petra_uv)
    build_street_name_plate(collector, (64.0, 11.46, bz_7003 + 2.35), (0.0, -1.0), omar_uv, width=0.92, height=0.54)

    # --- 1D. Christian Information Centre (`7004`, Moved South to `x = 58..82, y = -34.0..-18.5`) ---
    bz_7004 = elev_fn(68.0, -18.5)
    build_building_from_footprint(
        collector,
        [(58.0, -34.0), (82.0, -34.0), (82.0, -18.5), (58.0, -18.5)],
        base_z=bz_7004,
        levels=3,
        sign_atlas=sign_atlas,
        building_id=7004,
        is_hero=True,
        force_souq_ground=True,
        has_dome=True,
    )
    ci_uv = get_sign_uv(sign_atlas, "christian_info")
    build_fascia_sign(collector, (68.0, -18.46, bz_7004 + 3.68), (-1.0, 0.0), (0.0, 1.0), width=2.75, height=0.58, sign_uv_rect=ci_uv)
    build_blade_sign(collector, (57.96, -19.5, bz_7004 + 3.55), (0.0, -1.0), (-1.0, 0.0), width=0.90, height=0.50, sign_uv_rect=ci_uv)

    # --- 2. Latin Patriarchate Road (`x = 19..34, y = 25..56`) Overhead Flying-Buttress Arches ---
    build_blade_sign(collector, (19.54, 27.5, elev_fn(19.5, 27.5) + 3.6), (0.0, 1.0), (1.0, 0.0), width=0.90, height=0.50, sign_uv_rect=latin_uv)
    build_street_name_plate(collector, (19.54, 26.2, elev_fn(19.5, 26.2) + 2.35), (1.0, 0.0), latin_uv, width=0.90, height=0.52)

    build_building_from_footprint(
        collector,
        [(8.0, 25.0), (19.5, 25.0), (19.5, 52.0), (8.0, 52.0)],
        base_z=elev_fn(16.0, 36.0),
        levels=3,
        sign_atlas=sign_atlas,
        building_id=7005,
        is_hero=True,
        force_souq_ground=True,
        forced_sign_key="pilgrim_books",
    )
    build_building_from_footprint(
        collector,
        [(33.5, 28.0), (48.0, 28.0), (48.0, 56.0), (33.5, 56.0)],
        base_z=elev_fn(38.0, 40.0),
        levels=3,
        sign_atlas=sign_atlas,
        building_id=7006,
        is_hero=True,
        force_souq_ground=True,
        forced_sign_key="jerusalem_pottery",
    )

    for arch_y in (36.0, 48.0):
        az = elev_fn(26.5, arch_y)
        span = 14.0
        collector.add_box("Buildings_Details", "mat_stone_boss", (20.1, arch_y, az + 2.8), (1.2, 1.4, 5.8))
        collector.add_box("Buildings_Details", "mat_stone_boss", (32.9, arch_y, az + 2.8), (1.2, 1.4, 5.8))
        collector.add_box("Buildings_Details", "mat_stone_ashlar_aged", (26.5, arch_y, az + 6.8), (span, 1.25, 1.6), bottom_face=True)
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_ashlar_aged",
            (26.5, arch_y, az + 4.5),
            (1.0, 0.0),
            (0.0, -1.0),
            span_w=11.6,
            arch_h=1.9,
            ring_thickness=0.42,
            offset_n0=-0.62,
            offset_n1=0.62,
            num_voussoirs=11,
        )
        build_wall_lantern(collector, (26.5, arch_y, az + 5.7), (1.0, 0.0), (0.0, -1.0))

    # --- 3. David Street (`Suq el-Bazar`) Stepped Stone Descent & Bazaar Arcade (`x = 78..168`) ---
    num_steps = 42
    x_start = 78.0
    step_tread = 1.90
    y_south = -2.2
    y_north = 6.2
    step_w = y_north - y_south

    for s_idx in range(num_steps):
        sx0 = x_start + s_idx * step_tread
        sx1 = sx0 + step_tread
        smid = 0.5 * (sx0 + sx1)
        sz_top = elev_fn(smid, 2.0) + 0.06
        sz_bot = sz_top - 0.38
        collector.add_box(
            "Ground_Paving",
            "mat_paving_flagstone",
            (smid, 2.0, 0.5 * (sz_top + sz_bot)),
            (step_tread + 0.04, step_w, sz_top - sz_bot),
        )
        collector.add_box(
            "Ground_Paving",
            "mat_stone_ashlar_aged",
            (sx1 - 0.08, 2.0, sz_top - 0.06),
            (0.18, step_w, 0.14),
        )

    souq_terraces = [
        (80.0, 96.0, "old_city_bazaar", "holy_land_rugs"),
        (96.0, 112.0, "jerusalem_pottery", "falafel_cafe"),
        (112.0, 128.0, "st_james_souvenirs", "money_exchange"),
        (128.0, 144.0, "pilgrim_books", "swedish_study"),
        (144.0, 158.0, "old_city_bazaar", "holy_land_rugs"),
    ]
    david_plaque_uv = get_sign_uv(sign_atlas, "david_st_plaque")
    bz_david0 = elev_fn(80.5, 2.0)
    build_blade_sign(collector, (80.5, 5.75, bz_david0 + 3.45), (1.0, 0.0), (0.0, -1.0), width=0.92, height=0.52, sign_uv_rect=david_plaque_uv)
    build_street_name_plate(collector, (80.2, 5.76, bz_david0 + 2.35), (0.0, -1.0), david_plaque_uv, width=0.92, height=0.54)
    build_street_name_plate(collector, (80.2, -1.76, bz_david0 + 2.35), (0.0, 1.0), david_plaque_uv, width=0.92, height=0.54)

    for t_idx, (tx0, tx1, north_sign, south_sign) in enumerate(souq_terraces):
        t_mid = 0.5 * (tx0 + tx1)
        bz = elev_fn(t_mid, 2.0)
        build_building_from_footprint(
            collector,
            [(tx0, 5.8), (tx1, 5.8), (tx1, 14.2), (tx0, 14.2)],
            base_z=bz,
            levels=3,
            sign_atlas=sign_atlas,
            building_id=7100 + t_idx * 2,
            is_hero=True,
            force_souq_ground=True,
            has_dome=(t_idx % 2 == 0),
            forced_sign_key=north_sign,
        )
        build_building_from_footprint(
            collector,
            [(tx0, -10.2), (tx1, -10.2), (tx1, -1.8), (tx0, -1.8)],
            base_z=bz,
            levels=3,
            sign_atlas=sign_atlas,
            building_id=7101 + t_idx * 2,
            is_hero=True,
            force_souq_ground=True,
            has_dome=(t_idx % 2 == 1),
            forced_sign_key=south_sign,
        )

        arch_x = tx0 + 6.0
        az = elev_fn(arch_x, 2.0)
        collector.add_box("Buildings_Details", "mat_stone_ashlar_aged", (arch_x, 2.0, az + 6.4), (1.35, 8.2, 1.5), bottom_face=True)
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_ashlar_aged",
            (arch_x, 2.0, az + 4.3),
            (0.0, 1.0),
            (-1.0, 0.0),
            span_w=7.4,
            arch_h=1.7,
            ring_thickness=0.38,
            offset_n0=-0.65,
            offset_n1=0.65,
            num_voussoirs=9,
        )
        build_wall_lantern(collector, (arch_x, 2.0, az + 5.4), (0.0, 1.0), (-1.0, 0.0))

    end_z = elev_fn(158.0, 2.0)
    build_building_from_footprint(
        collector,
        [(157.5, -12.0), (168.5, -12.0), (168.5, 16.0), (157.5, 16.0)],
        base_z=end_z,
        levels=3,
        sign_atlas=sign_atlas,
        building_id=7200,
        is_hero=True,
        force_souq_ground=True,
        has_dome=True,
        forced_sign_key="old_city_bazaar",
    )

    # --- 4. Western Mamilla Horizon Embankment & Terraced Buildings (`x = -132 .. -115, y = -95 .. 95`) ---
    # Closes the distant western horizon when looking west through the 1898 Breach (`groundHoleFraction = 0.0000`)
    collector.add_box("CityWalls", "mat_stone_rubble", (-126.0, 0.0, 1.2), (12.0, 190.0, 14.5))
    for blk_idx, by0 in enumerate((-75.0, -45.0, -18.0, 10.0, 38.0)):
        build_building_from_footprint(
            collector,
            [(-124.0, by0), (-110.0, by0), (-110.0, by0 + 22.0), (-124.0, by0 + 22.0)],
            base_z=-1.5,
            levels=3,
            sign_atlas=sign_atlas,
            building_id=7300 + blk_idx,
            is_hero=False,
        )
