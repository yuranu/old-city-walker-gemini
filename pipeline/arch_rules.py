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

from fetch_osm_and_dem import street_elevation_z as _global_street_elevation_z
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
]

AWNING_MATS: List[str] = [
    "mat_awning_red",
    "mat_awning_green",
]

# Active walkable player/camera corridors across all 9 regions of the Old City for 3-Tier Architectural LOD
# Synchronized with exact OpenStreetMap thoroughfare coordinates
_CURATED_STREET_POLYLINES: List[List[Vec2]] = [
    [(-95.0, 14.0), (-42.0, 6.0), (-16.0, 2.5), (-6.2, 2.7), (8.1, -7.7), (34.8, 1.8), (62.5, 10.6), (83.7, 4.8)],  # Jaffa Gate & Omar Sq
    [(-6.2, 2.7), (8.1, -7.7), (10.8, -46.8), (15.5, -80.2), (12.3, -139.2), (16.5, -173.4)],  # Citadel Moat & Maronite Convent St
    [(83.7, 4.8), (93.4, 11.1), (122.8, 10.6), (145.9, 10.1), (162.5, 11.4), (179.8, 15.0), (198.7, 15.0), (230.6, 15.7), (255.4, 18.0), (285.8, 19.2), (302.4, 20.4), (313.6, 20.2), (329.2, 19.4)],  # David St
    [(329.2, 19.4), (331.8, 11.1), (352.1, 12.0), (379.8, 14.1), (404.2, 18.6), (440.9, 24.2), (472.6, 29.2), (505.5, 33.9), (521.3, 36.7), (552.6, 46.4), (587.2, 56.7), (616.5, 65.4), (626.2, 73.2)],  # Street of the Chain
    [(254.8, 555.9), (263.7, 550.5), (271.4, 545.2), (284.4, 527.4), (290.7, 513.0), (295.1, 496.6), (290.5, 493.1), (290.6, 470.3), (292.7, 444.4), (294.2, 403.5), (296.2, 398.5), (298.1, 349.2), (299.5, 329.4), (300.5, 300.8), (304.0, 293.2), (305.1, 264.9), (306.7, 237.1), (306.8, 215.2), (307.7, 193.8), (307.9, 172.5)],  # Khan el-Zeit
    [(307.9, 172.5), (313.8, 138.7), (319.1, 20.2), (320.2, 3.0), (315.4, -46.5), (312.7, -80.0), (310.7, -113.3), (307.7, -170.2), (306.0, -193.5)],  # Triple Souq & Roman Cardo
    [(295.1, 496.6), (298.5, 496.8), (308.5, 489.4), (328.9, 463.3), (350.3, 435.8), (359.0, 425.3), (365.4, 425.4), (380.8, 412.0), (388.1, 404.6), (412.0, 374.7), (431.3, 347.7), (444.4, 330.0), (450.0, 325.4), (460.3, 315.8), (475.1, 305.2), (486.3, 286.1), (498.3, 269.1), (502.5, 247.1), (509.6, 218.3), (511.9, 196.8), (513.3, 176.9), (520.4, 142.6), (522.8, 129.6), (525.4, 109.1), (526.8, 83.3), (537.2, 66.3), (541.5, 35.8)],  # Al-Wad (HaGai St)
    [(879.9, 468.5), (868.6, 466.8), (830.7, 459.6), (792.8, 453.8), (746.3, 444.0), (701.4, 436.9), (674.9, 431.2), (665.3, 429.5), (626.7, 421.6), (592.3, 414.4), (556.6, 407.2), (536.2, 403.0), (509.1, 397.1), (481.1, 392.1), (458.6, 387.8), (432.1, 382.6), (412.0, 374.7), (431.3, 347.7), (450.0, 325.4), (391.3, 311.3), (342.4, 301.4), (304.0, 293.2)],  # Via Dolorosa
    [(162.5, 11.4), (161.5, 21.2), (160.0, 56.0), (157.4, 101.2), (154.7, 134.5), (153.7, 150.0), (152.2, 182.9), (147.2, 222.0), (146.3, 262.7)],  # Christian Quarter St
    [(34.8, 1.8), (23.3, 16.8), (-0.7, 46.1), (-32.6, 71.9), (-61.4, 96.1), (-40.6, 121.4), (-60.8, 141.0), (-51.1, 148.6), (6.3, 168.4), (-10.5, 195.8), (-33.3, 220.1), (-56.3, 231.1), (-95.6, 208.0), (-125.3, 247.8), (-135.3, 273.8), (-139.7, 294.8)],  # Latin Patriarchate & New Gate Rd
    [(6.3, 168.4), (89.5, 200.9), (114.4, 210.6), (147.2, 222.0)],  # Greek Patriarchate St
    [(-33.3, 220.1), (-24.9, 235.1), (-0.9, 239.9), (33.2, 246.9), (95.0, 257.4), (146.3, 262.7), (193.7, 270.8), (253.3, 281.1), (304.0, 293.2)],  # St. Francis St & Aqabat el-Khanqa
    [(153.7, 150.0), (180.5, 155.2), (211.4, 157.9), (250.7, 154.8), (298.6, 155.4), (307.9, 172.5)],  # Muristan (Suq Aftimos)
    [(250.7, 154.8), (251.2, 120.5), (251.6, 99.8), (254.1, 60.0), (255.4, 18.0)],  # Muristan St
    [(83.7, 4.8), (93.8, -8.2), (100.3, -33.5), (97.6, -54.9), (95.1, -76.2), (100.1, -81.1), (106.3, -104.7), (106.4, -130.4), (100.3, -180.3), (94.0, -220.1), (92.7, -256.3), (94.0, -290.1), (95.3, -342.4), (96.1, -382.0), (104.3, -401.2), (159.1, -404.7), (189.1, -399.7), (207.5, -406.5)],  # Armenian Patriarchate Rd
    [(100.3, -180.3), (141.4, -179.9), (143.2, -163.4), (175.8, -161.5), (204.6, -149.9), (233.3, -148.8), (262.8, -152.7), (295.4, -165.8)],  # St. James & Or HaHayim St
    [(264.6, -57.9), (239.8, -77.0), (220.5, -104.8), (204.6, -149.9), (205.6, -186.1), (203.2, -223.7), (197.9, -260.3), (219.0, -276.0), (209.1, -305.9), (211.0, -338.1), (211.0, -379.4), (213.0, -407.5)],  # Ararat St
    [(302.4, 20.4), (303.0, -41.9), (299.6, -98.5), (295.4, -165.8), (291.2, -230.9), (290.1, -299.0), (285.7, -320.7)],  # Chabad St
    [(329.2, 19.4), (331.8, 11.1), (329.2, -43.8), (322.6, -91.0), (322.7, -124.7), (314.9, -167.1), (311.3, -216.1), (313.8, -245.0), (316.4, -264.7)],  # Jewish Quarter St
    [(322.7, -124.7), (377.1, -115.5), (400.2, -122.1), (427.7, -124.5), (447.3, -122.5), (465.3, -136.2), (490.1, -132.1)],  # Tiferet Israel St
    [(472.6, 29.2), (482.5, -9.2), (484.3, -60.2), (483.6, -87.0), (490.1, -132.1), (487.5, -165.1), (483.1, -198.9), (478.4, -235.5), (491.6, -252.5)],  # Misgav Ladach St
    [(296.0, -320.5), (349.1, -314.7), (399.3, -312.1), (441.2, -293.6), (504.8, -262.4), (533.7, -237.1), (562.2, -188.1), (568.0, -150.2), (574.5, -139.3), (597.0, -148.0), (607.4, -177.7), (614.5, -197.2), (621.2, -218.3)],  # Batei Mahase & Dung Gate Rd
    [(520.4, 142.6), (537.0, 145.2), (567.7, 150.7), (586.3, 152.6), (613.5, 151.9)],  # Suq al-Qattanin (Cotton Market)
    [(307.9, 172.5), (376.9, 176.1), (412.9, 175.5), (468.3, 181.7), (482.8, 121.9), (522.8, 129.6)],  # Aqabat al-Khalidiya
    [(306.7, 237.1), (390.3, 247.3), (411.7, 251.0), (498.3, 269.1), (590.2, 277.9)],  # Aqabat el-Saraya
    [(511.9, 196.8), (556.7, 189.3), (605.1, 197.9)],  # Bab el-Hadid St
    [(569.3, 697.4), (595.0, 696.0), (615.1, 683.4), (626.1, 653.6), (637.3, 582.7), (660.9, 448.8), (665.3, 429.5)],  # Al-Qadisiya / Herod's Gate Rd
    [(708.5, 628.3), (713.3, 596.7), (723.7, 560.3), (729.1, 529.3), (743.7, 460.3), (752.1, 399.8)],  # Bab Hutta Rd
]

WALKABLE_CORRIDORS: List[Tuple[Vec2, Vec2]] = [
    (poly[k], poly[k + 1])
    for poly in _CURATED_STREET_POLYLINES
    for k in range(len(poly) - 1)
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


# Global registry of all building footprints `(building_id, ccw_ring)` for 2D bay clearance checks
SCENE_BUILDING_RINGS: List[Tuple[int, List[Vec2]]] = []


def clear_scene_building_rings() -> None:
    SCENE_BUILDING_RINGS.clear()


def register_scene_building_ring(building_id: int, ring_xy: Sequence[Sequence[float]]) -> None:
    for existing_id, _ in SCENE_BUILDING_RINGS:
        if existing_id == building_id:
            return
    pts = ensure_ccw_2d(ring_xy)
    if len(pts) >= 3:
        SCENE_BUILDING_RINGS.append((building_id, pts))


def _point_in_poly_2d(px: float, py: float, poly: Sequence[Vec2]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > py) != (y1 > py):
            x_int = (x1 - x0) * (py - y0) / (y1 - y0 + 1e-12) + x0
            if px < x_int:
                inside = not inside
    return inside


def _find_safe_roof_interior_point(pts: Sequence[Vec2], clearance_r: float) -> Optional[Vec2]:
    """
    Finds an interior point `(rx, ry)` inside polygon `pts` such that the entire circle of
    radius `clearance_r` around `(rx, ry)` lies strictly inside `pts` and at distance
    `>= clearance_r` from every polygon edge. Returns None if the polygon is too narrow or
    concave to accommodate `clearance_r`.
    """
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if (max_x - min_x) < 2.0 * clearance_r or (max_y - min_y) < 2.0 * clearance_r:
        return None

    n = len(pts)
    candidates: List[Vec2] = [(sum(xs) / n, sum(ys) / n)]
    # Sample 7x7 interior grid across bounding box
    for gx in range(1, 8):
        tx = min_x + clearance_r + (max_x - min_x - 2.0 * clearance_r) * (gx / 8.0)
        for gy in range(1, 8):
            ty = min_y + clearance_r + (max_y - min_y - 2.0 * clearance_r) * (gy / 8.0)
            candidates.append((tx, ty))
    # Also sample midpoints of non-adjacent vertices
    for i in range(n):
        for j in range(i + 2, min(n, i + 6)):
            candidates.append((0.5 * (pts[i][0] + pts[j][0]), 0.5 * (pts[i][1] + pts[j][1])))

    best_pt: Optional[Vec2] = None
    best_dist = -1.0
    angles = [k * (math.pi / 4.0) for k in range(8)]

    for qx, qy in candidates:
        if not _point_in_poly_2d(qx, qy, pts):
            continue
        min_edge_d = min(
            _point_to_segment_dist(qx, qy, pts[i][0], pts[i][1], pts[(i + 1) % n][0], pts[(i + 1) % n][1])
            for i in range(n)
        )
        if min_edge_d < clearance_r or min_edge_d <= best_dist:
            continue
        circle_ok = True
        for ang in angles:
            cx_p = qx + clearance_r * math.cos(ang)
            cy_p = qy + clearance_r * math.sin(ang)
            if not _point_in_poly_2d(cx_p, cy_p, pts):
                circle_ok = False
                break
        if circle_ok:
            best_dist = min_edge_d
            best_pt = (qx, qy)

    return best_pt


def is_bay_clearance_valid(
    bx: float,
    by: float,
    tx: float,
    ty: float,
    nx: float,
    ny: float,
    building_id: int,
    edge_idx: int = -1,
    half_w: float = 1.05,
) -> bool:
    """
    Tests a 2D clearance grid in front of a bay `(bx + nx*d + tx*s, by + ny*d + ty*s)` for
    `d in (0.18, 0.55)` and `s in (-half_w * 0.7, 0.0, +half_w * 0.7)` against all building
    polygons and face-to-face abutting walls in `SCENE_BUILDING_RINGS`.
    Allows narrow Jerusalem alleys (>= 1.4m wide) and adjacent row houses sharing party walls!
    """
    sample_ds = (0.18, 0.55)
    sample_ss = (-half_w * 0.70, 0.0, half_w * 0.70)

    for other_id, poly in SCENE_BUILDING_RINGS:
        # Fast AABB reject
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        if bx + 2.2 < min(xs) or bx - 2.2 > max(xs) or by + 2.2 < min(ys) or by - 2.2 > max(ys):
            continue

        # 1. Check if any sample point in front of the bay is inside another building polygon
        if other_id != building_id:
            for d in sample_ds:
                for s in sample_ss:
                    qx = bx + nx * d + tx * s
                    qy = by + ny * d + ty * s
                    if _point_in_poly_2d(qx, qy, poly):
                        return False

        # 2. Check distance to non-collinear self-walls or face-to-face party walls
        n_v = len(poly)
        for j in range(n_v):
            if other_id == building_id and j == edge_idx:
                continue
            a = poly[j]
            b = poly[(j + 1) % n_v]
            edx, edy = b[0] - a[0], b[1] - a[1]
            el = math.hypot(edx, edy)
            if el < 0.2:
                continue
            etx, ety = edx / el, edy / el
            if other_id == building_id:
                if abs(tx * etx + ty * ety) > 0.85:
                    continue
                for d in sample_ds:
                    qx = bx + nx * d
                    qy = by + ny * d
                    if _point_to_segment_dist(qx, qy, a[0], a[1], b[0], b[1]) < 0.65:
                        return False
            else:
                # Only reject if another building's wall is directly face-to-face in front of this bay (< 0.85m)
                qx = bx + nx * 0.42
                qy = by + ny * 0.42
                if _point_to_segment_dist(qx, qy, a[0], a[1], b[0], b[1]) < 0.48:
                    return False

    return True


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
    Constructs a continuous, watertight 3D radial stone arch ring (voussoirs) with a proud
    crown keystone where adjacent segments share exact mitered vertices (zero stepped notches
    or open side cracks) and both intrados soffit and extrados top faces are sealed.
    """
    cx, cy, cz = center_xyz
    tx, ty = tangent_xy
    nx, ny = normal_xy
    rx_in = span_w * 0.5
    rz_in = arch_h

    def arch_pts(a: float):
        s_a = math.sin(a)
        c_a = math.cos(a)
        point = (s_a ** 2) * 0.14 * arch_h
        k_fac = s_a ** 6
        rt = ring_thickness + keystone_boost * k_fac
        n1 = offset_n1 + keystone_boost * 0.50 * k_fac
        u_in = c_a * rx_in
        z_in = s_a * rz_in + point
        u_out = c_a * (rx_in + rt)
        z_out = s_a * (rz_in + rt) + point
        return u_in, z_in, u_out, z_out, n1

    def wp(u: float, z_off: float, d_n: float) -> Vec3:
        return (cx + tx * u + nx * d_n, cy + ty * u + ny * d_n, cz + z_off)

    for i in range(num_voussoirs):
        a0 = math.pi * (1.0 - i / num_voussoirs)
        a1 = math.pi * (1.0 - (i + 1) / num_voussoirs)
        mid_a = 0.5 * (a0 + a1)

        u_in0, z_in0, u_out0, z_out0, n1_0 = arch_pts(a0)
        u_in1, z_in1, u_out1, z_out1, n1_1 = arch_pts(a1)

        f0 = wp(u_in0, z_in0, n1_0)
        f1 = wp(u_in1, z_in1, n1_1)
        f2 = wp(u_out1, z_out1, n1_1)
        f3 = wp(u_out0, z_out0, n1_0)
        collector.add_quad(group, mat_name, f0, f1, f2, f3, uv_rect=None, uv_scale=0.35, expected_normal=(nx, ny, 0.0))

        b0 = wp(u_in0, z_in0, offset_n0)
        b1 = wp(u_in1, z_in1, offset_n0)
        in_nx = -tx * math.cos(mid_a)
        in_ny = -ty * math.cos(mid_a)
        in_nz = -math.sin(mid_a)
        collector.add_quad(group, mat_name, b0, b1, f1, f0, uv_rect=None, uv_scale=0.35, expected_normal=(in_nx, in_ny, in_nz))

        bo0 = wp(u_out0, z_out0, offset_n0)
        bo1 = wp(u_out1, z_out1, offset_n0)
        collector.add_quad(group, mat_name, f3, f2, bo1, bo0, uv_rect=None, uv_scale=0.35, expected_normal=(-in_nx, -in_ny, -in_nz))

        if back_face:
            collector.add_quad(group, mat_name, b1, b0, bo0, bo1, uv_rect=None, uv_scale=0.35, expected_normal=(-nx, -ny, 0.0))


def add_arch_spandrel_wall(
    collector: MeshCollector,
    group: str,
    mat_name: str,
    center_xyz: Vec3,
    tangent_xy: Vec2,
    normal_xy: Vec2,
    span_w: float,
    arch_h: float,
    z_top: float,
    offset_n0: float = -0.40,
    offset_n1: float = 0.0,
    num_voussoirs: int = 9,
    emit_soffit: bool = False,
) -> None:
    """
    Constructs a solid 3D stone spandrel wall extending from the exact pointed arch curve
    `(u_in, z_in)` up to `z_top` across depth `offset_n0 .. offset_n1`, completely eliminating
    triangular corner gaps above monumental arched portals.
    """
    cx, cy, cz = center_xyz
    tx, ty = tangent_xy
    nx, ny = normal_xy
    rx_in = span_w * 0.5
    rz_in = arch_h

    def wp(u: float, z_abs: float, d_n: float) -> Vec3:
        return (cx + tx * u + nx * d_n, cy + ty * u + ny * d_n, z_abs)

    for i in range(num_voussoirs):
        a0 = math.pi * (1.0 - i / num_voussoirs)
        a1 = math.pi * (1.0 - (i + 1) / num_voussoirs)
        mid_a = 0.5 * (a0 + a1)
        point0 = (math.sin(a0) ** 2) * 0.14 * arch_h
        point1 = (math.sin(a1) ** 2) * 0.14 * arch_h
        u0 = math.cos(a0) * rx_in
        z0 = cz + math.sin(a0) * rz_in + point0
        u1 = math.cos(a1) * rx_in
        z1 = cz + math.sin(a1) * rz_in + point1
        if z_top <= max(z0, z1) + 0.01:
            continue

        f0 = wp(u0, z0, offset_n1)
        f1 = wp(u1, z1, offset_n1)
        f2 = wp(u1, z_top, offset_n1)
        f3 = wp(u0, z_top, offset_n1)
        collector.add_quad(group, mat_name, f0, f1, f2, f3, uv_scale=0.28, expected_normal=(nx, ny, 0.0))

        b0 = wp(u0, z0, offset_n0)
        b1 = wp(u1, z1, offset_n0)
        b2 = wp(u1, z_top, offset_n0)
        b3 = wp(u0, z_top, offset_n0)
        collector.add_quad(group, mat_name, b1, b0, b3, b2, uv_scale=0.28, expected_normal=(-nx, -ny, 0.0))

        if emit_soffit:
            in_nx = -tx * math.cos(mid_a)
            in_ny = -ty * math.cos(mid_a)
            in_nz = -math.sin(mid_a)
            collector.add_quad(group, mat_name, b0, b1, f1, f0, uv_scale=0.28, expected_normal=(in_nx, in_ny, in_nz))


def add_arched_opening_panel(
    collector: MeshCollector,
    group: str,
    mat_name: str,
    center_xy: Vec2,
    z_bot: float,
    arch_spring_z: float,
    arch_h: float,
    span_w: float,
    panel_d: float,
    tangent_xy: Vec2,
    normal_xy: Vec2,
    uv_rect: UVRect,
    num_voussoirs: int = 5,
    add_wall_spandrel: bool = False,
    z_top_wall: float = 0.0,
    wall_mat: str = "mat_stone_boss",
) -> None:
    """
    Constructs an arched doorway/shutter/window panel whose top edge follows the exact
    `num_voussoirs`-segment curve of `add_arch_voussoirs` so ZERO rectangular top corners
    ever protrude outside the stone arch ring!
    Maps `(U, V)` linearly from `(u_in, z)` across `uv_rect` so horizontal wood/shutter
    lines are 100% undistorted and UV aspect stretch is ~1.0!
    """
    from mathutils import Vector

    cx, cy = center_xy
    tx, ty = tangent_xy
    nx, ny = normal_xy
    hw = span_w * 0.5
    u0, v0, u1, v1 = uv_rect
    z_crown = arch_spring_z + arch_h * 1.14
    total_h = max(0.1, z_crown - z_bot)

    bm, uv_layer = collector.get_bm(group, mat_name)
    exp_n = Vector((nx, ny, 0.0))

    for i in range(num_voussoirs):
        a0 = math.pi * (1.0 - i / num_voussoirs)
        a1 = math.pi * (1.0 - (i + 1) / num_voussoirs)
        point0 = (math.sin(a0) ** 2) * 0.14 * arch_h
        point1 = (math.sin(a1) ** 2) * 0.14 * arch_h

        u_in0 = math.cos(a0) * hw
        z_in0 = arch_spring_z + math.sin(a0) * arch_h + point0
        u_in1 = math.cos(a1) * hw
        z_in1 = arch_spring_z + math.sin(a1) * arch_h + point1

        p0 = Vector((cx + tx * u_in0 + nx * panel_d, cy + ty * u_in0 + ny * panel_d, z_bot))
        p1 = Vector((cx + tx * u_in1 + nx * panel_d, cy + ty * u_in1 + ny * panel_d, z_bot))
        p2 = Vector((cx + tx * u_in1 + nx * panel_d, cy + ty * u_in1 + ny * panel_d, z_in1))
        p3 = Vector((cx + tx * u_in0 + nx * panel_d, cy + ty * u_in0 + ny * panel_d, z_in0))

        tu0 = max(0.0, min(1.0, (u_in0 + hw) / max(0.01, span_w)))
        tu1 = max(0.0, min(1.0, (u_in1 + hw) / max(0.01, span_w)))
        tv0 = max(0.0, min(1.0, (z_in0 - z_bot) / total_h))
        tv1 = max(0.0, min(1.0, (z_in1 - z_bot) / total_h))

        uv_p0 = (u0 + (u1 - u0) * tu0, v0)
        uv_p1 = (u0 + (u1 - u0) * tu1, v0)
        uv_p2 = (u0 + (u1 - u0) * tu1, v0 + (v1 - v0) * tv1)
        uv_p3 = (u0 + (u1 - u0) * tu0, v0 + (v1 - v0) * tv0)

        pts = [p0, p1, p2, p3]
        uvs = [uv_p0, uv_p1, uv_p2, uv_p3]
        if (p1 - p0).cross(p3 - p0).dot(exp_n) < 0.0:
            pts = [p1, p0, p3, p2]
            uvs = [uv_p1, uv_p0, uv_p3, uv_p2]

        try:
            f = bm.faces.new([bm.verts.new(pt) for pt in pts])
            f.normal_update()
            f.smooth = False
            for loop, uv in zip(f.loops, uvs):
                loop[uv_layer].uv = uv
        except ValueError:
            pass

        if add_wall_spandrel and z_top_wall > max(z_in0, z_in1) + 0.01:
            w_b0 = (cx + tx * u_in0, cy + ty * u_in0, z_in0)
            w_b1 = (cx + tx * u_in1, cy + ty * u_in1, z_in1)
            w_t1 = (cx + tx * u_in1, cy + ty * u_in1, z_top_wall)
            w_t0 = (cx + tx * u_in0, cy + ty * u_in0, z_top_wall)
            collector.add_quad(
                "Buildings_Walls",
                wall_mat,
                w_b0,
                w_b1,
                w_t1,
                w_t0,
                expected_normal=(nx, ny, 0.0),
            )


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
    if has_balcony:
        sill_z = z0 + 0.14
        head_z = z0 + bay_h * 0.78
        win_h = head_z - sill_z
    else:
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
        head_z + 0.02,
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
        head_z + 0.02,
        0.0,
        frame_proud,
        normal_xy,
        cap_ends=False,
    )

    # 3. Carved stone tympanum infill + 3D Arched stone voussoir hood mould + keystone (`num_voussoirs=5`)
    # Springs cleanly at `head_z + 0.02` with `offset_n0 = 0.0` so zero stone corners dip into the rectangular window glass!
    win_cx = 0.5 * (w_p0[0] + w_p1[0])
    win_cy = 0.5 * (w_p0[1] + w_p1[1])
    add_arched_opening_panel(
        collector,
        "Buildings_Details",
        trim_mat,
        (win_cx, win_cy),
        head_z,
        head_z + 0.02,
        0.24,
        win_w,
        0.035,
        (tx, ty),
        normal_xy,
        uv_rect=(0.0, 0.0, 0.5, 0.35),
        num_voussoirs=5,
        add_wall_spandrel=False,
    )
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        trim_mat,
        (win_cx, win_cy, head_z + 0.02),
        (tx, ty),
        normal_xy,
        span_w=win_w,
        arch_h=0.24,
        ring_thickness=0.16,
        offset_n0=0.0,
        offset_n1=frame_proud + 0.02,
        num_voussoirs=5,
        keystone_boost=0.04,
    )

    # 4. Projecting stone window sill + optional twin stone corbel brackets (only on non-balcony windows; balcony windows use the full balcony slab!)
    if not has_balcony:
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

    # 6. 3D Open Louvred Wooden Shutters (folded nearly flat against the outer stone piers on non-balcony windows)
    if has_shutters and not has_balcony:
        shutter_leaf_w = min(win_w * 0.44, (u_left - frame_w - 0.08))
        if shutter_leaf_w > 0.22:
            open_ang = 0.10
            for side_sign, hinge_u in ((-1.0, u_left - frame_w), (1.0, u_right + frame_w)):
                hinge_xy = pt_at(hinge_u)
                hx = hinge_xy[0] + nx * (frame_proud + 0.01)
                hy = hinge_xy[1] + ny * (frame_proud + 0.01)
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

    if has_grille and not has_balcony:
        gr_d = -0.04
        gr0 = (w_p0[0] + nx * gr_d, w_p0[1] + ny * gr_d, sill_z)
        gr1 = (w_p1[0] + nx * gr_d, w_p1[1] + ny * gr_d, sill_z)
        gr2 = (w_p1[0] + nx * gr_d, w_p1[1] + ny * gr_d, head_z)
        gr3 = (w_p0[0] + nx * gr_d, w_p0[1] + ny * gr_d, head_z)
        collector.add_quad("Buildings_Details", "mat_iron_grille", gr0, gr1, gr2, gr3, uv_rect=(0.0, 0.0, 1.0, 1.5), double_sided=True)

    if has_balcony:
        bal_p0 = pt_at(u_left - 0.28)
        bal_p1 = pt_at(u_right + 0.28)
        collector.add_oriented_box("Buildings_Details", "mat_stone_mizzi", bal_p0, bal_p1, sill_z - 0.135, sill_z + 0.015, 0.02, 0.76, normal_xy, cap_top=True, cap_bottom=True)
        for cb_u in (u_left - 0.12, u_right - 0.06):
            collector.add_oriented_box("Buildings_Details", "mat_stone_boss", pt_at(cb_u), pt_at(cb_u + 0.18), sill_z - 0.38, sill_z - 0.14, 0.03, 0.56, normal_xy, cap_top=False, cap_bottom=True)
        b_w0 = (bal_p0[0] + nx * 0.04, bal_p0[1] + ny * 0.04, sill_z + 0.015)
        b_w1 = (bal_p1[0] + nx * 0.04, bal_p1[1] + ny * 0.04, sill_z + 0.015)
        b_c0 = (bal_p0[0] + nx * 0.68, bal_p0[1] + ny * 0.68, sill_z + 0.015)
        b_c1 = (bal_p1[0] + nx * 0.68, bal_p1[1] + ny * 0.68, sill_z + 0.015)
        build_iron_railing(collector, b_c0, b_c1, height=0.86, picket_spacing=0.28)
        build_iron_railing(collector, b_w0, b_c0, height=0.86, picket_spacing=0.28)
        build_iron_railing(collector, b_c1, b_w1, height=0.86, picket_spacing=0.28)


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
            (p0_xy[0], p0_xy[1], z0 - 9.5), (p1_xy[0], p1_xy[1], z0 - 9.5),
            (p1_xy[0], p1_xy[1], z1), (p0_xy[0], p0_xy[1], z1),
            expected_normal=(nx, ny, 0.0),
        )
        return

    tx, ty = dx / bay_w, dy / bay_w
    rx, ry = -ny, nx

    is_wide_opening = opening_mode in ("souq_open", "steel_shutter", "cafe_glass")
    open_w = min(2.35 if is_wide_opening else 1.50, bay_w * (0.66 if is_wide_opening else 0.48))
    u_left = (bay_w - open_w) * 0.5
    u_right = u_left + open_w

    def pt_at(u: float) -> Vec2:
        return (p0_xy[0] + tx * u, p0_xy[1] + ty * u)

    s_p0 = pt_at(u_left)
    s_p1 = pt_at(u_right)
    mid_x = 0.5 * (s_p0[0] + s_p1[0])
    mid_y = 0.5 * (s_p0[1] + s_p1[1])

    # Vertical proportions guaranteeing a dedicated >= 0.85m stone frieze band above the keystone:
    # `arch_spring_z = z0 + bay_h * 0.52` (~2.08m), `arch_h <= 0.54m`, `ring_thickness = 0.24m`, `keystone_boost = 0.04m`
    # -> Top of keystone is strictly <= `z0 + 2.97m`, leaving `z0 + 2.97m .. z0 + 4.00m` (`1.03m` tall) clear for the signboard!
    arch_spring_z = z0 + bay_h * 0.52
    arch_h = min(0.54, open_w * 0.24)
    head_z = arch_spring_z + arch_h

    # 1. Left & right outer stone piers extending down to `z0 - 9.5m` (so sloped streets never show gaps!)
    collector.add_quad(
        "Buildings_Walls", pier_mat,
        (p0_xy[0], p0_xy[1], z0 - 9.5), (s_p0[0], s_p0[1], z0 - 9.5), (s_p0[0], s_p0[1], z1), (p0_xy[0], p0_xy[1], z1),
        expected_normal=(nx, ny, 0.0),
    )
    collector.add_quad(
        "Buildings_Walls", pier_mat,
        (s_p1[0], s_p1[1], z0 - 9.5), (p1_xy[0], p1_xy[1], z0 - 9.5), (p1_xy[0], p1_xy[1], z1), (s_p1[0], s_p1[1], z1),
        expected_normal=(nx, ny, 0.0),
    )

    # Explicit 0.35m recessed stone jamb reveals from `z0 - 9.5m` up to `arch_spring_z`
    jl0 = (s_p0[0] - nx * recess, s_p0[1] - ny * recess, z0 - 9.5)
    jl1 = (s_p0[0], s_p0[1], z0 - 9.5)
    jl2 = (s_p0[0], s_p0[1], arch_spring_z)
    jl3 = (s_p0[0] - nx * recess, s_p0[1] - ny * recess, arch_spring_z)
    collector.add_quad("Buildings_Walls", pier_mat, jl0, jl1, jl2, jl3, expected_normal=(tx, ty, 0.0))

    jr0 = (s_p1[0], s_p1[1], z0 - 9.5)
    jr1 = (s_p1[0] - nx * recess, s_p1[1] - ny * recess, z0 - 9.5)
    jr2 = (s_p1[0] - nx * recess, s_p1[1] - ny * recess, arch_spring_z)
    jr3 = (s_p1[0], s_p1[1], arch_spring_z)
    collector.add_quad("Buildings_Walls", pier_mat, jr0, jr1, jr2, jr3, expected_normal=(-tx, -ty, 0.0))

    # 2. Flush stone threshold step inside the recess (`z0 - 9.5 .. z0 + 0.04`, `-recess .. 0.0`)
    collector.add_oriented_box("Buildings_Details", "mat_stone_ashlar_aged", s_p0, s_p1, z0 - 9.5, z0 + 0.04, -recess, 0.0, normal_xy)

    # 3. 3D Radial Stone Voussoir Arch Surround + Projecting Keystone (`num_voussoirs=5`, full recess soffit)
    add_arch_voussoirs(
        collector,
        "Buildings_Details",
        "mat_stone_ashlar",
        (mid_x, mid_y, arch_spring_z),
        (tx, ty),
        normal_xy,
        span_w=open_w,
        arch_h=arch_h,
        ring_thickness=0.24,
        offset_n0=-recess,
        offset_n1=0.08,
        num_voussoirs=5,
        keystone_boost=0.04,
    )

    # 4. Recessed Arched Bay Interior Panel + Outer Stone Spandrel Wall by `opening_mode`
    panel_d = -recess + 0.035

    if opening_mode in ("steel_shutter", "steel_door"):
        uv_q = STEEL_SHUTTER_UV.get(shutter_variant, STEEL_SHUTTER_UV["shutter_graffiti_1"])
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_steel_shutters",
            (mid_x, mid_y),
            z0 + 0.04,
            arch_spring_z,
            arch_h,
            open_w,
            panel_d,
            (tx, ty),
            normal_xy,
            uv_rect=uv_q,
            num_voussoirs=5,
            add_wall_spandrel=True,
            z_top_wall=z1,
            wall_mat=pier_mat,
        )

    elif opening_mode == "cafe_glass":
        riser_z = z0 + 0.68
        collector.add_oriented_box("Buildings_Details", "mat_stone_mizzi", s_p0, s_p1, z0 + 0.04, riser_z, -recess, -recess + 0.14, normal_xy)
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (mid_x, mid_y),
            riser_z,
            arch_spring_z,
            arch_h,
            open_w,
            panel_d,
            (tx, ty),
            normal_xy,
            uv_rect=WOOD_DOOR_UV["window_glass"],
            num_voussoirs=5,
            add_wall_spandrel=True,
            z_top_wall=z1,
            wall_mat=pier_mat,
        )

    else:
        uv_q = WOOD_DOOR_UV["souq_open"] if opening_mode == "souq_open" else WOOD_DOOR_UV["door_wood"]
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (mid_x, mid_y),
            z0 + 0.04,
            arch_spring_z,
            arch_h,
            open_w,
            panel_d,
            (tx, ty),
            normal_xy,
            uv_rect=uv_q,
            num_voussoirs=5,
            add_wall_spandrel=True,
            z_top_wall=z1,
            wall_mat=pier_mat,
        )

    # 5. Un-mirrored Framed Shop Fascia Sign in dedicated frieze band (`sign_z = z0 + bay_h - 0.45m`, `height = 0.44m`)
    # Guarantees >= 35cm vertical clearance above the arch keystone AND 23cm below the upper floor balcony/string course!
    # Never places `build_blade_sign` or a second fascia sign on the same bay!
    if sign_uv is not None:
        fascia_w = min(open_w * 0.88, 1.85)
        fascia_h = 0.44
        sign_z = z0 + bay_h - 0.45
        build_fascia_sign(
            collector,
            (mid_x, mid_y, sign_z),
            (tx, ty),
            normal_xy,
            width=fascia_w,
            height=fascia_h,
            sign_uv_rect=sign_uv,
        )

    # 6. 3D Sloped/Curved Awning on Wrought-Iron Brackets (only on bays without a fascia sign so awnings never obscure signs!)
    if has_awning and sign_uv is None and opening_mode in ("souq_open", "cafe_glass"):
        build_shop_awning(
            collector,
            (mid_x, mid_y, head_z + 0.36),
            (tx, ty),
            normal_xy,
            width=open_w + 0.22,
            projection=1.02,
            drop=0.38,
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
    - Ground-floor stone wall quad + arched shopfront/door polygon + 3-segment stone arch strip
    - Mid-facade horizontal stone string course between ground and upper floors
    - Upper-floor stone wall quad + stone-framed windows with projecting sills and open wooden louvred shutters
    - Projecting cornice strip + roof parapet
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    edge_len = math.hypot(dx, dy)
    tx, ty = dx / edge_len, dy / edge_len
    nx, ny = normal_xy
    rx, ry = -ny, nx
    roof_z = base_z + ground_h + (levels - 1) * upper_floor_h

    # Ground floor continuous stone wall quad (from base_z - 9.5 to base_z + ground_h)
    collector.add_quad(
        "Buildings_Walls", ground_pier_mat,
        (p0[0], p0[1], base_z - 9.5), (p1[0], p1[1], base_z - 9.5),
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

    # Mid-facade horizontal stone string course between floors
    if edge_len >= 2.0 and levels >= 2:
        collector.add_oriented_box(
            "Buildings_Details",
            trim_stone_mat,
            p0,
            p1,
            base_z + ground_h - 0.06,
            base_z + ground_h + 0.08,
            0.0,
            0.09,
            normal_xy,
            cap_ends=False,
        )

    num_bays = max(1, int(round(edge_len / 3.8)))
    bay_w = edge_len / num_bays
    shutter_keys = ["shutter_graffiti_1", "shutter_graffiti_2", "shutter_rusty_green", "steel_door_tagged"]
    corner_margin = min(0.80, max(0.28, (edge_len - 2.0) * 0.35))

    for b_idx in range(num_bays):
        if edge_len < 2.25 or bay_w < 1.85:
            continue
        mid_u = (b_idx + 0.5) * bay_w
        open_hw = min(0.84, bay_w * 0.27)
        if (mid_u - open_hw - 0.10) < corner_margin or (mid_u + open_hw + 0.10) > (edge_len - corner_margin):
            continue

        mx = p0[0] + tx * mid_u
        my = p0[1] + ty * mid_u
        bay_ok = is_bay_clearance_valid(mx, my, tx, ty, nx, ny, building_id, edge_idx=edge_idx, half_w=open_hw + 0.12)

        if bay_ok:
            # Ground floor arched opening panel (follows exact 3-segment arch curve so zero corners stick out!)
            open_h = 2.45
            arch_spring_z = base_z + open_h - 0.42
            arch_h = 0.42
            d_g = 0.032
            h_mode = (building_id + edge_idx + b_idx) % 4
            if h_mode == 0:
                sh_k = shutter_keys[(building_id + edge_idx + b_idx) % len(shutter_keys)]
                open_mat = "mat_steel_shutters"
                open_uv = STEEL_SHUTTER_UV[sh_k]
            elif h_mode == 1:
                open_mat = "mat_wood_doors"
                open_uv = WOOD_DOOR_UV["souq_open"]
            else:
                open_mat = "mat_wood_doors"
                open_uv = WOOD_DOOR_UV["door_wood"]

            add_arched_opening_panel(
                collector,
                "Buildings_Openings",
                open_mat,
                (mx, my),
                base_z + 0.05,
                arch_spring_z,
                arch_h,
                open_hw * 2.0,
                d_g,
                (tx, ty),
                normal_xy,
                uv_rect=open_uv,
                num_voussoirs=3,
                add_wall_spandrel=False,
            )

            if corridor_distance(mx, my) <= 28.0:
                add_arch_voussoirs(
                    collector,
                    "Buildings_Details",
                    trim_stone_mat,
                    (mx, my, arch_spring_z),
                    (tx, ty),
                    normal_xy,
                    span_w=open_hw * 2.0,
                    arch_h=arch_h,
                    ring_thickness=0.22,
                    offset_n0=0.015,
                    offset_n1=0.075,
                    num_voussoirs=3,
                    keystone_boost=0.04,
                )
            if h_mode == 1 and b_idx % 2 == 0 and corridor_distance(mx, my) <= 18.0:
                awning_mats = ("mat_awning_striped", "mat_awning_red", "mat_awning_green")
                build_shop_awning(
                    collector,
                    (mx, my, arch_spring_z + arch_h + 0.28),
                    (tx, ty),
                    normal_xy,
                    width=open_hw * 2.0 + 0.18,
                    projection=0.88,
                    drop=0.34,
                    mat_name=awning_mats[(building_id + edge_idx + b_idx) % len(awning_mats)],
                )

            # Upper floor window quads (stone surround + projecting stone sill + glass pane + open wooden louvred shutters)
            win_hw = 0.54
            win_h = 1.52
            fw_hw = win_hw + 0.15
            for lvl in range(1, levels):
                fl_z0 = base_z + ground_h + (lvl - 1) * upper_floor_h
                sill_z = fl_z0 + 0.75
                head_z = sill_z + win_h
                d_f = 0.032
                d_w = 0.052
                d_s = 0.066
                f0 = (mx - rx * fw_hw + nx * d_f, my - ry * fw_hw + ny * d_f, sill_z - 0.10)
                f1 = (mx + rx * fw_hw + nx * d_f, my + ry * fw_hw + ny * d_f, sill_z - 0.10)
                f2 = (mx + rx * fw_hw + nx * d_f, my + ry * fw_hw + ny * d_f, head_z + 0.16)
                f3 = (mx - rx * fw_hw + nx * d_f, my - ry * fw_hw + ny * d_f, head_z + 0.16)
                collector.add_quad("Buildings_Details", "mat_stone_mizzi", f0, f1, f2, f3, expected_normal=(nx, ny, 0.0))

                collector.add_oriented_box(
                    "Buildings_Details",
                    "mat_stone_boss",
                    (mx - rx * (fw_hw + 0.05), my - ry * (fw_hw + 0.05)),
                    (mx + rx * (fw_hw + 0.05), my + ry * (fw_hw + 0.05)),
                    sill_z - 0.12,
                    sill_z + 0.02,
                    0.0,
                    0.15,
                    normal_xy,
                    cap_ends=False,
                )

                w0 = (mx - rx * win_hw + nx * d_w, my - ry * win_hw + ny * d_w, sill_z)
                w1 = (mx + rx * win_hw + nx * d_w, my + ry * win_hw + ny * d_w, sill_z)
                w2 = (mx + rx * win_hw + nx * d_w, my + ry * win_hw + ny * d_w, head_z)
                w3 = (mx - rx * win_hw + nx * d_w, my - ry * win_hw + ny * d_w, head_z)
                collector.add_quad("Buildings_Openings", "mat_wood_doors", w0, w1, w2, w3, uv_rect=WOOD_DOOR_UV["window_glass"], expected_normal=(nx, ny, 0.0))

                for sh_sign in (-1.0, 1.0):
                    sh_c = sh_sign * (win_hw + 0.19)
                    sh_hw = 0.17
                    s0 = (mx + rx * (sh_c - sh_hw) + nx * d_s, my + ry * (sh_c - sh_hw) + ny * d_s, sill_z + 0.02)
                    s1 = (mx + rx * (sh_c + sh_hw) + nx * d_s, my + ry * (sh_c + sh_hw) + ny * d_s, sill_z + 0.02)
                    s2 = (mx + rx * (sh_c + sh_hw) + nx * d_s, my + ry * (sh_c + sh_hw) + ny * d_s, head_z - 0.02)
                    s3 = (mx + rx * (sh_c - sh_hw) + nx * d_s, my + ry * (sh_c - sh_hw) + ny * d_s, head_z - 0.02)
                    collector.add_quad("Buildings_Openings", "mat_wood_doors", s0, s1, s2, s3, uv_rect=WOOD_DOOR_UV["shutter_louvre"], expected_normal=(nx, ny, 0.0))

    # Projecting roof cornice strip
    collector.add_oriented_box(
        "Buildings_Details",
        trim_stone_mat,
        p0,
        p1,
        roof_z + 0.35,
        roof_z + 0.58,
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
    Builds a complete multi-bay Jerusalem Old City stone building (`corridor_dist > 55m`):
    - Split at `base_z + ground_h` with projecting stone belt course and roof cornice
    - Full multi-bay rhythm (`num_bays = max(1, int(round(edge_len / 4.2)))`) across every wall >= 3.6m
    - Arched ground-floor doorways/shopfronts with 3D stone voussoir arch hoods
    - Upper-floor stone-framed windows with projecting stone sills and open wooden louvred shutters
    - Traditional Jerusalem stone domes on roofs
    """
    roof_z = base_z + ground_h + (levels - 1) * upper_floor_h
    n_edges = len(pts)
    shutter_keys = ["shutter_graffiti_1", "shutter_graffiti_2", "shutter_rusty_green", "steel_door_tagged"]

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

        z_split = base_z + ground_h
        collector.add_quad(
            "Buildings_Walls", ground_pier_mat,
            (p0[0], p0[1], base_z - 9.5), (p1[0], p1[1], base_z - 9.5),
            (p1[0], p1[1], z_split), (p0[0], p0[1], z_split),
            expected_normal=(nx, ny, 0.0),
        )
        collector.add_quad(
            "Buildings_Walls", upper_wall_mat,
            (p0[0], p0[1], z_split), (p1[0], p1[1], z_split),
            (p1[0], p1[1], roof_z + 0.60), (p0[0], p0[1], roof_z + 0.60),
            expected_normal=(nx, ny, 0.0),
        )

        # Projecting stone roof cornice & mid-facade belt course on walls >= 2.0m
        if edge_len >= 2.0:
            if levels >= 2:
                collector.add_oriented_box(
                    "Buildings_Details",
                    "mat_stone_mizzi",
                    p0,
                    p1,
                    z_split - 0.06,
                    z_split + 0.08,
                    0.0,
                    0.09,
                    (nx, ny),
                    cap_ends=False,
                )
            collector.add_oriented_box(
                "Buildings_Details",
                "mat_stone_mizzi",
                p0,
                p1,
                roof_z + 0.36,
                roof_z + 0.60,
                0.0,
                0.16,
                (nx, ny),
                cap_ends=False,
            )

        # Full multi-bay ground-floor arched portals & upper-floor shuttered windows on walls >= 3.6m
        if edge_len >= 3.6:
            n_wins = max(1, int(round(edge_len / 4.2)))
            bay_w = edge_len / n_wins
            for w_i in range(n_wins):
                mid_u = (w_i + 0.5) * bay_w
                if mid_u < 0.90 or mid_u > (edge_len - 0.90):
                    continue
                bx = p0[0] + tx * mid_u
                by = p0[1] + ty * mid_u
                if not is_bay_clearance_valid(bx, by, tx, ty, nx, ny, building_id, edge_idx=i, half_w=0.8):
                    continue

                # Ground-floor arched doorway / shopfront with 3D stone voussoirs
                if w_i % 2 == 0 or n_wins == 1:
                    open_hw = min(0.80, bay_w * 0.25)
                    open_h = 2.40
                    arch_spring_z = base_z + open_h - 0.40
                    arch_h = 0.40
                    if (building_id + i + w_i) % 3 == 0:
                        sh_k = shutter_keys[(building_id + i + w_i) % len(shutter_keys)]
                        open_mat = "mat_steel_shutters"
                        open_uv = STEEL_SHUTTER_UV[sh_k]
                    else:
                        open_mat = "mat_wood_doors"
                        open_uv = WOOD_DOOR_UV["door_wood"]

                    add_arched_opening_panel(
                        collector,
                        "Buildings_Openings",
                        open_mat,
                        (bx, by),
                        base_z + 0.04,
                        arch_spring_z,
                        arch_h,
                        open_hw * 2.0,
                        0.032,
                        (tx, ty),
                        (nx, ny),
                        uv_rect=open_uv,
                        num_voussoirs=3,
                        add_wall_spandrel=False,
                    )

                win_hw = 0.54
                fw_hw = win_hw + 0.15
                fx, fy = bx + nx * 0.032, by + ny * 0.032
                mx, my = bx + nx * 0.052, by + ny * 0.052
                sx_p, sy_p = bx + nx * 0.066, by + ny * 0.066
                for lvl in range(1, levels):
                    sill_z = base_z + ground_h + (lvl - 1) * upper_floor_h + 0.75
                    head_z = sill_z + 1.48
                    collector.add_quad(
                        "Buildings_Details",
                        "mat_stone_mizzi",
                        (fx - rx * fw_hw, fy - ry * fw_hw, sill_z - 0.10),
                        (fx + rx * fw_hw, fy + ry * fw_hw, sill_z - 0.10),
                        (fx + rx * fw_hw, fy + ry * fw_hw, head_z + 0.16),
                        (fx - rx * fw_hw, fy - ry * fw_hw, head_z + 0.16),
                        expected_normal=(nx, ny, 0.0),
                    )
                    collector.add_oriented_box(
                        "Buildings_Details",
                        "mat_stone_boss",
                        (bx - rx * (fw_hw + 0.05), by - ry * (fw_hw + 0.05)),
                        (bx + rx * (fw_hw + 0.05), by + ry * (fw_hw + 0.05)),
                        sill_z - 0.12,
                        sill_z + 0.02,
                        0.0,
                        0.15,
                        (nx, ny),
                        cap_ends=False,
                    )
                    w0 = (mx - rx * win_hw, my - ry * win_hw, sill_z)
                    w1 = (mx + rx * win_hw, my + ry * win_hw, sill_z)
                    w2 = (mx + rx * win_hw, my + ry * win_hw, head_z)
                    w3 = (mx - rx * win_hw, my - ry * win_hw, head_z)
                    collector.add_quad("Buildings_Openings", "mat_wood_doors", w0, w1, w2, w3, uv_rect=WOOD_DOOR_UV["window_glass"], expected_normal=(nx, ny, 0.0))
                    for sh_sign in (-1.0, 1.0):
                        sh_c = sh_sign * (win_hw + 0.19)
                        sh_hw = 0.17
                        s0 = (sx_p + rx * (sh_c - sh_hw), sy_p + ry * (sh_c - sh_hw), sill_z + 0.02)
                        s1 = (sx_p + rx * (sh_c + sh_hw), sy_p + ry * (sh_c + sh_hw), sill_z + 0.02)
                        s2 = (sx_p + rx * (sh_c + sh_hw), sy_p + ry * (sh_c + sh_hw), head_z - 0.02)
                        s3 = (sx_p + rx * (sh_c - sh_hw), sy_p + ry * (sh_c - sh_hw), head_z - 0.02)
                        collector.add_quad("Buildings_Openings", "mat_wood_doors", s0, s1, s2, s3, uv_rect=WOOD_DOOR_UV["shutter_louvre"], expected_normal=(nx, ny, 0.0))

    roof_pts = [(p[0], p[1], roof_z + 0.60) for p in pts]
    collector.add_polygon_3d("Buildings_Walls", "mat_plaster", roof_pts, uv_scale=0.25)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if has_dome or (building_id % 3 == 0 and (max(xs) - min(xs)) >= 5.5 and (max(ys) - min(ys)) >= 5.5):
        safe_pt = _find_safe_roof_interior_point(pts, 2.35)
        if safe_pt is not None:
            collector.add_dome("Buildings_Details", "mat_stone_ashlar_aged", (safe_pt[0], safe_pt[1], roof_z + 0.55), radius=1.95, height=1.40, rings=4, segments=8)


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
    register_scene_building_ring(building_id, pts)

    levels = max(1, min(4, levels))
    if is_hero:
        ground_h = 5.1 if levels == 1 else 4.0
        upper_floor_h = 3.3
    else:
        ground_h = 4.8 if levels == 1 else (3.85 + (building_id % 5) * 0.11)
        upper_floor_h = 3.15 + ((building_id // 3) % 5) * 0.10
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
    # City-wide LOD 0 along all major thoroughfares across all quarters (no landmark-only restriction!)
    if is_hero or corridor_dist <= 13.5:
        lod_tier = 0
    elif corridor_dist <= 65.0:
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
        faces_away = (d_out > d_in + 0.6) and not is_hero

        # Even on a LOD 0 building, rear/courtyard walls that face away from the street use LOD 1!
        if lod_tier == 1 or (not is_hero and (wall_corridor_dist > 11.0 or faces_away)):
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
        # Hero buildings in r01 keep >= 1.50m corner margin & 4.8m min bay edge; regular Old City buildings adapt down to 2.75m facades
        corner_margin = 1.50 if is_hero else min(1.15, max(0.35, (edge_len - 2.15) * 0.40))
        min_bay_edge = 4.80 if is_hero else 2.75
        if edge_len < min_bay_edge:
            collector.add_quad(
                "Buildings_Walls", ground_pier_mat,
                (p0[0], p0[1], base_z - 9.5), (p1[0], p1[1], base_z - 9.5),
                (p1[0], p1[1], base_z + ground_h), (p0[0], p0[1], base_z + ground_h),
                expected_normal=(nx, ny, 0.0),
            )
            for lvl in range(1, levels):
                fl_z0 = base_z + ground_h + (lvl - 1) * upper_floor_h
                fl_z1 = fl_z0 + upper_floor_h
                collector.add_oriented_box(
                    "Buildings_Details", trim_stone_mat, p0, p1, fl_z0 - 0.07, fl_z0 + 0.08, 0.0, 0.085, (nx, ny), cap_ends=False
                )
                collector.add_quad(
                    "Buildings_Walls", upper_wall_mat,
                    (p0[0], p0[1], fl_z0 + 0.08), (p1[0], p1[1], fl_z0 + 0.08),
                    (p1[0], p1[1], fl_z1), (p0[0], p0[1], fl_z1),
                    expected_normal=(nx, ny, 0.0),
                )
        else:
            c_left = (p0[0] + tx * corner_margin, p0[1] + ty * corner_margin)
            c_right = (p1[0] - tx * corner_margin, p1[1] - ty * corner_margin)
            inner_len = edge_len - 2.0 * corner_margin
            target_bay_w = 3.4
            num_bays = max(1, int(round(inner_len / target_bay_w)))
            bay_w = inner_len / num_bays

            # Solid 1.50m corner piers on Ground Floor extending down to `base_z - 9.5m`
            collector.add_quad(
                "Buildings_Walls", ground_pier_mat,
                (p0[0], p0[1], base_z - 9.5), (c_left[0], c_left[1], base_z - 9.5),
                (c_left[0], c_left[1], base_z + ground_h), (p0[0], p0[1], base_z + ground_h),
                expected_normal=(nx, ny, 0.0),
            )
            collector.add_quad(
                "Buildings_Walls", ground_pier_mat,
                (c_right[0], c_right[1], base_z - 9.5), (p1[0], p1[1], base_z - 9.5),
                (p1[0], p1[1], base_z + ground_h), (c_right[0], c_right[1], base_z + ground_h),
                expected_normal=(nx, ny, 0.0),
            )

            # Pre-evaluate 2D clearance in front of each bay against all building polygons & walls
            bay_clear_flags: List[bool] = []
            for b_idx in range(num_bays):
                bm_u = corner_margin + (b_idx + 0.5) * bay_w
                bm_x = p0[0] + tx * bm_u
                bm_y = p0[1] + ty * bm_u
                bay_clear_flags.append(
                    is_bay_clearance_valid(
                        bm_x, bm_y, tx, ty, nx, ny, building_id, edge_idx=i, half_w=min(1.05, bay_w * 0.36)
                    )
                )

            # 1. Ground Floor (`base_z .. base_z + ground_h`)
            for b_idx in range(num_bays):
                b_p0 = (c_left[0] + tx * (b_idx * bay_w), c_left[1] + ty * (b_idx * bay_w))
                b_p1 = (c_left[0] + tx * ((b_idx + 1) * bay_w), c_left[1] + ty * ((b_idx + 1) * bay_w))

                if bay_w >= 2.1 and bay_clear_flags[b_idx]:
                    h_val = (building_id * 7 + i * 13 + b_idx * 5) % 10
                    if building_id == 7004:
                        mode = "cafe_glass" if b_idx % 2 == 0 else "wood_door"
                    elif all_upper_balconies:
                        if b_idx % 3 == 1:
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
                    if building_id in (7002, 7003) and sh_var == "shutter_rusty_green":
                        sh_var = "shutter_graffiti_1"
                    is_center_bay = (b_idx == num_bays // 2)
                    if forced_sign_key and is_center_bay:
                        show_sign = True
                        s_key = forced_sign_key
                    else:
                        show_sign = (mode in ("souq_open", "cafe_glass", "steel_shutter")) and (
                            (b_idx % 3 == 0 and abs(b_idx - num_bays // 2) >= 2)
                            if (forced_sign_key or all_upper_balconies)
                            else (b_idx % 2 == 0)
                        )
                        s_key = SHOP_SIGN_KEYS[(building_id + i + b_idx) % len(SHOP_SIGN_KEYS)]
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
                        has_blade_sign=False,
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
                        (b_p0[0], b_p0[1], base_z - 9.5), (b_p1[0], b_p1[1], base_z - 9.5),
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

                # Solid 1.50m corner piers on Upper Floors
                collector.add_quad(
                    "Buildings_Walls", upper_wall_mat,
                    (p0[0], p0[1], fl_z0 + 0.08), (c_left[0], c_left[1], fl_z0 + 0.08),
                    (c_left[0], c_left[1], fl_z1), (p0[0], p0[1], fl_z1),
                    expected_normal=(nx, ny, 0.0),
                )
                collector.add_quad(
                    "Buildings_Walls", upper_wall_mat,
                    (c_right[0], c_right[1], fl_z0 + 0.08), (p1[0], p1[1], fl_z0 + 0.08),
                    (p1[0], p1[1], fl_z1), (c_right[0], c_right[1], fl_z1),
                    expected_normal=(nx, ny, 0.0),
                )

                add_3d_win_extras = (lvl <= 2) and (wall_corridor_dist <= 22.0)
                for b_idx in range(num_bays):
                    b_p0 = (c_left[0] + tx * (b_idx * bay_w), c_left[1] + ty * (b_idx * bay_w))
                    b_p1 = (c_left[0] + tx * ((b_idx + 1) * bay_w), c_left[1] + ty * ((b_idx + 1) * bay_w))
                    is_stepback_over_pavilion = (building_id == 7002 and (nx < -0.5 or wall_my > 16.0))
                    if bay_w >= 1.85 and bay_clear_flags[b_idx] and not (is_stepback_over_pavilion and lvl == 1):
                        use_balcony = (
                            all_upper_balconies
                            or (lvl == 1 and b_idx == num_bays // 2 and (is_hero or building_id % 4 == 0))
                        ) and not is_stepback_over_pavilion
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

                    if is_hero and b_idx < num_bays and bay_clear_flags[b_idx]:
                        pil_u = corner_margin + b_idx * bay_w
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
    # Strictly verified with `_find_safe_roof_interior_point` so concave or L-shaped buildings
    # NEVER spawn floating domes or rooftop boilers outside their roof polygon!
    if min(span_x, span_y) >= 6.0 and lod_tier <= 1:
        dome_r = min(2.6, min(span_x, span_y) * 0.24)
        safe_pt = _find_safe_roof_interior_point(pts, max(2.1, dome_r + 0.45))
        if safe_pt is not None:
            rx_in, ry_in = safe_pt
            if has_dome or (building_id % 4 == 0):
                collector.add_cylinder(
                    "Buildings_Details",
                    trim_stone_mat,
                    (rx_in, ry_in, roof_z + 0.15),
                    dome_r * 1.04,
                    dome_r,
                    0.55,
                    segments=8,
                    smooth=False,
                )
                collector.add_dome(
                    "Buildings_Details",
                    "mat_plaster",
                    (rx_in, ry_in, roof_z + 0.70),
                    radius=dome_r,
                    height=dome_r * 0.78,
                    rings=3,
                    segments=8,
                    smooth=True,
                )
            elif lod_tier <= 1 and (building_id % 2 == 0 or is_hero):
                build_rooftop_clutter(collector, rx_in, ry_in, roof_z + 0.15, span_x, span_y, seed=building_id)


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
    # --- A. Structural Piers framing the Walkable L-Turn Passage (`z = -1.0 .. 5.50m`) ---
    # 1. North Solid Wall Wing (`x = -8.50 .. 6.20, y = 4.90 .. 8.80, z = -1.00 .. 5.50`)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-1.15, 6.85, 2.25), (14.7, 3.9, 6.5), top_face=False, bottom_face=True)

    # 2. South-West Bastion Pier (`x = -8.50 .. -1.30, y = -5.50 .. 0.50, z = -1.00 .. 5.50`)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-4.90, -2.50, 2.25), (7.2, 6.0, 6.5), top_face=False, bottom_face=True)

    # 3. East Blind Curtain Wall & SE Pier (`x = 3.10 .. 6.20, y = -5.50 .. 4.90, z = -1.00 .. 5.50`)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (4.65, -0.30, 2.25), (3.1, 10.4, 6.5), top_face=False, bottom_face=True)

    # 4. Unified Upper Gatehouse Tower (`x = -8.50 .. 6.20, y = -5.50 .. 8.80, z = 5.50 .. 15.80`)
    # Its `bottom_face=True` at `z = 5.50m` forms the 100% sealed interior ceiling with ZERO coplanar wall overlap!
    collector.add_box(
        "JaffaGate_Monument",
        "mat_stone_ashlar",
        (-1.15, 1.65, 10.65),
        (14.7, 14.3, 10.3),
        top_face=True,
        bottom_face=True,
    )

    # Split Horizontal Stone Belt Course on West Facade (`x = -8.50 .. -8.62m, z = 5.42 .. 5.62m`):
    # Extended right up to the Outer Portal Arch extrados (`y = -5.50 .. 0.02` & `y = 5.38 .. 8.80`)
    collector.add_box("JaffaGate_Monument", "mat_stone_mizzi", (-8.56, -2.74, 5.52), (0.12, 5.52, 0.20), bottom_face=True)
    collector.add_box("JaffaGate_Monument", "mat_stone_mizzi", (-8.56, 7.09, 5.52), (0.12, 3.42, 0.20), bottom_face=True)

    # Solid 3D Curved Portal Spandrel Walls above Inner West Portal (`x = -8.15, span_w = 3.80`) & South Exit Portal (`y = -5.50, span_w = 3.70`)
    add_arch_spandrel_wall(
        collector,
        "JaffaGate_Monument",
        "mat_stone_boss",
        (-8.15, 2.7, 3.35),
        (0.0, 1.0),
        (-1.0, 0.0),
        span_w=3.8,
        arch_h=1.25,
        z_top=5.50,
        offset_n0=-0.95,
        offset_n1=-0.36,
        num_voussoirs=9,
        emit_soffit=True,
    )

    # South Exit Portal Side Jambs (`x = -1.30 .. -0.95` & `x = 2.75 .. 3.10`, `y = -5.50 .. -4.85`)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-1.125, -5.175, 2.25), (0.35, 0.65, 6.5), top_face=False)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (2.925, -5.175, 2.25), (0.35, 0.65, 6.5), top_face=False)

    add_arch_spandrel_wall(
        collector,
        "JaffaGate_Monument",
        "mat_stone_boss",
        (0.9, -5.50, 3.35),
        (1.0, 0.0),
        (0.0, -1.0),
        span_w=3.70,
        arch_h=1.35,
        z_top=5.50,
        offset_n0=-0.65,
        offset_n1=0.0,
        num_voussoirs=9,
        emit_soffit=True,
    )

    # 5. Complete Diaphragm Stone Vault Ribs inside the L-Passage (sealed right up to `z = 5.50m` ceiling!)
    # 5a. West-East entry arm ribs (`x = -6.6, -3.8`, `span_w = 3.84m` matching inner faces of 0.28m pilasters!)
    for arch_x in (-6.6, -3.8):
        for pil_y in (0.64, 4.76):
            collector.add_box("JaffaGate_Monument", "mat_stone_boss", (arch_x, pil_y, 2.25), (0.58, 0.34, 6.5), top_face=False)
            collector.add_box("JaffaGate_Monument", "mat_stone_mizzi", (arch_x, pil_y, 3.48), (0.66, 0.42, 0.18))
        add_arch_spandrel_wall(
            collector,
            "JaffaGate_Monument",
            "mat_stone_ashlar_aged",
            (arch_x, 2.7, 3.55),
            (0.0, 1.0),
            (-1.0, 0.0),
            span_w=3.84,
            arch_h=1.15,
            z_top=5.50,
            offset_n0=-0.24,
            offset_n1=0.24,
            num_voussoirs=9,
            emit_soffit=False,
        )
        add_arch_voussoirs(
            collector,
            "JaffaGate_Monument",
            "mat_stone_boss",
            (arch_x, 2.7, 3.55),
            (0.0, 1.0),
            (-1.0, 0.0),
            span_w=3.84,
            arch_h=1.15,
            ring_thickness=0.30,
            offset_n0=-0.26,
            offset_n1=0.26,
            num_voussoirs=9,
            keystone_boost=0.04,
            back_face=True,
        )
    # 5b. North-South turn chamber ribs (`y = -1.5, -3.5`, `span_w = 3.84m` matching inner faces of 0.28m pilasters!)
    for arch_y in (-1.5, -3.5):
        for pil_x in (-1.16, 2.96):
            collector.add_box("JaffaGate_Monument", "mat_stone_boss", (pil_x, arch_y, 2.25), (0.34, 0.58, 6.5), top_face=False)
            collector.add_box("JaffaGate_Monument", "mat_stone_mizzi", (pil_x, arch_y, 3.48), (0.42, 0.66, 0.18))
        add_arch_spandrel_wall(
            collector,
            "JaffaGate_Monument",
            "mat_stone_ashlar_aged",
            (0.9, arch_y, 3.55),
            (1.0, 0.0),
            (0.0, -1.0),
            span_w=3.84,
            arch_h=1.15,
            z_top=5.50,
            offset_n0=-0.24,
            offset_n1=0.24,
            num_voussoirs=9,
            emit_soffit=False,
        )
        add_arch_voussoirs(
            collector,
            "JaffaGate_Monument",
            "mat_stone_boss",
            (0.9, arch_y, 3.55),
            (1.0, 0.0),
            (0.0, -1.0),
            span_w=3.84,
            arch_h=1.15,
            ring_thickness=0.30,
            offset_n0=-0.26,
            offset_n1=0.26,
            num_voussoirs=9,
            keystone_boost=0.04,
            back_face=True,
        )
    # Corner turn stone pilasters (`(-1.14, 0.64)` & `(2.96, 4.76)`) and Ottoman carved interior inscription relief tablet (`pair_jaffa_gate_passage.jpg`)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-1.14, 0.64, 2.25), (0.36, 0.36, 6.5), top_face=False)
    collector.add_box("JaffaGate_Monument", "mat_stone_mizzi", (-1.14, 0.64, 3.48), (0.44, 0.44, 0.18))
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (1.25, 4.84, 3.65), (1.45, 0.14, 0.78))
    collector.add_box("JaffaGate_Monument", "mat_stone_mizzi", (1.25, 4.78, 3.65), (1.22, 0.08, 0.56))
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (3.04, 0.85, 3.65), (0.14, 1.35, 0.74))
    collector.add_box("JaffaGate_Monument", "mat_stone_mizzi", (2.98, 0.85, 3.65), (0.08, 1.12, 0.54))
    # South Exit Portal Arch (`y = -5.50, span_w = 3.70m, ring_thickness = 0.35m` -> extrados `x = -1.30 .. +3.10` sits 100% outside the piers!)
    add_arch_voussoirs(
        collector,
        "JaffaGate_Monument",
        "mat_stone_boss",
        (0.9, -5.50, 3.35),
        (1.0, 0.0),
        (0.0, -1.0),
        span_w=3.70,
        arch_h=1.35,
        ring_thickness=0.35,
        offset_n0=-0.45,
        offset_n1=0.14,
        num_voussoirs=9,
        keystone_boost=0.05,
    )

    # --- B. West Outer Monumental Pointed Portal (`x = -8.50`) ---
    # Side portal jambs filling `y = 0.50..0.80` & `y = 4.60..4.90` up to `z = 6.55m`
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-7.65, 0.65, 2.75), (1.7, 0.32, 7.6))
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-7.65, 4.75, 2.75), (1.7, 0.32, 7.6))

    # Inner West Portal Arch (`x = -8.15, z_spring = 3.35m, span_w = 3.80m`)
    add_arch_voussoirs(
        collector,
        "JaffaGate_Monument",
        "mat_stone_boss",
        (-8.15, 2.7, 3.35),
        (0.0, 1.0),
        (-1.0, 0.0),
        span_w=3.8,
        arch_h=1.25,
        ring_thickness=0.32,
        offset_n0=-0.35,
        offset_n1=0.08,
        num_voussoirs=9,
        keystone_boost=0.04,
    )

    # Solid Outer Portal Spandrel Wall (`x = -8.48..-8.12, span_w = 4.40m, z_top = 6.55m`, non-coplanar with `x = -8.50`!)
    add_arch_spandrel_wall(
        collector,
        "JaffaGate_Monument",
        "mat_stone_ashlar",
        (-8.50, 2.7, 4.20),
        (0.0, 1.0),
        (-1.0, 0.0),
        span_w=4.40,
        arch_h=2.05,
        z_top=6.55,
        offset_n0=-0.38,
        offset_n1=-0.02,
        num_voussoirs=11,
        emit_soffit=True,
    )
    # Carved stone lintel moulding across the recessed tympanum (`z = 4.82m`, cleanly above inner arch crown!)
    collector.add_box("JaffaGate_Monument", "mat_stone_mizzi", (-8.20, 2.7, 4.82), (0.10, 3.6, 0.12), bottom_face=True)

    # Pointed Outer Portal Arch (`x = -8.49 .. -8.84m, z_spring = 4.20m, span_w = 4.40m`)
    add_arch_voussoirs(
        collector,
        "JaffaGate_Monument",
        "mat_stone_boss",
        (-8.50, 2.7, 4.20),
        (0.0, 1.0),
        (-1.0, 0.0),
        span_w=4.40,
        arch_h=2.05,
        ring_thickness=0.42,
        offset_n0=-0.01,
        offset_n1=0.34,
        num_voussoirs=11,
        keystone_boost=0.06,
    )

    # Sultan Suleiman Inscription Tympanum Plaque recessed cleanly INSIDE the outer arch (`x = -8.20m` & `-8.25m`)
    # Width `2.15m`, height `0.66m` centered at `z = 5.32m` so its corners are 100% enclosed inside the pointed arch!
    collector.add_box("JaffaGate_Monument", "mat_stone_ashlar_aged", (-8.20, 2.7, 5.32), (0.08, 2.15, 0.66))
    collector.add_box("JaffaGate_Monument", "mat_stone_mizzi", (-8.25, 2.7, 5.32), (0.06, 1.90, 0.52))

    # Twin carved Ottoman stone rosettes oriented flat against the West wall (axis along -X, NOT vertical!)
    for ros_y in (0.15, 5.25):
        collector.add_branch_tube(
            "JaffaGate_Monument",
            "mat_stone_mizzi",
            (-8.50, ros_y, 6.65),
            (-8.60, ros_y, 6.65),
            0.36,
            0.32,
            segments=8,
            smooth=False,
            cap_end=True,
        )

    # Twin recessed vertical arrow-slit embrasures on Upper West Gatehouse (`z = 8.15 .. 10.25m`) + string course
    collector.add_box("JaffaGate_Monument", "mat_stone_mizzi", (-8.55, 1.65, 10.55), (0.10, 14.3, 0.16), bottom_face=True)
    for emb_y in (1.15, 4.25):
        collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-8.54, emb_y, 9.20), (0.10, 0.54, 2.15))
        collector.add_box("JaffaGate_Monument", "mat_iron_dark", (-8.59, emb_y, 9.20), (0.04, 0.14, 1.75))

    # Projecting Corbelled Stone Machicolation Box high on the West Facade (`z = 11.4 .. 13.7`)
    for c_y in (0.8, 1.75, 2.7, 3.65, 4.6):
        collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-8.72, c_y, 10.95), (0.44, 0.34, 0.45), bottom_face=True)
        collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-8.92, c_y, 11.35), (0.84, 0.36, 0.42), bottom_face=True)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-8.98, 2.7, 12.65), (0.95, 4.3, 2.25), bottom_face=True)

    for slit_y in (-2.6, 7.1):
        collector.add_branch_tube(
            "JaffaGate_Monument",
            "mat_stone_boss",
            (-8.50, slit_y, 12.2),
            (-8.60, slit_y, 12.2),
            0.40,
            0.36,
            segments=8,
            smooth=False,
            cap_end=True,
        )

    # Crenellated Ottoman Bevelled Merlons (`z = 15.8 .. 17.2`)
    collector.add_box("JaffaGate_Monument", "mat_stone_ashlar_aged", (-1.15, 1.65, 15.65), (15.1, 14.7, 0.28))
    for mx in (-7.6, -5.2, -2.8, -0.4, 2.0, 4.4):
        for my in (-5.0, 8.3):
            collector.add_box("JaffaGate_Monument", "mat_stone_ashlar_aged", (mx, my, 16.5), (1.35, 0.65, 1.4))
    for my in (-4.4, -2.0, 0.4, 2.8, 5.2, 7.6):
        for mx in (-8.15, 5.85):
            collector.add_box("JaffaGate_Monument", "mat_stone_ashlar_aged", (mx, my, 16.5), (0.65, 1.35, 1.4))

    # --- C. Interior L-Passage Iron-Plated Doors (`z = 0.10 .. 3.26m`), Foreground Acacia Tree in Round Planter, Plaques & Lanterns ---
    collector.add_box("JaffaGate_Monument", "mat_iron_dark", (-5.20, 4.78, 1.68), (1.55, 0.14, 3.16))
    collector.add_box("JaffaGate_Monument", "mat_iron_dark", (-5.20, 0.62, 1.68), (1.55, 0.14, 3.16))
    for strap_z in (0.65, 1.68, 2.70):
        collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-5.20, 4.69, strap_z), (1.58, 0.03, 0.09))
        collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-5.20, 0.71, strap_z), (1.58, 0.03, 0.09))
    # Cylindrical stone planter + slender branching tree flanking the southern pier of Jaffa Gate (`(-10.2, -3.4)`) clear of the archway & plaque
    ac_z = -0.05
    collector.add_branch_tube("StreetProps", "mat_stone_mizzi", (-10.2, -3.4, ac_z), (-10.2, -3.4, ac_z + 0.68), 0.45, 0.43, segments=12, cap_end=True)
    build_olive_tree(collector, -10.2, -3.4, ac_z + 0.62, height=3.8, seed=108, with_kerb=False)

    # Trilingual Karakashian ceramic street name plaques ("JAFFA GATE / BAB EL-KHALIL")
    jg_uv = get_sign_uv(sign_atlas, "jaffa_gate_plaque")
    build_street_name_plate(collector, (3.08, 2.6, 2.45), (-1.0, 0.0), jg_uv, width=1.25, height=0.62)
    build_street_name_plate(collector, (-8.52, -1.5, 2.45), (-1.0, 0.0), jg_uv, width=1.15, height=0.58)
    build_street_name_plate(collector, (4.65, -5.52, 2.45), (0.0, -1.0), jg_uv, width=1.05, height=0.54)

    build_wall_lantern(collector, (-5.20, 4.88, 3.85), (1.0, 0.0), (0.0, -1.0))
    build_wall_lantern(collector, (3.08, -2.50, 3.85), (0.0, 1.0), (-1.0, 0.0))

    # --- C2. East Inner Facade of Jaffa Gate (`x = 6.20`, seen looking west across Omar Square in `pair_omar_square_east.jpg`) ---
    # Tall pointed blind arch + deep dark recessed portal shadow + twin Ottoman stone rosettes
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (6.36, -3.4, 3.2), (0.36, 0.85, 7.6))
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (6.36, 3.8, 3.2), (0.36, 0.85, 7.6))
    # Dark recessed arched portal panel on East face (`x = 6.24`) so the pointed arch silhouette reads crisply from across Omar Square!
    add_arched_opening_panel(
        collector,
        "Buildings_Openings",
        "mat_wood_doors",
        (6.24, 0.2),
        0.0,
        4.8,
        2.35,
        5.55,
        0.02,
        (0.0, -1.0),
        (1.0, 0.0),
        uv_rect=WOOD_DOOR_UV["window_glass"],
        num_voussoirs=9,
        add_wall_spandrel=False,
    )
    add_arch_voussoirs(
        collector,
        "JaffaGate_Monument",
        "mat_stone_boss",
        (6.20, 0.2, 4.8),
        (0.0, -1.0),
        (1.0, 0.0),
        span_w=5.65,
        arch_h=2.45,
        ring_thickness=0.45,
        offset_n0=-0.02,
        offset_n1=0.38,
        num_voussoirs=11,
        keystone_boost=0.06,
    )
    for ros_y_e in (-2.4, 2.8):
        collector.add_branch_tube(
            "JaffaGate_Monument",
            "mat_stone_mizzi",
            (6.20, ros_y_e, 8.6),
            (6.32, ros_y_e, 8.6),
            0.38,
            0.34,
            segments=8,
            smooth=False,
            cap_end=True,
        )

    # --- C3. North Curtain Wall Recessed Pointed Blind Arches & Bosses (`pair_west_plaza.jpg` & `pair_jaffa_gate_portal.jpg`) ---
    # 1. Blind pointed arch directly on the North Wing of Jaffa Gate (`x = -8.50, y = 5.4 .. 8.6`, visible in `pair_jaffa_gate_portal.jpg` left!)
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-8.64, 5.45, 2.5), (0.28, 0.42, 5.2))
    collector.add_box("JaffaGate_Monument", "mat_stone_boss", (-8.64, 8.55, 2.5), (0.28, 0.42, 5.2))
    add_arched_opening_panel(
        collector,
        "JaffaGate_Monument",
        "mat_stone_rubble",
        (-8.53, 7.0),
        0.0,
        3.15,
        1.55,
        2.75,
        0.02,
        (0.0, 1.0),
        (-1.0, 0.0),
        uv_rect=(0.0, 0.0, 1.0, 1.0),
        num_voussoirs=9,
        add_wall_spandrel=False,
    )
    add_arch_voussoirs(
        collector,
        "JaffaGate_Monument",
        "mat_stone_boss",
        (-8.50, 7.0, 3.15),
        (0.0, 1.0),
        (-1.0, 0.0),
        span_w=2.80,
        arch_h=1.60,
        ring_thickness=0.36,
        offset_n0=-0.02,
        offset_n1=0.30,
        num_voussoirs=9,
        keystone_boost=0.05,
    )
    # 2. Flanking North Curtain Wall Wing (`x = -8.50, y = 8.80 .. 15.20`, visible in `pair_west_plaza.jpg` left!)
    collector.add_box("CityWalls", "mat_stone_boss", (-6.2, 12.0, 5.9), (4.6, 6.4, 15.8))
    collector.add_box("CityWalls", "mat_stone_boss", (-8.68, 9.10, 2.6), (0.36, 0.65, 7.2))
    collector.add_box("CityWalls", "mat_stone_boss", (-8.68, 14.30, 2.6), (0.36, 0.65, 7.2))
    add_arch_spandrel_wall(
        collector,
        "CityWalls",
        "mat_stone_ashlar_aged",
        (-8.50, 11.7, 3.80),
        (0.0, 1.0),
        (-1.0, 0.0),
        span_w=4.55,
        arch_h=2.15,
        z_top=6.45,
        offset_n0=0.02,
        offset_n1=0.36,
        num_voussoirs=11,
        emit_soffit=True,
    )
    add_arch_voussoirs(
        collector,
        "CityWalls",
        "mat_stone_boss",
        (-8.50, 11.7, 3.80),
        (0.0, 1.0),
        (-1.0, 0.0),
        span_w=4.55,
        arch_h=2.15,
        ring_thickness=0.40,
        offset_n0=-0.02,
        offset_n1=0.38,
        num_voussoirs=11,
        keystone_boost=0.06,
    )
    for ros_y_n in (9.4, 14.0):
        collector.add_branch_tube(
            "CityWalls",
            "mat_stone_mizzi",
            (-8.50, ros_y_n, 6.45),
            (-8.64, ros_y_n, 6.45),
            0.36,
            0.32,
            segments=8,
            smooth=False,
            cap_end=True,
        )
    for my_n in (9.6, 11.8, 14.0):
        collector.add_box("CityWalls", "mat_stone_ashlar_aged", (-8.15, my_n, 14.5), (0.65, 1.35, 1.4))

    # --- D. Northern Ottoman City Wall Segment & Southern Citadel Moat Rampart Parapet ---
    nw_wall_pts = [
        (-6.2, 14.8),
        (-14.0, 26.0),
        (-28.0, 50.0),
        (-48.0, 82.0),
        (-74.0, 122.0),
    ]
    for wall_chain in (nw_wall_pts,):
        for j_pt in wall_chain[1:-1]:
            collector.add_box("CityWalls", "mat_stone_boss", (j_pt[0], j_pt[1], 1.5), (3.1, 3.1, 8.0))
            collector.add_box("CityWalls", "mat_stone_ashlar_aged", (j_pt[0], j_pt[1], 9.15), (2.7, 2.7, 7.3))
        for i in range(len(wall_chain) - 1):
            wp0 = wall_chain[i]
            wp1 = wall_chain[i + 1]
            dx = wp1[0] - wp0[0]
            dy = wp1[1] - wp0[1]
            seg_l = math.hypot(dx, dy)
            tx, ty = dx / seg_l, dy / seg_l
            nx, ny = -dy / seg_l, dx / seg_l
            ext0 = (wp0[0] - tx * 0.9, wp0[1] - ty * 0.9)
            ext1 = (wp1[0] + tx * 0.9, wp1[1] + ty * 0.9)
            collector.add_oriented_box("CityWalls", "mat_stone_boss", ext0, ext1, -2.5, 5.5, -1.4, 1.4, (nx, ny), cap_back=True)
            collector.add_oriented_box("CityWalls", "mat_stone_ashlar_aged", ext0, ext1, 5.5, 12.8, -1.2, 1.2, (nx, ny), cap_back=True)
            num_m = int(seg_l / 2.4)
            for m_i in range(num_m):
                u0 = m_i * 2.4 + 0.3
                u1 = u0 + 1.35
                mp0 = (wp0[0] + tx * u0, wp0[1] + ty * u0)
                mp1 = (wp0[0] + tx * u1, wp0[1] + ty * u1)
                collector.add_oriented_box("CityWalls", "mat_stone_ashlar_aged", mp0, mp1, 12.8, 14.1, 0.4, 1.2, (nx, ny), cap_back=True)

    # Southern Citadel Moat & Maronite Convent St Rampart Promenade (`x = 11.2..16.2, y = -138.0..-32.0`):
    # Waist-high crenellated western rampart parapet (`x = 10.6`) overlooking the valley + eastern iron handrail (`x = 15.8`)
    # + recessed pointed-arch casemate vaults along the lower Citadel moat scarp (`x = 16.0, y = -68..-34`)
    for y_cas in (-62.0, -54.0, -46.0, -38.0):
        add_arched_opening_panel(
            collector,
            "CityWalls",
            "mat_stone_rubble",
            (15.92, y_cas),
            -3.6,
            2.2,
            1.45,
            3.2,
            0.02,
            (0.0, -1.0),
            (-1.0, 0.0),
            uv_rect=(0.0, 0.0, 1.0, 1.0),
            num_voussoirs=9,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "CityWalls",
            "mat_stone_boss",
            (15.92, y_cas, 2.2),
            (0.0, -1.0),
            (-1.0, 0.0),
            span_w=3.35,
            arch_h=1.50,
            ring_thickness=0.38,
            offset_n0=-0.04,
            offset_n1=0.24,
            num_voussoirs=9,
        )
    for y_seg in range(-136, -74, 4):
        ym_s = float(y_seg) + 2.0
        pz_s = _global_street_elevation_z(13.8, ym_s)
        # Western stone rampart parapet wall (`x = 10.6`, waist-high `1.45m` with stepped merlons)
        collector.add_box("CityWalls", "mat_stone_boss", (10.6, ym_s, pz_s - 1.2), (1.1, 4.05, 5.2), uv_scale=0.24)
        if (y_seg // 4) % 2 == 0:
            collector.add_box("CityWalls", "mat_stone_ashlar_aged", (10.6, ym_s, pz_s + 1.85), (0.95, 2.2, 0.95), uv_scale=0.28)
        # Eastern wrought-iron safety railing (`x = 16.2`) along Maronite Convent St
        collector.add_box("StreetProps", "mat_iron_dark", (16.2, ym_s, pz_s + 0.98), (0.06, 4.05, 0.06))
        collector.add_box("StreetProps", "mat_iron_dark", (16.2, ym_s, pz_s + 0.52), (0.05, 4.05, 0.05))
        for p_dy in (-1.5, 0.0, 1.5):
            collector.add_box("StreetProps", "mat_iron_dark", (16.2, ym_s + p_dy, pz_s + 0.50), (0.06, 0.06, 1.0))

    # Elevated Western Citadel Moat Rampart Walkway (`x = 10.0..12.6, y = -76.0..-40.0`, framing `citadel_moat_1_fwd/rev` at `(11.86, -54.3)`)
    for y_seg_m in range(-76, -40, 4):
        ym_m = float(y_seg_m) + 2.0
        pz_m = _global_street_elevation_z(11.8, ym_m)
        # Grounded stone retaining wall supporting the `x = 10.0..12.5` rampart walkway from `moat_z = -6.0` up to `pz_m`
        collector.add_box("CityWalls", "mat_stone_ashlar_aged", (11.3, ym_m, 0.5 * (-6.0 + pz_m)), (2.5, 4.05, pz_m + 6.0), uv_scale=0.24)
        # Western crenellated rampart parapet wall (`x = 10.2`)
        collector.add_box("CityWalls", "mat_stone_boss", (10.2, ym_m, pz_m + 0.85), (0.65, 4.05, 1.70), uv_scale=0.25)
        if (y_seg_m // 4) % 2 == 0:
            collector.add_box("CityWalls", "mat_stone_boss", (10.2, ym_m, pz_m + 2.05), (0.62, 2.1, 0.75), uv_scale=0.28)
        # Eastern wrought-iron safety railing (`x = 12.48`) overlooking the Citadel moat & Tower of David
        collector.add_box("StreetProps", "mat_iron_dark", (12.48, ym_m, pz_m + 0.98), (0.06, 4.05, 0.06))
        collector.add_box("StreetProps", "mat_iron_dark", (12.48, ym_m, pz_m + 0.52), (0.05, 4.05, 0.05))
        for p_dy in (-1.5, 0.0, 1.5):
            collector.add_box("StreetProps", "mat_iron_dark", (12.48, ym_m + p_dy, pz_m + 0.50), (0.06, 0.06, 1.0))


def build_citadel_and_tower_of_david(
    collector: MeshCollector,
    sign_atlas: Dict[str, dict],
) -> None:
    """
    Constructs the Citadel (`El-Qal'a`) and Tower of David (`Phasael Tower`) complex:
    - Widened 13.5m Sunken Dry Moat (`x = 10..54, y = -29.0..-15.5, z = -6.0m`) with low
      `0.42m` stone counterscarp curb + `0.78m` open wrought-iron railing so the full
      moat basin, tumbled ruins, and lush Mediterranean moat trees are clearly
      visible from Omar Square and the 1898 Breach!
    - Crenellated NW Moat Corner Parapet + "Night Spectacular / Tower of David" sign plaque.
    - Continuous Battered Stone Glacis (`z = -6.0m .. +1.4m`) rising seamlessly into the
      unified `mat_stone_boss` Citadel Curtain Wall (`z = +1.4m .. +14.6m`).
    - Phasael Tower (`Tower of David`) at `(26.25, -37.0)` rising to `z = +27.5m` (`33.5m` above moat)
      with recessed dark iron-grilled central windows and upper square embrasures.
    - Arched Stone Moat Bridge (`x = 42..47, y = -29.0..-15.2`) & Museum Gatehouse (`y = -30.5`).
    - Cylindrical Mamluk Minaret at `(21.6, -66.0)` with triple muqarnas balconies.
    """
    moat_z = -6.0
    street_z = 0.35

    # --- 1. Widened Sunken Dry Moat Floor & Low 0.42m Counterscarp Curb + 0.78m Open Iron Railing ---
    moat_floor = [
        (9.5, -15.0, moat_z),
        (54.5, -15.0, moat_z),
        (54.5, -76.5, moat_z),
        (9.5, -76.5, moat_z),
    ]
    collector.add_polygon_3d("Citadel_TowerOfDavid", "mat_stone_rubble", moat_floor, uv_scale=0.22)

    counterscarp_segs = [
        ((9.42, -15.5), (42.0, -15.5), (0.0, -1.0)),   # North-West moat edge (flush corner at x=9.42)
        ((47.0, -15.5), (54.58, -15.5), (0.0, -1.0)),  # North-East moat edge (flush corner at x=54.58)
        ((10.0, -76.0), (10.0, -15.5), (1.0, 0.0)),    # West moat edge along Breach/Moat road
        ((9.42, -76.0), (16.5, -76.0), (0.0, 1.0)),    # South-West moat closure wall
        ((54.0, -29.5), (54.0, -15.5), (-1.0, 0.0)),   # East moat edge along Omar Sq south
    ]
    for p0, p1, n_in in counterscarp_segs:
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

    # Crenellated stone merlons along the NW Moat Corner (`x = 9.7..20.5, y = -15.24` & `x = 9.74, y = -25.0..-15.5`, seen in `pair_tower_of_david.jpg`)
    for cm_x in (10.3, 12.5, 14.7, 16.9, 19.1):
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (cm_x, -15.24, street_z + 0.78), (1.25, 0.48, 0.68))
    for cm_y in (-16.2, -18.4, -20.6, -22.8, -25.0):
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (9.74, cm_y, street_z + 0.78), (0.48, 1.25, 0.68))
    tod_uv_corner = get_sign_uv(sign_atlas, "tower_of_david_museum")
    build_street_name_plate(collector, (10.3, -14.96, street_z + 0.80), (0.0, 1.0), tod_uv_corner, width=1.05, height=0.62)

    # Mitered wrought-iron safety railings centered on the stone curb cap (`offset = -0.26m`)
    railing_segs = [
        ((9.74, -75.74), (9.74, -25.80)),
        ((20.00, -15.24), (42.20, -15.24)),
        ((46.80, -15.24), (54.26, -15.24)),
        ((54.26, -15.24), (54.26, -29.00)),
    ]
    for rp0, rp1 in railing_segs:
        build_iron_railing(
            collector,
            (rp0[0], rp0[1], street_z + 0.44),
            (rp1[0], rp1[1], street_z + 0.44),
            height=0.78,
            picket_spacing=0.24,
        )

    # --- 2. Lush Mediterranean Moat Trees, Date Palms & Coursed Ruin Foundations inside the Widened Moat ---
    build_palm_tree(collector, 22.5, -22.2, moat_z, height=13.2, seed=101)
    build_palm_tree(collector, 34.5, -22.2, moat_z, height=11.5, seed=106)
    # Dense dark-green Mediterranean foliage kept >= 3.2m inside the moat walls so zero leaves ever clip the curb (`pair_breach_1898.jpg` & `pair_tower_of_david.jpg`)
    build_cypress_tree(collector, 13.5, -22.0, moat_z, height=11.8, seed=111, with_kerb=False)
    build_cypress_tree(collector, 13.5, -30.0, moat_z, height=11.2, seed=112, with_kerb=False)
    build_cypress_tree(collector, 13.5, -38.5, moat_z, height=10.8, seed=113, with_kerb=False)
    build_olive_tree(collector, 18.8, -22.8, moat_z, height=7.4, seed=115, with_kerb=False)
    build_olive_tree(collector, 29.0, -22.8, moat_z, height=7.4, seed=116, with_kerb=False)
    build_cypress_tree(collector, 38.8, -22.5, moat_z, height=10.8, seed=117, with_kerb=False)

    ruin_blocks = [
        (18.0, -21.5, 2.4, 1.5, 0.52, 0.0),
        (26.5, -21.8, 3.0, 1.4, 0.48, 0.0),
        (31.5, -23.0, 2.2, 1.8, 0.55, 0.0),
        (38.5, -21.5, 2.4, 1.4, 0.45, 0.0),
        (50.2, -22.0, 2.2, 1.6, 0.50, 0.0),
    ]
    for rx, ry, sx, sy, sz, rot in ruin_blocks:
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (rx, ry, moat_z + sz * 0.5), (sx, sy, sz), rot_z=rot)

    # --- 3. Continuous Mitered Battered Stone Glacis + Unified Citadel Curtain Wall (`mat_stone_boss`) ---
    citadel_ring = [
        (16.5, -29.0),  # NW corner (Phasael Tower base, leaving 3.9m clear moat floor inside x=10.0!)
        (16.0, -54.0),  # West curtain wall
        (16.0, -76.0),  # SW Minaret Bastion corner
        (48.0, -82.0),  # South curtain wall
        (58.0, -54.0),  # SE Bastion
        (56.0, -29.0),  # NE Gatehouse & Omar Square corner
    ]
    n_cit = len(citadel_ring)
    glacis_top_z = 1.4
    curtain_top_z = 14.6
    glacis_flare = 2.6

    # Precompute segment outward normals and mitered corner normals so the 6-sided glacis has zero corner gaps!
    seg_normals: List[Vec2] = []
    for i in range(n_cit):
        cp0 = citadel_ring[i]
        cp1 = citadel_ring[(i + 1) % n_cit]
        dx, dy = cp1[0] - cp0[0], cp1[1] - cp0[1]
        sl = math.hypot(dx, dy)
        seg_normals.append((dy / sl, -dx / sl))

    miter_normals: List[Vec2] = []
    for i in range(n_cit):
        n_prev = seg_normals[(i - 1) % n_cit]
        n_curr = seg_normals[i]
        mx = n_prev[0] + n_curr[0]
        my = n_prev[1] + n_curr[1]
        ml = math.hypot(mx, my)
        if ml > 1e-5:
            mx, my = mx / ml, my / ml
            cos_half = max(0.75, mx * n_curr[0] + my * n_curr[1])
            miter_normals.append((mx / cos_half, my / cos_half))
        else:
            miter_normals.append(n_curr)

    for i in range(n_cit):
        cp0 = citadel_ring[i]
        cp1 = citadel_ring[(i + 1) % n_cit]
        dx = cp1[0] - cp0[0]
        dy = cp1[1] - cp0[1]
        seg_l = math.hypot(dx, dy)
        tx, ty = dx / seg_l, dy / seg_l
        nx, ny = seg_normals[i]
        mn0 = miter_normals[i]
        mn1 = miter_normals[(i + 1) % n_cit]

        # 3a. Watertight Mitered Sloped Battered Stone Glacis (`z = -6.0 .. +1.4m`)
        gb0 = (cp0[0] + mn0[0] * glacis_flare, cp0[1] + mn0[1] * glacis_flare, moat_z)
        gb1 = (cp1[0] + mn1[0] * glacis_flare, cp1[1] + mn1[1] * glacis_flare, moat_z)
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

    # --- 4. Phasael Tower (`Tower of David`, NW Bastion aligned flush with `citadel_ring[0] = (16.5, -29.0)` above `glacis_top_z = 1.4m`) ---
    px, py = 26.25, -37.0
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px, py, 8.2), (19.5, 16.0, 13.6))
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px, py, 15.3), (18.6, 15.2, 1.0))
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px, py, 20.8), (17.8, 14.4, 10.0))
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px, py, 26.05), (18.5, 15.1, 0.5))

    # Prominent recessed dark iron-grilled central windows on Phasael Tower West (`x = 16.5`) & North (`y = -29.0`) faces (`pair_breach_1898.jpg` & `pair_tower_of_david.jpg`)
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_mizzi", (16.44, py + 1.2, 10.2), (0.18, 1.95, 2.85))
    collector.add_box("Buildings_Openings", "mat_wood_doors", (16.38, py + 1.2, 10.2), (0.08, 1.45, 2.35), uv_rect=WOOD_DOOR_UV["window_glass"])
    collector.add_box("Buildings_Openings", "mat_iron_grille", (16.34, py + 1.2, 10.2), (0.04, 1.45, 2.35))

    collector.add_box("Citadel_TowerOfDavid", "mat_stone_mizzi", (px - 2.0, -28.94, 10.2), (1.95, 0.18, 2.85))
    collector.add_box("Buildings_Openings", "mat_wood_doors", (px - 2.0, -28.88, 10.2), (1.45, 0.08, 2.35), uv_rect=WOOD_DOOR_UV["window_glass"])
    collector.add_box("Buildings_Openings", "mat_iron_grille", (px - 2.0, -28.84, 10.2), (1.45, 0.04, 2.35))

    # Upper row of 3 square embrasure openings under the Phasael Tower parapet (`z = 24.4m`)
    for m_x in (px - 4.2, px, px + 4.2):
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (m_x, py + 7.65, 23.0), (2.3, 0.95, 2.1), bottom_face=True)
        collector.add_box("Citadel_TowerOfDavid", "mat_iron_dark", (m_x, py + 8.15, 19.6), (0.24, 0.12, 1.9))
        collector.add_box("Buildings_Openings", "mat_iron_dark", (m_x, py + 7.22, 24.6), (0.92, 0.12, 1.05))
    for m_y in (py - 3.6, py, py + 3.6):
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px - 9.25, m_y, 23.0), (0.95, 2.2, 2.1), bottom_face=True)
        collector.add_box("Citadel_TowerOfDavid", "mat_iron_dark", (px - 9.75, m_y, 19.6), (0.12, 0.24, 1.9))
        collector.add_box("Buildings_Openings", "mat_iron_dark", (px - 8.92, m_y, 24.6), (0.12, 0.92, 1.05))

    for mx in range(-8, 9, 3):
        for side_y in (py - 7.1, py + 7.1):
            collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (px + mx, side_y, 27.0), (1.55, 0.68, 1.4))
    for my in range(-5, 6, 3):
        for side_x in (px - 8.8, px + 8.8):
            collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (side_x, py + my, 27.0), (0.68, 1.55, 1.4))

    collector.add_box("Citadel_TowerOfDavid", "mat_stone_ashlar_aged", (px + 3.2, py + 1.8, 27.7), (4.4, 4.0, 2.8))
    collector.add_dome("Citadel_TowerOfDavid", "mat_stone_ashlar_aged", (px + 3.2, py + 1.8, 29.1), radius=1.8, height=1.2)

    # --- 5. Arched Stone Citadel Moat Bridge (`x = 42..47, y = -29.0..-15.2`), NE Bastion & Museum Entrance ---
    bx = 44.5
    for pier_y in (-19.8, -24.5):
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (bx, pier_y, -2.8), (4.8, 1.4, 6.4))
    collector.add_box("Citadel_TowerOfDavid", "mat_paving_flagstone", (bx, -22.1, 0.28), (4.8, 14.4, 0.55), bottom_face=True)
    for side_x in (bx - 2.2, bx + 2.2):
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_ashlar_aged", (side_x, -22.1, 0.72), (0.42, 14.4, 0.42))
        build_iron_railing(collector, (side_x, -15.24, 0.94), (side_x, -28.0, 0.94), height=0.68)

    # Solid Gatehouse Bastion extending down to moat_z (-6.0) so zero gap exists between bridge parapet and glacis
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (bx, -30.2, 2.4), (8.4, 4.4, 16.8))
    # Prominent Citadel Northeast Bastion Tower (`x = 48.5..57.5, y = -36.4..-28.0, z = -6.0..15.6m`)
    # Placed strictly SOUTH of the eastern moat (`y <= -28.0`) so the eastern dry moat is 100% open and clear!
    collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (53.0, -32.2, 4.8), (9.0, 8.4, 21.6))
    for mx_b in (49.5, 52.0, 54.5, 56.8):
        collector.add_box("Citadel_TowerOfDavid", "mat_stone_boss", (mx_b, -28.35, 16.25), (1.35, 0.65, 1.3))

    # Recessed arched wooden museum entrance double doors (`door_wood` quadrant) + stone arch voussoirs
    add_arched_opening_panel(
        collector,
        "Buildings_Openings",
        "mat_wood_doors",
        (bx, -27.95),
        0.42,
        2.85,
        1.45,
        3.2,
        0.04,
        (-1.0, 0.0),
        (0.0, 1.0),
        uv_rect=WOOD_DOOR_UV["door_wood"],
        num_voussoirs=7,
        add_wall_spandrel=False,
    )
    add_arch_voussoirs(
        collector,
        "Citadel_TowerOfDavid",
        "mat_stone_boss",
        (bx, -27.95, 2.85),
        (-1.0, 0.0),
        (0.0, 1.0),
        span_w=3.2,
        arch_h=1.45,
        ring_thickness=0.38,
        offset_n0=-0.10,
        offset_n1=0.22,
        num_voussoirs=7,
    )
    tod_uv = get_sign_uv(sign_atlas, "tower_of_david_museum")
    build_fascia_sign(collector, (bx, -27.92, 5.55), (1.0, 0.0), (0.0, 1.0), width=2.8, height=0.88, sign_uv_rect=tod_uv)
    build_street_name_plate(collector, (41.6, -15.35, 1.25), (0.0, 1.0), tod_uv, width=1.45, height=0.68)

    # --- 6. Mamluk Citadel Minaret (`x = 21.6, y = -66.0`, flush with `x = 16.0` western curtain wall) ---
    mx, my = 21.6, -66.0
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

def _build_david_st_souq_trellis_and_bazaar(
    collector: MeshCollector,
    elev_fn: Callable[[float, float], float],
) -> None:
    """
    Constructs the iconic Green-Painted Arched Metal Souq Entrance Trellis Canopy (`x = 80.5..86.5, y = -0.9..4.9`)
    and bustling hanging carpet/textile displays at the entrance of David Street (`pair_david_street_souq.jpg`).
    """
    bz = elev_fn(82.5, 2.0)
    beam_z = bz + 3.45

    # Longitudinal dark-green iron lintel beams along both sides (`y = -0.82` and `y = 4.82`)
    for side_y in (-0.82, 4.82):
        collector.add_box("StreetProps", "mat_awning_green", (83.5, side_y, beam_z), (6.2, 0.14, 0.16))
        for post_x in (80.6, 83.5, 86.4):
            collector.add_box("StreetProps", "mat_awning_green", (post_x, side_y, bz + 1.72), (0.12, 0.12, 3.45))

    # Transverse green iron tie-bars & arched souq ribs spanning `y = -0.82 .. 4.82`
    for rib_x in (80.7, 82.6, 84.5, 86.3):
        collector.add_box("StreetProps", "mat_awning_green", (rib_x, 2.0, beam_z), (0.12, 5.7, 0.12))
        # Central raised barrel-vaulted iron cage (`y = 0.65 .. 3.35, z = beam_z .. beam_z + 1.85`)
        n_seg = 8
        for s_i in range(n_seg):
            a0 = math.pi * (s_i / n_seg)
            a1 = math.pi * ((s_i + 1) / n_seg)
            y0 = 2.0 - 1.38 * math.cos(a0)
            z0 = beam_z + 0.35 + 1.45 * math.sin(a0)
            y1 = 2.0 - 1.38 * math.cos(a1)
            z1 = beam_z + 0.35 + 1.45 * math.sin(a1)
            collector.add_branch_tube("StreetProps", "mat_awning_green", (rib_x, y0, z0), (rib_x, y1, z1), 0.055, 0.055, segments=6, smooth=False)
        for col_y in (0.62, 3.38):
            collector.add_box("StreetProps", "mat_awning_green", (rib_x, col_y, beam_z + 0.20), (0.09, 0.09, 0.42))

    # Longitudinal purlins across the vaulted iron cage
    for s_i in range(1, 8):
        ang = math.pi * (s_i / 8.0)
        py = 2.0 - 1.38 * math.cos(ang)
        pz = beam_z + 0.35 + 1.45 * math.sin(ang)
        collector.add_box("StreetProps", "mat_awning_green", (83.5, py, pz), (5.8, 0.06, 0.06))

    # Hanging souq lantern in the center of the metal arch canopy
    build_wall_lantern(collector, (82.6, 2.0, beam_z + 1.25), (0.0, 1.0), (-1.0, 0.0))

    # Bustling hanging carpets, textiles & spice displays flanking the David Street souq steps (`x = 81.8..108.0`)
    for disp_x in (82.2, 85.8, 90.5, 99.0, 106.5):
        dz = elev_fn(disp_x, 2.0)
        build_souq_display(collector, (disp_x, 4.72, dz), (-1.0, 0.0), (0.0, -1.0), seed=int(disp_x * 10) + 1)
        build_souq_display(collector, (disp_x + 1.4, -0.72, dz), (1.0, 0.0), (0.0, 1.0), seed=int(disp_x * 10) + 2)


def build_hero_omar_square_and_david_street(
    collector: MeshCollector,
    sign_atlas: Dict[str, dict],
    elev_fn: Callable[[float, float], float],
) -> None:
    """
    Constructs the curated hero buildings, street name plates, stepped bazaar corridors,
    and western Mamilla horizon embankment:
    1. Omar ibn al-Khattab Square:
       - `7001`: 3-storey chamfered-corner West Wing (`x = 5.5..21.2, y = 10.0..24.5`) with full
         upper-storey wrought-iron balconies, rooftop balustrade, and arched ground shops
       - `7012`: 1-storey high-arched Franciscan/Samara Corner Pavilion (`x = 26.8..36.8, y = 11.2..19.5`)
         with twin fanlight arches, curved awnings, and colorful hanging bazaar textiles
       - `7002`: 3-storey `New Imperial Hotel` (`x = 26.8..54.5, y = 19.5..27.5` + `x = 36.8..54.5, y = 12.2..27.5`)
       - `7003`: `Petra Hotel` at NE corner of Omar Square (`x = 54.5..80.0, y = 10.5..27.5`)
       - `7004`: `Christian Information Centre` (`x = 62..82, y = -28.0..-14.0`)
    2. Narrow `5.6m`-wide Latin Patriarchate Road (`x = 21.2..26.8, y = 24.5..58.0`) with flying-buttress
       arches (`qantara`) placed strictly at solid `3.0m` building-junction piers (`y = 26.0, 41.0, 56.8`)
    3. Narrow `6.0m`-wide David Street (`Suq el-Bazar`) Stepped Bazaar (`x = 80.2..158.0, y = -1.0..5.0`)
       with green metal souq entrance trellis canopy and flying-buttress arches strictly at solid
       `3.0m` building-junction piers (`x = 96.0, 112.0, 128.0, 144.0, 156.8`)
    4. Western Mamilla Horizon Embankment & Terraced Buildings (`x = -80..-66, y = -52..50`)
    """
    hero_rings = [
        (7001, [(5.5, 10.0), (19.6, 10.0), (21.2, 11.6), (21.2, 24.5), (5.5, 24.5)]),
        (7012, [(26.8, 11.2), (36.8, 11.2), (36.8, 19.5), (26.8, 19.5)]),
        (7002, [(36.8, 12.2), (54.5, 12.2), (54.5, 27.5), (26.8, 27.5), (26.8, 19.5), (36.8, 19.5)]),
        (7003, [(54.5, 10.5), (80.0, 10.5), (80.0, 27.5), (54.5, 27.5)]),
        (7004, [(62.0, -28.0), (82.0, -28.0), (82.0, -14.0), (62.0, -14.0)]),
        (7005, [(8.0, 24.5), (21.2, 24.5), (21.2, 41.0), (8.0, 41.0)]),
        (7015, [(8.0, 41.0), (21.2, 41.0), (21.2, 58.0), (8.0, 58.0)]),
        (7006, [(26.8, 27.5), (46.0, 27.5), (46.0, 41.0), (26.8, 41.0)]),
        (7016, [(26.8, 41.0), (46.0, 41.0), (46.0, 58.0), (26.8, 58.0)]),
        (7007, [(15.0, 57.5), (39.0, 57.5), (39.0, 70.0), (15.0, 70.0)]),
        (7100, [(80.0, 5.0), (96.0, 5.0), (96.0, 10.5), (80.0, 10.5)]),
        (7101, [(80.0, -9.5), (96.0, -9.5), (96.0, -1.0), (80.0, -1.0)]),
        (7102, [(96.0, 5.0), (112.0, 5.0), (112.0, 13.5), (96.0, 13.5)]),
        (7103, [(96.0, -9.5), (112.0, -9.5), (112.0, -1.0), (96.0, -1.0)]),
        (7104, [(112.0, 5.0), (128.0, 5.0), (128.0, 13.5), (112.0, 13.5)]),
        (7105, [(112.0, -9.5), (128.0, -9.5), (128.0, -1.0), (112.0, -1.0)]),
        (7106, [(128.0, 5.0), (144.0, 5.0), (144.0, 13.5), (128.0, 13.5)]),
        (7107, [(128.0, -9.5), (144.0, -9.5), (144.0, -1.0), (128.0, -1.0)]),
        (7108, [(144.0, 5.0), (158.0, 5.0), (158.0, 13.5), (144.0, 13.5)]),
        (7109, [(144.0, -9.5), (158.0, -9.5), (158.0, -1.0), (144.0, -1.0)]),
    ]
    for hid, hring in hero_rings:
        register_scene_building_ring(hid, hring)

    # --- 1A. West Wing of New Imperial Hotel / Tourist Info (`7001`, `x = 5.5..21.2, y = 10.0..24.5`, 3-storey chamfered corner!) ---
    bz_7001 = elev_fn(14.0, 10.0)
    build_building_from_footprint(
        collector,
        [(5.5, 10.0), (19.6, 10.0), (21.2, 11.6), (21.2, 24.5), (5.5, 24.5)],
        base_z=bz_7001,
        levels=3,
        sign_atlas=sign_atlas,
        building_id=7001,
        is_hero=True,
        force_souq_ground=True,
        has_dome=True,
        forced_sign_key="new_imperial_hotel",
        all_upper_balconies=True,
        rooftop_balustrade=True,
    )
    omar_uv = get_sign_uv(sign_atlas, "omar_sq_plaque")
    build_street_name_plate(collector, (18.85, 9.96, bz_7001 + 2.45), (0.0, -1.0), omar_uv, width=0.92, height=0.52)
    build_street_name_plate(collector, (21.24, 12.35, bz_7001 + 2.45), (1.0, 0.0), omar_uv, width=0.92, height=0.52)

    # 2-Storey White Limestone Tourist Information Office Wing (`x = 6.2..19.2, y = 5.2..10.0`) abutting inner Jaffa Gate (`jaffa_gate_2_rev`)
    tio_z = elev_fn(12.5, 7.5)
    collector.add_box("Buildings_Walls", "mat_stone_mizzi", (12.7, 7.6, tio_z + 3.8), (13.0, 4.8, 7.6), uv_scale=0.24)
    collector.add_box("Buildings_Details", "mat_stone_boss", (12.7, 7.6, tio_z + 3.95), (13.3, 5.1, 0.24), uv_scale=0.32)
    collector.add_box("Buildings_Details", "mat_stone_boss", (12.7, 7.6, tio_z + 7.72), (13.5, 5.3, 0.34), uv_scale=0.30)
    for tio_x in (9.5, 15.8):
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (tio_x, 5.18),
            tio_z,
            tio_z + 2.25,
            1.05,
            2.15,
            0.01,
            (1.0, 0.0),
            (0.0, -1.0),
            uv_rect=WOOD_DOOR_UV["door_wood" if tio_x < 12.0 else "window_glass"],
            num_voussoirs=7,
            add_wall_spandrel=False,
        )
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_boss",
            (tio_x, 5.18, tio_z + 2.25),
            (1.0, 0.0),
            (0.0, -1.0),
            span_w=2.25,
            arch_h=1.10,
            ring_thickness=0.32,
            offset_n0=-0.04,
            offset_n1=0.20,
            num_voussoirs=7,
        )
        add_arched_opening_panel(
            collector,
            "Buildings_Openings",
            "mat_wood_doors",
            (tio_x, 5.18),
            tio_z + 4.45,
            tio_z + 6.15,
            0.68,
            1.35,
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
            (tio_x, 5.18, tio_z + 6.15),
            (1.0, 0.0),
            (0.0, -1.0),
            span_w=1.42,
            arch_h=0.70,
            ring_thickness=0.26,
            offset_n0=-0.04,
            offset_n1=0.18,
            num_voussoirs=5,
        )
    # Omar Square Foreground Furnishings (`jaffa_gate_2_fwd`): Red/White Shop Umbrella (`(23.5, 9.2)`), Stone Bench (`(26.2, 6.2)`), Trash Bin & Striped Security Block (`(33.2, -1.4)`)
    umb_o_z = elev_fn(23.5, 9.2)
    collector.add_branch_tube("StreetProps", "mat_iron_dark", (23.5, 9.2, umb_o_z), (23.5, 9.2, umb_o_z + 2.45), 0.05, 0.04, segments=6)
    collector.add_branch_tube("StreetProps", "mat_awning_red", (23.5, 9.2, umb_o_z + 2.15), (23.5, 9.2, umb_o_z + 2.68), 1.55, 0.10, segments=8, cap_end=True)
    bn_o_z = elev_fn(26.2, 6.2)
    collector.add_box("StreetProps", "mat_stone_mizzi", (26.2, 6.2, bn_o_z + 0.25), (0.65, 2.1, 0.50), rot_z=math.radians(-18.0), uv_scale=0.35)
    sb_o_z = elev_fn(33.2, -1.4)
    collector.add_box("StreetProps", "mat_awning_red", (33.2, -1.4, sb_o_z + 0.42), (0.95, 0.95, 0.84), rot_z=math.radians(15.0), uv_scale=0.45)

    # --- 1B. 1-Storey High-Arched Corner Pavilion (`7012`, `x = 26.8..36.8, y = 11.2..19.5`, seen in `pair_omar_square_north.jpg`) ---
    bz_7012 = elev_fn(31.5, 11.2)
    build_building_from_footprint(
        collector,
        [(26.8, 11.2), (36.8, 11.2), (36.8, 19.5), (26.8, 19.5)],
        base_z=bz_7012,
        levels=1,
        sign_atlas=sign_atlas,
        building_id=7012,
        is_hero=True,
        force_souq_ground=True,
        has_dome=False,
        forced_sign_key="old_city_bazaar",
    )
    # Hang colorful bazaar rugs/textiles under the pavilion awnings (`pair_omar_square_north.jpg`)
    build_souq_display(collector, (31.8, 10.85, bz_7012), (1.0, 0.0), (0.0, -1.0), seed=7012)
    build_souq_display(collector, (35.2, 10.85, bz_7012), (1.0, 0.0), (0.0, -1.0), seed=7013)

    # --- 1B2. New Imperial Hotel 3-Storey Main Block (`7002`, rising behind `7012` and along `x = 36.8..54.5`) ---
    bz_7002 = elev_fn(42.0, 12.2)
    build_building_from_footprint(
        collector,
        [
            (36.8, 12.2),
            (54.5, 12.2),
            (54.5, 27.5),
            (26.8, 27.5),
            (26.8, 19.5),
            (36.8, 19.5),
        ],
        base_z=bz_7002,
        levels=3,
        sign_atlas=sign_atlas,
        building_id=7002,
        is_hero=True,
        force_souq_ground=True,
        has_dome=True,
        forced_sign_key="new_imperial_hotel",
        all_upper_balconies=True,
        rooftop_balustrade=True,
    )
    nih_uv = get_sign_uv(sign_atlas, "new_imperial_hotel")
    build_blade_sign(collector, (26.76, 12.0, bz_7012 + 3.05), (0.0, 1.0), (-1.0, 0.0), width=0.90, height=0.50, sign_uv_rect=nih_uv)

    # Trilingual Street Name Plate on SW corner pier facing Latin Patriarchate Rd
    latin_uv = get_sign_uv(sign_atlas, "latin_rd_plaque")
    build_street_name_plate(collector, (26.76, 12.2, bz_7012 + 2.35), (-1.0, 0.0), latin_uv, width=0.92, height=0.54)

    # --- 1C. Petra Hotel (`7003`, `x = 54.5..80.0, y = 10.5..27.5`, meeting `7002` at `x = 54.5` and `7100` at `x = 80.0` cleanly!) ---
    bz_7003 = elev_fn(66.0, 10.5)
    build_building_from_footprint(
        collector,
        [(54.5, 10.5), (80.0, 10.5), (80.0, 27.5), (54.5, 27.5)],
        base_z=bz_7003,
        levels=3,
        sign_atlas=sign_atlas,
        building_id=7003,
        is_hero=True,
        force_souq_ground=True,
        has_dome=True,
        forced_sign_key="petra_hotel",
        all_upper_balconies=True,
        rooftop_balustrade=True,
    )
    petra_uv = get_sign_uv(sign_atlas, "petra_hotel")
    build_blade_sign(collector, (54.46, 11.2, bz_7003 + 3.05), (0.0, 1.0), (-1.0, 0.0), width=0.90, height=0.50, sign_uv_rect=petra_uv)
    build_street_name_plate(collector, (55.25, 10.46, bz_7003 + 2.45), (0.0, -1.0), omar_uv, width=0.92, height=0.54)

    # --- 1D. Christian Information Centre (`7004`, East of Citadel Moat at `x = 62..82, y = -28.0..-14.0`) ---
    bz_7004 = elev_fn(70.0, -14.0)
    build_building_from_footprint(
        collector,
        [(62.0, -28.0), (82.0, -28.0), (82.0, -14.0), (62.0, -14.0)],
        base_z=bz_7004,
        levels=3,
        sign_atlas=sign_atlas,
        building_id=7004,
        is_hero=True,
        force_souq_ground=False,
        has_dome=True,
        forced_sign_key="christian_info",
    )
    ci_uv = get_sign_uv(sign_atlas, "christian_info")
    build_blade_sign(collector, (61.96, -14.65, bz_7004 + 3.05), (0.0, -1.0), (-1.0, 0.0), width=0.88, height=0.48, sign_uv_rect=ci_uv)

    # --- 2. Narrow `5.6m`-Wide Latin Patriarchate Road (`x = 21.2..26.8, y = 24.5..58.0`) ---
    # Split into lower (`24.5..41.0`) and upper (`41.0..58.0`) blocks so `y = 26.0, 41.0, 56.8` are 100% solid `3.0m` corner piers!
    build_blade_sign(collector, (21.24, 25.25, elev_fn(21.2, 25.25) + 3.05), (0.0, 1.0), (1.0, 0.0), width=0.88, height=0.48, sign_uv_rect=latin_uv)
    build_street_name_plate(collector, (21.24, 25.3, elev_fn(21.2, 25.3) + 2.40), (1.0, 0.0), latin_uv, width=0.90, height=0.52)

    for b_id, ring_l, s_key in (
        (7005, [(8.0, 24.5), (21.2, 24.5), (21.2, 41.0), (8.0, 41.0)], "pilgrim_books"),
        (7015, [(8.0, 41.0), (21.2, 41.0), (21.2, 58.0), (8.0, 58.0)], "st_james_souvenirs"),
        (7006, [(26.8, 27.5), (46.0, 27.5), (46.0, 41.0), (26.8, 41.0)], "jerusalem_pottery"),
        (7016, [(26.8, 41.0), (46.0, 41.0), (46.0, 58.0), (26.8, 58.0)], "swedish_study"),
    ):
        ym_l = 0.5 * (ring_l[0][1] + ring_l[2][1])
        build_building_from_footprint(
            collector,
            ring_l,
            base_z=elev_fn(24.0, ym_l),
            levels=3,
            sign_atlas=sign_atlas,
            building_id=b_id,
            is_hero=True,
            force_souq_ground=True,
            forced_sign_key=s_key,
        )

    # North end of Latin Patriarchate Road (`y = 58.0`) left open to connect directly into `r02_christian_quarter`!

    # Overhead stone `qantara` flying-buttress arches placed strictly at solid `3.0m` building-junction piers (`y = 28.2, 41.0, 56.8`)!
    # Supported by upper-wall stone corbels (`z = az + 4.1m`) so zero ground piers block the street and zero windows/shutters clip!
    for arch_y in (28.2, 41.0, 56.8):
        az = elev_fn(24.0, arch_y)
        span = 5.6
        collector.add_box("Buildings_Details", "mat_stone_boss", (21.38, arch_y, az + 4.15), (0.36, 1.25, 0.65), bottom_face=True)
        collector.add_box("Buildings_Details", "mat_stone_boss", (26.62, arch_y, az + 4.15), (0.36, 1.25, 0.65), bottom_face=True)
        add_arch_spandrel_wall(
            collector,
            "Buildings_Details",
            "mat_stone_ashlar_aged",
            (24.0, arch_y, az + 4.45),
            (1.0, 0.0),
            (0.0, -1.0),
            span_w=span,
            arch_h=1.45,
            z_top=az + 7.15,
            offset_n0=-0.52,
            offset_n1=0.52,
            num_voussoirs=9,
            emit_soffit=False,
        )
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (24.0, arch_y, az + 7.28), (span + 0.08, 1.25, 0.24), bottom_face=True)
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_ashlar_aged",
            (24.0, arch_y, az + 4.45),
            (1.0, 0.0),
            (0.0, -1.0),
            span_w=span,
            arch_h=1.45,
            ring_thickness=0.36,
            offset_n0=-0.58,
            offset_n1=0.58,
            num_voussoirs=9,
            back_face=True,
        )
        build_wall_lantern(collector, (24.0, arch_y, az + 5.45), (1.0, 0.0), (0.0, -1.0))

    # --- 3. Narrow `6.0m`-Wide David Street (`Suq el-Bazar`) Stepped Stone Descent & Bazaar Arcade (`x = 80.2..158.0, y = -1.0..5.0`) ---
    num_steps = 41
    x_start = 80.2
    step_tread = 1.90
    y_south = -0.98
    y_north = 4.98
    step_w = y_north - y_south

    for s_idx in range(num_steps):
        sx0 = x_start + s_idx * step_tread
        sx1 = sx0 + step_tread
        smid = 0.5 * (sx0 + sx1)
        sz_top = elev_fn(smid, 2.0) + (0.02 if s_idx == 0 else 0.05)
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
            (sx1 - 0.08, 2.0, sz_top - 0.05),
            (0.18, step_w, 0.12),
        )

    # Iconic Green-Painted Arched Metal Souq Entrance Trellis Canopy & Hanging Bazaar Carpets (`pair_david_street_souq.jpg`)
    _build_david_st_souq_trellis_and_bazaar(collector, elev_fn)

    souq_terraces = [
        (80.0, 96.0, "old_city_bazaar", "holy_land_rugs"),
        (96.0, 112.0, "jerusalem_pottery", "falafel_cafe"),
        (112.0, 128.0, "st_james_souvenirs", "money_exchange"),
        (128.0, 144.0, "pilgrim_books", "swedish_study"),
        (144.0, 158.0, "old_city_bazaar", "holy_land_rugs"),
    ]
    david_plaque_uv = get_sign_uv(sign_atlas, "david_st_plaque")
    bz_david0 = elev_fn(80.5, 2.0)
    build_blade_sign(collector, (80.65, 4.95, bz_david0 + 3.05), (1.0, 0.0), (0.0, -1.0), width=0.88, height=0.48, sign_uv_rect=david_plaque_uv)
    build_street_name_plate(collector, (80.75, 4.96, bz_david0 + 2.40), (0.0, -1.0), david_plaque_uv, width=0.92, height=0.54)
    build_street_name_plate(collector, (80.75, -0.96, bz_david0 + 2.40), (0.0, 1.0), david_plaque_uv, width=0.92, height=0.54)

    for t_idx, (tx0, tx1, north_sign, south_sign) in enumerate(souq_terraces):
        t_mid = 0.5 * (tx0 + tx1)
        bz = elev_fn(t_mid, 2.0)
        north_y1 = 10.5 if t_idx == 0 else 13.5
        build_building_from_footprint(
            collector,
            [(tx0, 5.0), (tx1, 5.0), (tx1, north_y1), (tx0, north_y1)],
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
            [(tx0, -9.5), (tx1, -9.5), (tx1, -1.0), (tx0, -1.0)],
            base_z=bz,
            levels=3,
            sign_atlas=sign_atlas,
            building_id=7101 + t_idx * 2,
            is_hero=True,
            force_souq_ground=True,
            has_dome=(t_idx % 2 == 1),
            forced_sign_key=south_sign,
        )

    # Overhead stone flying-buttress arches placed strictly at solid `3.0m` building-junction piers (`x = 96.0, 112.0, 128.0, 144.0, 156.8`)
    # where `corner_margin = 1.50m` guarantees 100% solid stone masonry with ZERO windows, shutters, or balconies!
    for arch_x in (96.0, 112.0, 128.0, 144.0, 156.8):
        az = elev_fn(arch_x, 2.0)
        span_d = 6.0
        collector.add_box("Buildings_Details", "mat_stone_boss", (arch_x, -0.82, az + 4.05), (1.25, 0.36, 0.65), bottom_face=True)
        collector.add_box("Buildings_Details", "mat_stone_boss", (arch_x, 4.82, az + 4.05), (1.25, 0.36, 0.65), bottom_face=True)
        add_arch_spandrel_wall(
            collector,
            "Buildings_Details",
            "mat_stone_ashlar_aged",
            (arch_x, 2.0, az + 4.35),
            (0.0, 1.0),
            (-1.0, 0.0),
            span_w=span_d,
            arch_h=1.45,
            z_top=az + 7.05,
            offset_n0=-0.52,
            offset_n1=0.52,
            num_voussoirs=9,
            emit_soffit=False,
        )
        collector.add_box("Buildings_Details", "mat_stone_mizzi", (arch_x, 2.0, az + 7.18), (1.25, span_d + 0.08, 0.24), bottom_face=True)
        add_arch_voussoirs(
            collector,
            "Buildings_Details",
            "mat_stone_ashlar_aged",
            (arch_x, 2.0, az + 4.35),
            (0.0, 1.0),
            (-1.0, 0.0),
            span_w=span_d,
            arch_h=1.45,
            ring_thickness=0.36,
            offset_n0=-0.58,
            offset_n1=0.58,
            num_voussoirs=9,
            back_face=True,
        )
        build_wall_lantern(collector, (arch_x, 2.0, az + 5.35), (0.0, 1.0), (-1.0, 0.0))

    # --- 4. Western Mamilla Arcade Terrace & Hinnom Valley Retaining Wall Parapet ---
    # 4a. Continuous Stepped Limestone Retaining Wall & Parapet along the west edge of the Breach/Moat Road (`x = -2.5, y = -76..-14`)
    # and the West Plaza terrace (`(-38, -12) -> (-16, -9.5) -> (-2.5, -15.5)`) so there is zero empty road drop-off!
    west_road_wall_segs = [
        ((-42.0, -14.0), (-16.0, -9.5)),
        ((-16.0, -9.5), (-2.5, -15.5)),
        ((-2.5, -15.5), (-2.5, -76.0)),
    ]
    for wp0, wp1 in west_road_wall_segs:
        dx, dy = wp1[0] - wp0[0], wp1[1] - wp0[1]
        sl = math.hypot(dx, dy)
        nx, ny = -dy / sl, dx / sl
        z_surf = elev_fn(0.5 * (wp0[0] + wp1[0]), 0.5 * (wp0[1] + wp1[1]))
        collector.add_oriented_box("CityWalls", "mat_stone_boss", wp0, wp1, -6.0, z_surf + 0.85, -0.75, 0.0, (nx, ny), cap_back=True)
        collector.add_oriented_box("CityWalls", "mat_stone_mizzi", wp0, wp1, z_surf + 0.85, z_surf + 0.96, -0.82, 0.05, (nx, ny), cap_back=True)

    # West Plaza Sunken Jaffa Road Retaining Wall, Blind Stone Arches & Pedestrian Guardrail (`(-35.0, -4.0) -> (-62.0, 12.0)`, `jaffa_gate_1_rev`)
    for wg_i in range(6):
        wt0 = wg_i / 6.0
        wt1 = (wg_i + 1) / 6.0
        wx0 = -35.0 + (-62.0 - -35.0) * wt0
        wy0 = -4.0 + (12.0 - -4.0) * wt0
        wx1 = -35.0 + (-62.0 - -35.0) * wt1
        wy1 = -4.0 + (12.0 - -4.0) * wt1
        wmx, wmy = 0.5 * (wx0 + wx1), 0.5 * (wy0 + wy1)
        wz_s = elev_fn(wmx, wmy)
        collector.add_oriented_box("CityWalls", "mat_stone_mizzi", (wx0, wy0), (wx1, wy1), wz_s - 2.5, wz_s + 0.92, -0.35, 0.35, (0.51, 0.86), cap_back=True)
        collector.add_oriented_box("StreetProps", "mat_iron_dark", (wx0, wy0), (wx1, wy1), wz_s + 1.25, wz_s + 1.32, -0.04, 0.04, (0.51, 0.86), cap_back=True)

    # 4b. Detailed LOD 1 Mamilla Alrov Limestone Arcade Buildings overlooking the West Plaza (`x = -82..-66`)
    collector.add_box("CityWalls", "mat_stone_rubble", (-96.0, 0.0, -0.5), (18.0, 185.0, 9.5))
    for blk_idx, (by0, by1, b_lvls) in enumerate(((-52.0, -28.0, 3), (-24.0, -2.0, 3), (2.0, 24.0, 2), (28.0, 50.0, 3))):
        ring_mamilla = [(-80.0, by0), (-66.0, by0), (-66.0, by1), (-80.0, by1)]
        bz_m = elev_fn(-66.0, 0.5 * (by0 + by1))
        for e_i in range(4):
            mp0 = ring_mamilla[e_i]
            mp1 = ring_mamilla[(e_i + 1) % 4]
            mdx, mdy = mp1[0] - mp0[0], mp1[1] - mp0[1]
            mlen = math.hypot(mdx, mdy)
            mnx, mny = mdy / mlen, -mdx / mlen
            _build_lod1_wall(
                collector,
                mp0,
                mp1,
                base_z=bz_m,
                ground_h=4.0,
                upper_floor_h=3.3,
                levels=b_lvls,
                normal_xy=(mnx, mny),
                ground_pier_mat="mat_stone_boss",
                upper_wall_mat="mat_stone_ashlar",
                trim_stone_mat="mat_stone_mizzi",
                building_id=7300 + blk_idx,
                edge_idx=e_i,
            )
        m_roof_z = bz_m + 4.0 + (b_lvls - 1) * 3.3 + 0.15
        collector.add_polygon_3d("Buildings_Walls", "mat_plaster", [(p[0], p[1], m_roof_z) for p in ring_mamilla], uv_scale=0.25)

    # Keep only the western extramural Hinnom slope backdrop (`x < 0`), leaving `x >= 8` (`Maronite Convent St` & `r04_armenian_quarter`) 100% open!
    collector.add_box("CityWalls", "mat_stone_boss", (-18.0, -84.0, 2.8), (34.0, 14.0, 17.6))
