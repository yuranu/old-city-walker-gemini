#!/usr/bin/env python3
"""
Clean-Room Headless Blender 3D City, Landmark, Facade & Street-Prop Builder
for the Jaffa Gate District, Old City of Jerusalem (`pipeline/build_city.py`).

Pass 2 Upgrades:
  - Multi-texture Jerusalem limestone & paving variety (`mat_stone_ashlar`, `mat_stone_mizzi`,
    `mat_stone_ashlar_aged`, `mat_stone_boss`, `mat_stone_rubble`, `mat_paving`,
    `mat_paving_flagstone`, `mat_steel_shutters`)
  - Muted, realistic heritage accent colors (zero cartoonish oversaturation)
  - Per-vertex baked Ambient Occlusion & wall-base rising damp/dirt (`COLOR_0` exported via
    `export_vertex_color='ACTIVE'`)
  - Widened 13.5m Citadel Sunken Dry Moat cutout (`x in [10.2..53.8], y in [-55.8..-15.4]`)
  - Street trash scatter, curb grit patches, utility boxes, and trilingual street name plates
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bmesh
import bpy

PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from arch_rules import (
    build_building_from_footprint,
    build_citadel_and_tower_of_david,
    build_hero_omar_square_and_david_street,
    build_jaffa_gate_monument,
    corridor_distance,
    ensure_ccw_2d,
)
from props_lib import (
    MeshCollector,
    build_bollard,
    build_cypress_tree,
    build_garbage_pile,
    build_iron_railing,
    build_olive_tree,
    build_street_bin,
    build_street_lamp,
    build_street_trash_scatter,
    build_wheelie_bin,
)


# ==============================================================================
# 1. CONTINUOUS ELEVATION MODEL (Synchronized with `fetch_osm_and_dem.py`)
# ==============================================================================

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 == edge0:
        return 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def street_elevation_z(x: float, y: float) -> float:
    """
    Continuous street-level elevation z(x, y) in metres relative to Jaffa Gate (766.0m ASL).
    """
    z_base = 0.0

    if x < -10.0:
        w_t = smoothstep(-10.0, -90.0, x)
        z_base -= 2.5 * w_t
        if y < -10.0:
            sw_t = smoothstep(-10.0, -80.0, y) * smoothstep(-5.0, -60.0, x)
            z_base -= 1.8 * sw_t

    if x >= -5.0:
        omar_x = smoothstep(-5.0, 22.0, x) * (1.0 - smoothstep(75.0, 105.0, x))
        omar_y = 1.0 - smoothstep(10.0, 45.0, abs(y + 2.0))
        crown = 0.35 + 0.45 * smoothstep(15.0, 65.0, x)
        z_base += crown * omar_x * omar_y

    if x > 75.0:
        david_x_in = smoothstep(75.0, 88.0, x)
        david_prog = smoothstep(85.0, 160.0, x)
        extra_east = max(0.0, (x - 160.0) / 60.0) * 1.2 if x > 160.0 else 0.0
        z_david = 0.5 - 4.0 * david_prog - extra_east
        corridor_y = 1.0 - 0.35 * smoothstep(12.0, 80.0, abs(y - 4.0))
        z_base = (1.0 - david_x_in) * z_base + david_x_in * (z_david * corridor_y)

    if y > 8.0 and x > -15.0:
        north_t = smoothstep(8.0, 85.0, y)
        east_fade = 1.0 - 0.5 * smoothstep(65.0, 140.0, x)
        west_in = smoothstep(-15.0, 10.0, x)
        z_base += 2.2 * north_t * east_fade * west_in

    if y < -15.0 and x > 35.0:
        south_t = smoothstep(-15.0, -90.0, y) * smoothstep(35.0, 75.0, x)
        z_base += 1.2 * south_t

    gate_pin = (1.0 - smoothstep(4.0, 16.0, abs(x + 6.0))) * (1.0 - smoothstep(3.0, 9.0, abs(y - 3.0)))
    z_base = (1.0 - gate_pin) * z_base + gate_pin * 0.0

    return round(z_base, 3)


# ==============================================================================
# 2. BLENDER SCENE CLEAR & PRINCIPLED BSDF PBR MATERIAL WIRING
# ==============================================================================

def clear_blender_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def _load_image(tex_dir: Path, filename: str, fallback_filename: Optional[str] = None, is_data: bool = False) -> Optional[bpy.types.Image]:
    path = tex_dir / filename
    if not path.exists() and fallback_filename is not None:
        path = tex_dir / fallback_filename
    if not path.exists():
        print(f"[WARN] Texture not found: {path}")
        return None
    img = bpy.data.images.load(str(path), check_existing=True)
    img.colorspace_settings.name = "Non-Color" if is_data else "sRGB"
    return img


def make_pbr_textured_material(
    name: str,
    tex_dir: Path,
    prefix: str,
    normal_strength: float = 1.15,
    metallic: float = 0.0,
    fallback_prefix: str = "stone_ashlar",
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (400, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (80, 0)
    bsdf.inputs["Metallic"].default_value = metallic
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    diff_img = _load_image(tex_dir, f"{prefix}_diff.jpg", f"{fallback_prefix}_diff.jpg", is_data=False)
    if diff_img is not None:
        tex_diff = nodes.new(type="ShaderNodeTexImage")
        tex_diff.image = diff_img
        tex_diff.location = (-350, 220)
        links.new(tex_diff.outputs["Color"], bsdf.inputs["Base Color"])

    rough_img = _load_image(tex_dir, f"{prefix}_rough.jpg", f"{fallback_prefix}_rough.jpg", is_data=True)
    if rough_img is not None:
        tex_rough = nodes.new(type="ShaderNodeTexImage")
        tex_rough.image = rough_img
        tex_rough.location = (-350, -40)
        links.new(tex_rough.outputs["Color"], bsdf.inputs["Roughness"])

    norm_img = _load_image(tex_dir, f"{prefix}_norm.jpg", f"{fallback_prefix}_norm.jpg", is_data=True)
    if norm_img is not None:
        tex_norm = nodes.new(type="ShaderNodeTexImage")
        tex_norm.image = norm_img
        tex_norm.location = (-550, -320)
        norm_map = nodes.new(type="ShaderNodeNormalMap")
        norm_map.location = (-220, -320)
        norm_map.inputs["Strength"].default_value = normal_strength
        links.new(tex_norm.outputs["Color"], norm_map.inputs["Color"])
        links.new(norm_map.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def make_alpha_cutout_material(
    name: str,
    tex_dir: Path,
    filename: str,
    roughness: float = 0.65,
    alpha_threshold: float = 0.35,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_backface_culling = False
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    if hasattr(mat, "alpha_threshold"):
        try:
            mat.alpha_threshold = alpha_threshold
        except Exception:
            pass

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (400, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (80, 0)
    bsdf.inputs["Roughness"].default_value = roughness
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    img = _load_image(tex_dir, filename, is_data=False)
    if img is not None:
        tex_node = nodes.new(type="ShaderNodeTexImage")
        tex_node.image = img
        tex_node.location = (-320, 100)
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])

    return mat


def make_solid_material(
    name: str,
    base_color: Tuple[float, float, float],
    roughness: float = 0.5,
    metallic: float = 0.0,
    emission_color: Optional[Tuple[float, float, float]] = None,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (base_color[0], base_color[1], base_color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission_color is not None and emission_strength > 0.0:
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (
                emission_color[0],
                emission_color[1],
                emission_color[2],
                1.0,
            )
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def create_all_materials(tex_dir: Path) -> Dict[str, bpy.types.Material]:
    mats: Dict[str, bpy.types.Material] = {}

    # 1. Core & New Multi-Variety PBR Texture Sets
    mats["mat_stone_ashlar"] = make_pbr_textured_material("mat_stone_ashlar", tex_dir, "stone_ashlar", normal_strength=1.1)
    mats["mat_stone_ashlar_aged"] = make_pbr_textured_material(
        "mat_stone_ashlar_aged", tex_dir, "stone_ashlar_aged", normal_strength=1.2, fallback_prefix="stone_ashlar"
    )
    mats["mat_stone_mizzi"] = make_pbr_textured_material(
        "mat_stone_mizzi", tex_dir, "stone_mizzi", normal_strength=1.0, fallback_prefix="stone_ashlar"
    )
    mats["mat_stone_boss"] = make_pbr_textured_material("mat_stone_boss", tex_dir, "stone_boss", normal_strength=1.25)
    mats["mat_stone_rubble"] = make_pbr_textured_material("mat_stone_rubble", tex_dir, "stone_rubble", normal_strength=1.25)
    mats["mat_paving"] = make_pbr_textured_material("mat_paving", tex_dir, "paving_setts", normal_strength=1.1)
    mats["mat_paving_flagstone"] = make_pbr_textured_material(
        "mat_paving_flagstone", tex_dir, "paving_flagstone", normal_strength=1.0, fallback_prefix="paving_setts"
    )
    mats["mat_steel_shutters"] = make_pbr_textured_material(
        "mat_steel_shutters", tex_dir, "steel_shutters", normal_strength=1.3, metallic=0.35, fallback_prefix="wood_doors"
    )
    mats["mat_plaster"] = make_pbr_textured_material("mat_plaster", tex_dir, "plaster_warm", normal_strength=1.1)
    mats["mat_roof_tile"] = make_pbr_textured_material("mat_roof_tile", tex_dir, "roof_tile", normal_strength=1.2)
    mats["mat_wood_doors"] = make_pbr_textured_material("mat_wood_doors", tex_dir, "wood_doors", normal_strength=1.2)
    mats["mat_bark"] = make_pbr_textured_material("mat_bark", tex_dir, "bark", normal_strength=1.5)

    # 2. Alpha Cutout Botanical & Wrought-Iron Materials
    mats["mat_foliage_olive"] = make_alpha_cutout_material("mat_foliage_olive", tex_dir, "foliage_olive.png", roughness=0.68)
    mats["mat_foliage_cypress"] = make_alpha_cutout_material("mat_foliage_cypress", tex_dir, "foliage_cypress.png", roughness=0.70)
    mats["mat_iron_grille"] = make_alpha_cutout_material("mat_iron_grille", tex_dir, "iron_grille.png", roughness=0.42)

    # 3. Multilingual Sign Atlas Material
    mat_signs = bpy.data.materials.new(name="mat_signs")
    s_nodes = mat_signs.node_tree.nodes
    s_links = mat_signs.node_tree.links
    s_nodes.clear()
    s_out = s_nodes.new(type="ShaderNodeOutputMaterial")
    s_bsdf = s_nodes.new(type="ShaderNodeBsdfPrincipled")
    s_bsdf.inputs["Roughness"].default_value = 0.45
    s_links.new(s_bsdf.outputs["BSDF"], s_out.inputs["Surface"])
    sign_img = _load_image(tex_dir, "sign_atlas.png", is_data=False)
    if sign_img is not None:
        s_tex = s_nodes.new(type="ShaderNodeTexImage")
        s_tex.image = sign_img
        s_links.new(s_tex.outputs["Color"], s_bsdf.inputs["Base Color"])
    mats["mat_signs"] = mat_signs

    # 4. Muted, Dusty Heritage Accent Materials (No cartoonish oversaturation)
    mats["mat_iron_dark"] = make_solid_material("mat_iron_dark", (0.11, 0.12, 0.13), roughness=0.42, metallic=0.75)
    mats["mat_trash_black"] = make_solid_material("mat_trash_black", (0.05, 0.055, 0.06), roughness=0.26, metallic=0.04)
    mats["mat_trash_blue"] = make_solid_material("mat_trash_blue", (0.08, 0.16, 0.29), roughness=0.28, metallic=0.04)
    mats["mat_cardboard"] = make_solid_material("mat_cardboard", (0.44, 0.34, 0.22), roughness=0.85)
    mats["mat_terracotta"] = make_solid_material("mat_terracotta", (0.48, 0.24, 0.15), roughness=0.76)
    mats["mat_awning_red"] = make_solid_material("mat_awning_red", (0.38, 0.11, 0.09), roughness=0.82)
    mats["mat_awning_green"] = make_solid_material("mat_awning_green", (0.12, 0.24, 0.15), roughness=0.82)
    mats["mat_awning_striped"] = make_solid_material("mat_awning_striped", (0.76, 0.71, 0.61), roughness=0.80)
    mats["mat_corrugated_metal"] = make_solid_material("mat_corrugated_metal", (0.42, 0.44, 0.46), roughness=0.52, metallic=0.55)
    mats["mat_lantern_glow"] = make_solid_material(
        "mat_lantern_glow",
        (0.98, 0.84, 0.50),
        roughness=0.18,
        emission_color=(0.98, 0.80, 0.42),
        emission_strength=3.2,
    )

    return mats


# ==============================================================================
# 3. GROUND PAVING MESH, SIDEWALK KERBS & STREET FURNITURE PLACEMENT
# ==============================================================================

def build_continuous_ground_mesh(collector: MeshCollector) -> None:
    """
    Builds a continuous, walkable stone-paved ground surface across [-170..200] x [-170..170]m
    using a 2-tier adaptive resolution grid:
      - Core walkable zone (`-60 <= x <= 130` and `-60 <= y <= 55`): fine `2.5m` grid spacing
        with an exact cutout for the widened Citadel Sunken Dry Moat (`x in [10.2..53.8],
        y in [-55.8..-15.4]`, keeping the Moat Bridge strip `x in [41.8..47.2]` solid).
      - Outer surround zone: coarse `10.0m` grid spacing (~3,500–4,500 faces total instead of 25,000!).
    Alternates between `mat_paving` (cobble setts for roadways) and `mat_paving_flagstone`
    (smooth dressed Jerusalem flagstones for pedestrian plazas, Jaffa Gate passage, and souqs).
    """
    def is_in_moat_cutout(cx: float, cy: float) -> bool:
        if 10.2 < cx < 53.8 and -55.8 < cy < -15.4:
            if 41.8 <= cx <= 47.2 and -29.0 <= cy <= -15.0:
                return False
            return True
        return False

    def emit_ground_cell(x0: float, y0: float, x1: float, y1: float) -> None:
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        if is_in_moat_cutout(xm, ym):
            return

        z00 = street_elevation_z(x0, y0)
        z10 = street_elevation_z(x1, y0)
        z11 = street_elevation_z(x1, y1)
        z01 = street_elevation_z(x0, y1)

        is_paved = (-125.0 <= xm <= 175.0 and -98.0 <= ym <= 115.0)
        if not is_paved:
            mat = "mat_stone_rubble"
            uv_sc = 0.22
        else:
            is_roadway = (-14.0 <= xm <= 56.0 and -14.5 <= ym <= 8.5) and not (-9.0 <= xm <= 5.0 and -1.0 <= ym <= 6.5)
            mat = "mat_paving" if is_roadway else "mat_paving_flagstone"
            uv_sc = 0.42 if is_roadway else 0.36

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

    # 1. Core Walkable Zone (`-60 <= x <= 130`, `-60 <= y <= 55`) at fine `2.5m` spacing
    core_step = 2.5
    core_xs = [-60.0 + i * core_step for i in range(int(190.0 / core_step) + 1)]
    core_ys = [-60.0 + j * core_step for j in range(int(115.0 / core_step) + 1)]
    for j in range(len(core_ys) - 1):
        y0, y1 = core_ys[j], core_ys[j + 1]
        for i in range(len(core_xs) - 1):
            x0, x1 = core_xs[i], core_xs[i + 1]
            emit_ground_cell(x0, y0, x1, y1)

    # 2. Outer Surround Zone (`-170..200` x `-170..170` outside `-60..130` x `-60..55`) at `10.0m` coarse spacing
    outer_xs = [-170.0 + i * 10.0 for i in range(38)]
    outer_ys = [-170.0 + j * 10.0 for j in range(11)] + [-60.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 65.0] + [70.0 + j * 10.0 for j in range(11)]
    for j in range(len(outer_ys) - 1):
        y0, y1 = outer_ys[j], outer_ys[j + 1]
        ym = 0.5 * (y0 + y1)
        for i in range(len(outer_xs) - 1):
            x0, x1 = outer_xs[i], outer_xs[i + 1]
            xm = 0.5 * (x0 + x1)
            if -60.0 < xm < 130.0 and -60.0 < ym < 55.0:
                continue
            emit_ground_cell(x0, y0, x1, y1)

    # Raised limestone sidewalk kerbs along West Plaza, 1898 Breach & Omar Square promenade
    kerb_runs = [
        ((-55.0, 7.5), (-9.0, 7.5), 1.8),
        ((-55.0, -7.5), (-9.0, -7.5), 1.8),
        ((10.5, -14.8), (41.5, -14.8), 2.2),
        ((18.0, 13.8), (80.0, 13.8), 1.6),
    ]
    for (kx0, ky0), (kx1, ky1), kw in kerb_runs:
        segs = 6
        for s in range(segs):
            t0 = s / segs
            t1 = (s + 1) / segs
            px0 = kx0 + (kx1 - kx0) * t0
            py0 = ky0 + (ky1 - ky0) * t0
            px1 = kx0 + (kx1 - kx0) * t1
            py1 = ky0 + (ky1 - ky0) * t1
            zm = street_elevation_z(0.5 * (px0 + px1), 0.5 * (py0 + py1))
            collector.add_oriented_box(
                "Ground_Paving",
                "mat_stone_ashlar_aged",
                (px0, py0),
                (px1, py1),
                zm - 0.15,
                zm + 0.14,
                -kw * 0.5,
                kw * 0.5,
                (0.0, 1.0),
                uv_scale=0.35,
                cap_back=True,
                cap_ends=(s == 0 or s == segs - 1),
            )


def populate_street_props_and_trees(
    collector: MeshCollector,
    osm_trees: List[dict],
) -> None:
    """
    Populates geo-registered trees, ornate street lamps, bollards, garbage bag clusters,
    street trash/dirt scatters, wheelie bins, and cast-iron litter bins.
    """
    # 1. Hero Curated Trees (Citadel Moat Promenade, Omar Square, West Plaza, Latin Patriarchate Rd)
    hero_trees = [
        (16.5, -12.2, "olive", 6.8, 1),
        (24.5, -13.0, "cypress", 9.4, 2),
        (33.2, -13.2, "olive", 7.0, 3),
        (40.2, -13.0, "cypress", 9.0, 4),
        (51.5, -12.5, "olive", 6.9, 5),
        (29.0, 5.2, "olive", 7.4, 6),
        (55.5, 4.8, "olive", 6.6, 7),
        (-22.0, -9.5, "olive", 6.5, 8),
        (-28.5, 11.5, "cypress", 9.0, 9),
        (-36.0, -14.0, "olive", 6.8, 10),
        (21.2, 33.0, "cypress", 8.5, 11),
    ]
    placed_xy: List[Tuple[float, float]] = []

    for tx, ty, species, th, seed in hero_trees:
        tz = street_elevation_z(tx, ty)
        if species == "cypress":
            build_cypress_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)
        else:
            build_olive_tree(collector, tx, ty, tz, height=th, seed=seed, with_kerb=True)
        placed_xy.append((tx, ty))

    for idx, t in enumerate(osm_trees):
        tx, ty = float(t["x"]), float(t["y"])
        if corridor_distance(tx, ty) > 55.0:
            continue
        if -15.0 <= tx <= 14.0 and -16.0 <= ty <= 12.0:
            continue
        if 8.0 <= tx <= 64.0 and -85.0 <= ty <= -14.0:
            continue
        if 72.0 <= tx <= 168.0 and -12.0 <= ty <= 16.0:
            continue
        if any(math.hypot(tx - px, ty - py) < 8.5 for px, py in placed_xy):
            continue
        tz = street_elevation_z(tx, ty)
        sp_str = str(t.get("species", "")).lower()
        if "cupressus" in sp_str or "cypress" in sp_str or idx % 3 == 1:
            build_cypress_tree(collector, tx, ty, tz, height=8.8, seed=200 + idx, with_kerb=True)
        else:
            build_olive_tree(collector, tx, ty, tz, height=6.5, seed=200 + idx, with_kerb=True)
        placed_xy.append((tx, ty))
        if len(placed_xy) >= 28:
            break

    # 2. Limestone & Cast-Iron Bollards
    bollard_coords = [
        (-13.5, -2.5, "stone"),
        (-13.5, -0.5, "stone"),
        (-13.5, 5.8, "stone"),
        (-2.5, -7.5, "stone"),
        (0.5, -9.0, "stone"),
        (3.5, -10.5, "stone"),
        (6.8, -12.0, "stone"),
        (9.5, -13.2, "stone"),
        (19.5, 12.8, "iron"),
        (26.0, 12.8, "iron"),
        (34.0, 12.8, "iron"),
        (42.0, 12.8, "iron"),
        (50.0, 12.8, "iron"),
        (62.0, 10.5, "iron"),
        (76.5, -1.2, "stone"),
        (76.5, 5.2, "stone"),
    ]
    for bx, by, b_style in bollard_coords:
        bz = street_elevation_z(bx, by)
        build_bollard(collector, bx, by, bz, style=b_style)

    # 3. Ornate Jerusalem Street Lamps
    lamp_coords = [
        (-24.0, 7.2),
        (-14.5, 7.2),
        (-14.5, -6.5),
        (5.5, 7.8),
        (9.2, -13.8),
        (20.0, -13.8),
        (34.0, -13.8),
        (48.0, -13.8),
        (21.0, 13.2),
        (38.0, 13.2),
        (58.0, 11.2),
        (75.0, 8.5),
        (20.5, 29.0),
        (32.5, 42.0),
    ]
    for lx, ly in lamp_coords:
        lz = street_elevation_z(lx, ly)
        build_street_lamp(collector, lx, ly, lz, height=4.4)

    # 4. Garbage Bag Clusters & Street Trash/Dirt Scatters
    garbage_spots = [
        (5.2, 9.4, 0.3, 4),
        (21.4, 14.2, -0.2, 4),
        (50.4, 14.4, 0.8, 5),
        (63.4, 11.2, 0.1, 4),
        (56.6, -17.8, 1.2, 4),
        (-8.8, -5.8, -0.5, 3),
        (20.1, 26.5, 0.4, 4),
        (83.5, 5.3, 0.0, 4),
        (97.5, -1.3, 1.5, 4),
        (114.5, 5.3, 0.2, 3),
        (131.0, -1.3, -0.4, 4),
    ]
    for idx, (gx, gy, grot, gcnt) in enumerate(garbage_spots):
        gz = street_elevation_z(gx, gy)
        build_garbage_pile(collector, gx, gy, gz, rot_z=grot, bag_count=gcnt, seed=400 + idx)

    # Street trash scatters (swept curb dirt patches, flattened cardboard, wrappers, cans, bags)
    scatter_spots = [
        (-11.0, -5.5, 2.0),    # West Plaza south curb
        (-10.5, 6.2, 1.8),     # West Plaza north curb near Jaffa Gate
        (-3.2, 0.9, 1.4),      # Jaffa Gate L-passage inner corner
        (4.5, -6.2, 1.8),      # South exit of Jaffa Gate into Breach
        (14.5, 9.5, 2.1),      # Tourist Info curb on Omar Square
        (24.0, 14.0, 2.0),     # New Imperial Hotel curb
        (46.0, 14.0, 2.2),     # East end of New Imperial Hotel
        (66.0, 11.0, 2.0),     # Petra Hotel curb
        (78.5, 5.2, 1.8),      # David Street Souq entrance north curb
        (78.5, -1.5, 1.8),     # David Street Souq entrance south curb
        (91.0, 5.2, 1.7),      # David Street steps north gutter
        (106.0, -1.4, 1.7),    # David Street steps south gutter
        (122.0, 5.2, 1.7),     # Lower David Street north gutter
        (20.2, 32.0, 1.8),     # Latin Patriarchate Rd west gutter
        (33.0, 39.0, 1.8),     # Latin Patriarchate Rd east gutter
    ]
    for s_idx, (sx, sy, srad) in enumerate(scatter_spots):
        sz = street_elevation_z(sx, sy)
        build_street_trash_scatter(collector, sx, sy, sz, radius=srad, seed=600 + s_idx)

    wheelie_spots = [
        (19.2, 10.2, 0.2),
        (51.2, 15.2, -0.4),
        (79.2, -4.8, 0.8),
        (20.2, 38.0, 0.0),
    ]
    for wx, wy, wrot in wheelie_spots:
        wz = street_elevation_z(wx, wy)
        build_wheelie_bin(collector, wx, wy, wz, rot_z=wrot)

    bin_spots = [
        (-11.5, 6.2),
        (6.2, 6.5),
        (27.5, 13.2),
        (45.5, 13.2),
        (68.0, 10.5),
        (78.5, 5.5),
        (105.0, 5.4),
    ]
    for sbx, sby in bin_spots:
        sbz = street_elevation_z(sbx, sby)
        build_street_bin(collector, sbx, sby, sbz)


# ==============================================================================
# 4. OSM FOOTPRINT FILTERING & FULL SCENE ORCHESTRATION
# ==============================================================================

def should_skip_osm_building(b: dict) -> bool:
    osm_id = int(b.get("osm_id", 0))
    tags = b.get("tags", {})
    if osm_id in (5862586, 7880713, 136164, 1393747403, 1392276721, 126086870, 158063844, 158063849):
        return True
    if "building" not in tags:
        return True
    if tags.get("type") == "boundary" or "admin_level" in tags or "water" in tags:
        return True

    ring = b.get("ring", [])
    if len(ring) >= 3:
        rxs = [float(p[0]) for p in ring]
        rys = [float(p[1]) for p in ring]
        if (max(rxs) - min(rxs)) > 75.0 or (max(rys) - min(rys)) > 75.0:
            return True

    cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
    if corridor_distance(cx, cy) > 95.0:
        return True

    # 1. Jaffa Gate & 1898 Breach zone
    if -18.0 <= cx <= 16.0 and -18.0 <= cy <= 14.0:
        return True
    # 2. Citadel, Widened Dry Moat & Moved Christian Information Centre zone
    if 6.0 <= cx <= 84.0 and -88.0 <= cy <= -10.0:
        return True
    # 3. Omar Square open plaza & Hero Hotel footprints
    if 4.0 <= cx <= 88.0 and -22.0 <= cy <= 29.0:
        return True
    # 4. Latin Patriarchate Rd lower hero corridor
    if 7.0 <= cx <= 49.0 and 24.0 <= cy <= 57.0:
        return True
    # 5. David Street stepped souq corridor (`x = 76..170, y = -13..17`)
    if 76.0 <= cx <= 170.0 and -13.0 <= cy <= 17.0:
        return True

    for pt in b.get("ring", []):
        px, py = float(pt[0]), float(pt[1])
        if -12.0 <= px <= 78.0 and -15.5 <= py <= 13.5:
            return True
        if 76.0 <= px <= 166.0 and -4.5 <= py <= 8.5:
            return True
        if 20.0 <= px <= 33.0 and 12.0 <= py <= 65.0:
            return True

    return False


def patch_glb_alpha_modes(glb_path: Path) -> None:
    """
    Ensures alpha-cutout materials (`mat_foliage_olive`, `mat_foliage_cypress`, `mat_iron_grille`)
    are marked with `"alphaMode": "MASK"` and `"alphaCutoff": 0.35` in the glTF 2.0 JSON chunk.
    """
    import struct

    data = glb_path.read_bytes()
    if len(data) < 20:
        return
    magic, version, total_len = struct.unpack("<4sII", data[:12])
    if magic != b"glTF":
        return
    json_len, json_type = struct.unpack("<II", data[12:20])
    json_bytes = data[20 : 20 + json_len]
    bin_rest = data[20 + json_len :]

    gltf = json.loads(json_bytes.decode("utf-8"))
    mask_mats = {"mat_foliage_olive", "mat_foliage_cypress", "mat_iron_grille"}
    for m in gltf.get("materials", []):
        if m.get("name") in mask_mats:
            m["alphaMode"] = "MASK"
            m["alphaCutoff"] = 0.35
            m["doubleSided"] = True

    new_json_raw = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad = (4 - (len(new_json_raw) % 4)) % 4
    new_json_padded = new_json_raw + (b" " * pad)
    new_total_len = 12 + 8 + len(new_json_padded) + len(bin_rest)

    out = bytearray()
    out += struct.pack("<4sII", b"glTF", version, new_total_len)
    out += struct.pack("<II", len(new_json_padded), json_type)
    out += new_json_padded
    out += bin_rest
    glb_path.write_bytes(bytes(out))


def main() -> None:
    t0 = time.time()
    osm_path = ROOT_DIR / "data" / "osm_jaffa.json"
    tex_dir = ROOT_DIR / "public" / "textures"
    sign_atlas_path = tex_dir / "sign_atlas.json"
    out_glb_path = ROOT_DIR / "public" / "models" / "jaffa_gate_city.glb"
    out_manifest_path = ROOT_DIR / "public" / "data" / "scene_manifest.json"

    out_glb_path.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if osm_path.exists():
        with open(osm_path, "r", encoding="utf-8") as f:
            osm_data = json.load(f)
    else:
        print(f"[WARN] {osm_path} not found, building with empty OSM list.")
        osm_data = {"buildings": [], "streets": [], "walls": [], "trees": [], "pois": []}

    if sign_atlas_path.exists():
        with open(sign_atlas_path, "r", encoding="utf-8") as f:
            sign_atlas = json.load(f)
    else:
        sign_atlas = {}

    clear_blender_scene()
    materials = create_all_materials(tex_dir)
    collector = MeshCollector()

    print("[1/5] Building continuous stone-paved ground mesh (flagstones + cobble setts) & kerbs...")
    build_continuous_ground_mesh(collector)

    print("[2/5] Building Jaffa Gate (Bab el-Khalil), Sealed Vaulted L-Turn Passage & City Wall...")
    build_jaffa_gate_monument(collector, sign_atlas)

    print("[3/5] Building Citadel, Widened 13.5m Sunken Dry Moat, Glacis, Phasael Tower & Minaret...")
    build_citadel_and_tower_of_david(collector, sign_atlas)

    print("[4/5] Building Hero Omar Square Hotels, Latin Patriarchate Rd, David St & West Horizon...")
    build_hero_omar_square_and_david_street(collector, sign_atlas, street_elevation_z)

    osm_buildings = osm_data.get("buildings", [])
    built_osm_count = 0
    for idx, b in enumerate(osm_buildings):
        if should_skip_osm_building(b):
            continue
        ring = b.get("ring", [])
        cx, cy = float(b["centroid"][0]), float(b["centroid"][1])
        bz = street_elevation_z(cx, cy)
        lvls = int(b.get("levels", 2))
        build_building_from_footprint(
            collector,
            ring,
            base_z=bz,
            levels=lvls,
            sign_atlas=sign_atlas,
            building_id=int(b.get("osm_id", idx)),
            is_hero=False,
        )
        built_osm_count += 1

    print(f"      Built {built_osm_count} surrounding OSM buildings + 21 hero landmark structures.")

    print("[5/5] Populating structured non-blob trees, trash scatters, utility boxes, lamps & bins...")
    populate_street_props_and_trees(collector, osm_data.get("trees", []))

    created_objs = collector.build_blender_objects(materials, ground_fn=street_elevation_z)
    total_verts = sum(len(obj.data.vertices) for obj in created_objs)
    total_faces = sum(len(obj.data.polygons) for obj in created_objs)
    print(f"[MESH] Created {len(created_objs)} batched mesh objects ({total_verts:,} verts, {total_faces:,} faces).")

    print(f"[GLTF] Exporting {out_glb_path} (with COLOR_0 baked vertex AO/dirt)...")
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb_path),
        export_format="GLB",
        export_apply=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_vertex_color="ACTIVE",
        export_materials="EXPORT",
    )
    patch_glb_alpha_modes(out_glb_path)
    glb_size = out_glb_path.stat().st_size if out_glb_path.exists() else 0
    print(f"[OK] Exported {out_glb_path} ({glb_size / (1024 * 1024):.2f} MB) in {time.time() - t0:.1f}s")

    manifest = {
        "origin": {"lat": 31.77661, "lon": 35.22768, "alt": 766.0},
        "glb_path": "/models/jaffa_gate_city.glb",
        "glb_size_bytes": glb_size,
        "mesh_objects": [obj.name for obj in created_objs],
        "stats": {
            "total_vertices": total_verts,
            "total_faces": total_faces,
            "osm_buildings_built": built_osm_count,
            "prop_counts": collector.prop_counts,
        },
        "landmarks": [
            {"id": "jaffa_gate", "name": "Jaffa Gate (Bab el-Khalil)", "enu": [0.0, 2.5, 0.0]},
            {"id": "wall_breach", "name": "1898 Wall Breach", "enu": [5.0, -9.5, 0.1]},
            {"id": "tower_of_david", "name": "Tower of David (Phasael Tower)", "enu": [24.5, -34.5, 0.35]},
            {"id": "citadel_minaret", "name": "Citadel Mamluk Minaret", "enu": [18.0, -66.0, 0.35]},
            {"id": "new_imperial_hotel", "name": "New Imperial Hotel", "enu": [36.0, 20.5, 0.55]},
            {"id": "petra_hotel", "name": "Petra Hotel", "enu": [74.0, 19.5, 0.65]},
            {"id": "david_street_souq", "name": "David Street Stepped Souq (Suq el-Bazar)", "enu": [95.0, 2.0, 0.1]},
        ],
    }
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Wrote {out_manifest_path}")


if __name__ == "__main__":
    main()
