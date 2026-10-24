#!/usr/bin/env python3
"""
Procedural High-Detail Seamless PBR Texture, Alpha Cutout & Jerusalem Signage Generator
Generates 100% from-scratch PBR sets (albedo, normal, roughness), botanical alpha cutouts,
architectural opening & souq shopfront atlas, and multilingual Jerusalem street/store signage
into public/textures/.
"""

import json
import math
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "public" / "textures"


# =============================================================================
# Core Procedural Noise & Tangent-Space Normal Map Helpers
# =============================================================================

def tileable_fbm(shape, octaves=5, base_freq=4, persistence=0.5, seed=42):
    """Generate seamless periodic fractal Brownian motion noise in [0, 1]."""
    rng = np.random.default_rng(seed)
    h, w = shape
    result = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total_amp = 0.0

    # Frequency grid in Fourier space for periodic filtering or cosine synthesis
    yy = np.arange(h, dtype=np.float32)[:, None] / h
    xx = np.arange(w, dtype=np.float32)[None, :] / w

    for o in range(octaves):
        freq = base_freq * (2 ** o)
        # Generate small random grid and upsample with periodic wrapping + bicubic/gaussian
        gh = min(h, max(4, freq * 2))
        gw = min(w, max(4, freq * 2))
        grid = rng.standard_normal((gh, gw)).astype(np.float32)
        # Tile via Kronecker/repeat or FFT interpolation
        # Using 2D FFT band-limited noise is super fast and guaranteed 100% seamless!
        noise_full = rng.standard_normal((h, w)).astype(np.float32)
        sigma = max(0.6, h / (freq * 3.5))
        layer = gaussian_filter(noise_full, sigma=sigma, mode="wrap")
        std = np.std(layer)
        if std > 1e-6:
            layer /= std
        result += amp * layer
        total_amp += amp
        amp *= persistence

    result /= max(total_amp, 1e-6)
    rmin, rmax = result.min(), result.max()
    if rmax > rmin:
        result = (result - rmin) / (rmax - rmin)
    return result


def height_to_normal(height, strength=6.0):
    """
    Compute tangent-space normal map (R, G, B in uint8 [0..255]) from heightfield H(u, v)
    using periodic central/Sobel differences so mortar joints, chiselled comb-dressing marks,
    and stone bosses catch sunlight dramatically.
    """
    h = height.astype(np.float32)
    # Periodic Sobel-like derivatives along u (x, columns) and v (y, rows)
    # Left-to-right u: dH/du = (H(x+1) - H(x-1)) / 2
    dh_du = (
        1.0 * (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1))
        + 0.5 * (np.roll(np.roll(h, -1, axis=1), -1, axis=0) - np.roll(np.roll(h, 1, axis=1), -1, axis=0))
        + 0.5 * (np.roll(np.roll(h, -1, axis=1), 1, axis=0) - np.roll(np.roll(h, 1, axis=1), 1, axis=0))
    ) * 0.25

    # Top-to-bottom image row y (OpenGL tangent +Y is up, so -dh_dy is +dv)
    dh_dy = (
        1.0 * (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0))
        + 0.5 * (np.roll(np.roll(h, -1, axis=0), -1, axis=1) - np.roll(np.roll(h, 1, axis=0), -1, axis=1))
        + 0.5 * (np.roll(np.roll(h, -1, axis=0), 1, axis=1) - np.roll(np.roll(h, 1, axis=0), 1, axis=1))
    ) * 0.25

    nx = -dh_du * strength
    ny = dh_dy * strength  # OpenGL convention: +Y points up (towards row 0)
    nz = np.ones_like(h)

    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= norm
    ny /= norm
    nz /= norm

    rgb = np.stack([
        np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255),
        np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255),
        np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255),
    ], axis=-1).astype(np.uint8)
    return rgb


def save_pbr_set(prefix, diff_rgb, height, rough, norm_strength=10.0, quality=93):
    """Save *_diff.jpg, *_norm.jpg, *_rough.jpg from arrays."""
    diff_u8 = np.clip(diff_rgb, 0, 255).astype(np.uint8)
    norm_u8 = height_to_normal(height, strength=norm_strength)
    rough_u8 = np.clip(rough * 255.0, 0, 255).astype(np.uint8)
    if rough_u8.ndim == 2:
        rough_u8 = np.stack([rough_u8] * 3, axis=-1)

    Image.fromarray(diff_u8, mode="RGB").save(OUT_DIR / f"{prefix}_diff.jpg", quality=quality, subsampling=0)
    Image.fromarray(norm_u8, mode="RGB").save(OUT_DIR / f"{prefix}_norm.jpg", quality=quality, subsampling=0)
    Image.fromarray(rough_u8, mode="RGB").save(OUT_DIR / f"{prefix}_rough.jpg", quality=quality, subsampling=0)
    print(f"  [OK] Saved PBR set: {prefix}_{{diff,norm,rough}}.jpg ({diff_u8.shape[1]}x{diff_u8.shape[0]})")


# =============================================================================
# 1. Seamless Limestone, Cobblestone, Plaster & Roof PBR Sets (1024x1024)
# =============================================================================

def gen_stone_ashlar(size=1024):
    """
    stone_ashlar_{diff,norm,rough}.jpg:
    Warm golden-cream Jerusalem Meleke dressed limestone courses (running bond,
    8 courses per tile, deeply recessed mortar joints, subtle block-to-block warm
    buff/ochre/cream variation, fine diagonal chisel comb-dressing striations,
    subtle dark rainwater streaking below courses).
    """
    rng = np.random.default_rng(101)
    courses = 8
    y = np.arange(size, dtype=np.float32)[:, None]
    x = np.arange(size, dtype=np.float32)[None, :]

    course_h = size / courses
    course_idx = np.floor(y / course_h).astype(np.int32) % courses
    local_y = (y % course_h) / course_h  # [0, 1] within course

    # Wavy hand-carved mortar line perturbation
    waviness_x = (tileable_fbm((size, size), octaves=3, base_freq=6, seed=102) - 0.5) * 3.5
    waviness_y = (tileable_fbm((size, size), octaves=3, base_freq=6, seed=103) - 0.5) * 2.5

    # Per-course block count (4 or 5 blocks per row) and running bond offset
    blocks_per_course = [4, 4, 5, 4, 4, 5, 4, 4]
    offsets = [0.0, 0.48, 0.23, 0.71, 0.15, 0.62, 0.37, 0.84]

    block_id = np.zeros((size, size), dtype=np.int32)
    dist_v_joint = np.ones((size, size), dtype=np.float32)
    dist_h_joint = np.minimum(local_y, 1.0 - local_y) * course_h + waviness_y

    for c in range(courses):
        mask = (course_idx == c)
        nb = blocks_per_course[c]
        bw = size / nb
        shifted_x = (x + offsets[c] * size + waviness_x) % size
        b_idx = np.floor(shifted_x / bw).astype(np.int32) % nb
        local_x = (shifted_x % bw) / bw
        dv = np.minimum(local_x, 1.0 - local_x) * bw
        block_id = np.where(mask, c * 10 + b_idx, block_id)
        dist_v_joint = np.where(mask, dv, dist_v_joint)

    # Joint profile: 5.5px mortar width with bevelled stone chamfers
    mortar_half = 3.2
    chamfer_w = 6.5
    joint_dist = np.minimum(dist_h_joint, dist_v_joint)
    is_mortar = joint_dist < mortar_half
    block_profile = np.clip((joint_dist - mortar_half) / chamfer_w, 0.0, 1.0)
    # Smooth ease-out chamfer
    block_profile = 1.0 - (1.0 - block_profile) ** 2

    # Per-block variation (height, warm tint, comb angle)
    max_id = courses * 10 + 10
    b_val = rng.uniform(-1.0, 1.0, size=max_id).astype(np.float32)
    b_warm = rng.uniform(-1.0, 1.0, size=max_id).astype(np.float32)
    b_comb_dir = rng.choice([1.0, -1.0], size=max_id).astype(np.float32)
    b_height_off = rng.uniform(-0.08, 0.08, size=max_id).astype(np.float32)

    blk_val = b_val[block_id]
    blk_warm = b_warm[block_id]
    blk_dir = b_comb_dir[block_id]
    blk_hoff = b_height_off[block_id]

    # Fine diagonal chisel comb-dressing striations (classic Jerusalem Shaham/Matabbe dressing)
    # Only inside the stone face (fading near drafted edge)
    inner_mask = np.clip((joint_dist - 12.0) / 6.0, 0.0, 1.0)
    diag_coord = (x + blk_dir * y) * (2.0 * math.pi / 7.5)
    comb_striations = np.sin(diag_coord + tileable_fbm((size, size), octaves=2, base_freq=8, seed=104) * 2.5)
    # Interrupt comb strokes into 25px tool blows
    blow_mod = np.sin((x - blk_dir * y) * (2.0 * math.pi / 32.0)) * 0.5 + 0.5
    comb_relief = comb_striations * (0.4 + 0.6 * blow_mod) * inner_mask * 0.09

    # Multi-scale limestone grain & fossiliferous pores
    f_macro = tileable_fbm((size, size), octaves=5, base_freq=4, seed=105)
    f_micro = tileable_fbm((size, size), octaves=4, base_freq=16, seed=106)
    pores = np.clip(f_micro - 0.72, 0.0, 1.0) * (-0.35)

    # Heightfield
    height = (
        block_profile * (0.72 + blk_hoff + 0.16 * f_macro + comb_relief + pores)
        + (1.0 - block_profile) * (0.12 * f_micro)
    )

    # Rainwater streaking below horizontal mortar joints
    streak_noise = tileable_fbm((size, size), octaves=4, base_freq=12, seed=107)
    # Stretch vertically for drip streaks
    streak_v = gaussian_filter(streak_noise, sigma=(18.0, 1.5), mode="wrap")
    streak_v = (streak_v - streak_v.min()) / (streak_v.max() - streak_v.min() + 1e-6)
    rain_drip = np.exp(-local_y * 4.5) * (0.35 + 0.65 * streak_v) * 0.18

    # Base Jerusalem Meleke golden-cream palette: RGB ~(226, 206, 172)
    base_r = 226.0 + blk_val * 13.0 + blk_warm * 8.0 + (f_macro - 0.5) * 26.0 + (f_micro - 0.5) * 14.0 + comb_relief * 45.0
    base_g = 205.0 + blk_val * 12.0 + blk_warm * 3.0 + (f_macro - 0.5) * 24.0 + (f_micro - 0.5) * 13.0 + comb_relief * 40.0
    base_b = 171.0 + blk_val * 10.0 - blk_warm * 6.0 + (f_macro - 0.5) * 20.0 + (f_micro - 0.5) * 11.0 + comb_relief * 32.0

    # Apply rainwater darkening and warm patina at bottom of courses
    patina = np.exp(-(1.0 - local_y) * 6.0) * 0.08
    stone_r = base_r * (1.0 - rain_drip) + patina * 15.0
    stone_g = base_g * (1.0 - rain_drip * 1.05)
    stone_b = base_b * (1.0 - rain_drip * 1.12) - patina * 12.0

    # Recessed lime mortar colour: dusty grey-buff ~(142, 130, 110)
    mortar_r = 136.0 + (f_micro - 0.5) * 25.0
    mortar_g = 125.0 + (f_micro - 0.5) * 22.0
    mortar_b = 106.0 + (f_micro - 0.5) * 18.0

    # Ambient occlusion inside mortar joints
    ao = 0.65 + 0.35 * block_profile
    diff_r = (stone_r * block_profile + mortar_r * (1.0 - block_profile)) * ao
    diff_g = (stone_g * block_profile + mortar_g * (1.0 - block_profile)) * ao
    diff_b = (stone_b * block_profile + mortar_b * (1.0 - block_profile)) * ao

    diff = np.stack([diff_r, diff_g, diff_b], axis=-1)
    rough = np.clip(0.72 - block_profile * 0.14 + (1.0 - f_macro) * 0.12 + rain_drip * 0.15 + (1.0 - block_profile) * 0.22, 0.42, 0.96)

    save_pbr_set("stone_ashlar", diff, height, rough, norm_strength=14.0)


def gen_stone_boss(size=1024):
    """
    stone_boss_{diff,norm,rough}.jpg:
    Heavy Herodian / Ottoman rusticated bossed masonry (for Phasael Tower, Citadel glacis,
    Jaffa Gate bastions, and ground-floor rusticated piers) with recessed flat drafted
    margins around protruding, craggy rock-faced stone bosses.
    """
    rng = np.random.default_rng(201)
    courses = 6
    y = np.arange(size, dtype=np.float32)[:, None]
    x = np.arange(size, dtype=np.float32)[None, :]

    course_h = size / courses
    course_idx = np.floor(y / course_h).astype(np.int32) % courses
    local_y = (y % course_h) / course_h

    wav_x = (tileable_fbm((size, size), octaves=3, base_freq=6, seed=202) - 0.5) * 3.0
    wav_y = (tileable_fbm((size, size), octaves=3, base_freq=6, seed=203) - 0.5) * 2.5

    blocks_per_course = [3, 3, 4, 3, 3, 4]
    offsets = [0.0, 0.5, 0.22, 0.72, 0.18, 0.65]

    block_id = np.zeros((size, size), dtype=np.int32)
    dist_v_joint = np.ones((size, size), dtype=np.float32)
    dist_h_joint = np.minimum(local_y, 1.0 - local_y) * course_h + wav_y

    for c in range(courses):
        mask = (course_idx == c)
        nb = blocks_per_course[c]
        bw = size / nb
        shifted_x = (x + offsets[c] * size + wav_x) % size
        b_idx = np.floor(shifted_x / bw).astype(np.int32) % nb
        local_x = (shifted_x % bw) / bw
        dv = np.minimum(local_x, 1.0 - local_x) * bw
        block_id = np.where(mask, c * 10 + b_idx, block_id)
        dist_v_joint = np.where(mask, dv, dist_v_joint)

    joint_dist = np.minimum(dist_h_joint, dist_v_joint)

    # 1) Deep mortar joint (< 4px)
    mortar_w = 4.0
    margin_bevel = 4.0
    margin_step = np.clip((joint_dist - mortar_w) / margin_bevel, 0.0, 1.0)

    # 2) Flat chiselled drafted margin (from 4px to 24px inside each block edge)
    # 3) Protruding craggy rock-faced boss (from 24px inward with steep 18px bevel)
    boss_start = 22.0 + (tileable_fbm((size, size), octaves=4, base_freq=8, seed=204) - 0.5) * 9.0
    boss_ramp_w = 18.0
    boss_mask = np.clip((joint_dist - boss_start) / boss_ramp_w, 0.0, 1.0)
    # Pillow-like + craggy rock-face profile
    boss_pillow = np.sin(boss_mask * (math.pi * 0.5))

    crag_macro = tileable_fbm((size, size), octaves=5, base_freq=6, persistence=0.6, seed=205)
    crag_sharp = 1.0 - np.abs(2.0 * tileable_fbm((size, size), octaves=4, base_freq=10, seed=206) - 1.0)
    f_micro = tileable_fbm((size, size), octaves=4, base_freq=20, seed=207)

    # Per-block colour/height variation
    max_id = 80
    b_val = rng.uniform(-1.0, 1.0, size=max_id).astype(np.float32)[block_id]
    b_warm = rng.uniform(-1.0, 1.0, size=max_id).astype(np.float32)[block_id]
    b_hoff = rng.uniform(-0.08, 0.10, size=max_id).astype(np.float32)[block_id]

    # Margin chisel marks
    margin_only = margin_step * (1.0 - np.clip((joint_dist - boss_start) / 6.0, 0.0, 1.0))
    margin_chisel = np.sin((x + y) * (2.0 * math.pi / 6.0)) * 0.025 * margin_only

    # Total heightfield: mortar (0.0) -> drafted margin (0.28) -> craggy boss (0.65..1.0)
    height = (
        margin_step * 0.28
        + margin_chisel
        + boss_pillow * (0.48 + b_hoff + 0.26 * crag_macro + 0.14 * crag_sharp)
        + 0.05 * f_micro
    )

    # Albedo: warm weathered golden-ochre limestone with bright dressed margins and shadowed boss undercuts
    ao = 0.55 + 0.25 * margin_step + 0.20 * boss_pillow
    # Self-shadowing under boss bottom edge
    bottom_shadow = np.exp(-np.abs(dist_h_joint - boss_start) / 6.0) * (local_y > 0.5).astype(np.float32) * 0.18

    base_r = 216.0 + b_val * 14.0 + b_warm * 9.0 + (crag_macro - 0.5) * 32.0 + (crag_sharp - 0.5) * 18.0 + margin_only * 10.0
    base_g = 194.0 + b_val * 12.0 + b_warm * 4.0 + (crag_macro - 0.5) * 28.0 + (crag_sharp - 0.5) * 16.0 + margin_only * 9.0
    base_b = 158.0 + b_val * 10.0 - b_warm * 5.0 + (crag_macro - 0.5) * 24.0 + (crag_sharp - 0.5) * 14.0 + margin_only * 8.0

    mortar_r = 112.0 + (f_micro - 0.5) * 20.0
    mortar_g = 102.0 + (f_micro - 0.5) * 18.0
    mortar_b = 86.0 + (f_micro - 0.5) * 15.0

    diff_r = (base_r * margin_step + mortar_r * (1.0 - margin_step)) * ao * (1.0 - bottom_shadow)
    diff_g = (base_g * margin_step + mortar_g * (1.0 - margin_step)) * ao * (1.0 - bottom_shadow)
    diff_b = (base_b * margin_step + mortar_b * (1.0 - margin_step)) * ao * (1.0 - bottom_shadow)

    diff = np.stack([diff_r, diff_g, diff_b], axis=-1)
    rough = np.clip(0.84 - margin_only * 0.16 + (1.0 - crag_macro) * 0.12, 0.55, 0.97)

    save_pbr_set("stone_boss", diff, height, rough, norm_strength=18.0)


def gen_stone_rubble(size=1024):
    """
    stone_rubble_{diff,norm,rough}.jpg:
    Historic coursed rubble masonry with varied warm beige/grey-ochre fieldstones
    and deep lime mortar joints.
    """
    rng = np.random.default_rng(301)
    courses = 10
    y = np.arange(size, dtype=np.float32)[:, None]
    x = np.arange(size, dtype=np.float32)[None, :]

    # Perturb coordinates organically so stones look like hand-laid coursed fieldstone rubble
    warp_x = (tileable_fbm((size, size), octaves=4, base_freq=5, seed=302) - 0.5) * 28.0
    warp_y = (tileable_fbm((size, size), octaves=4, base_freq=5, seed=303) - 0.5) * 22.0

    wy = (y + warp_y) % size
    wx = (x + warp_x) % size

    course_h = size / courses
    c_idx = np.floor(wy / course_h).astype(np.int32) % courses
    ly = (wy % course_h) / course_h
    dh = np.minimum(ly, 1.0 - ly) * course_h

    stone_id = np.zeros((size, size), dtype=np.int32)
    dv = np.ones((size, size), dtype=np.float32)

    for c in range(courses):
        mask = (c_idx == c)
        nb = int(rng.integers(5, 8))
        bw = size / nb
        off = rng.uniform(0.0, size)
        sx = (wx + off) % size
        b_idx = np.floor(sx / bw).astype(np.int32) % nb
        lx = (sx % bw) / bw
        dv_c = np.minimum(lx, 1.0 - lx) * bw
        stone_id = np.where(mask, c * 12 + b_idx, stone_id)
        dv = np.where(mask, dv_c, dv)

    # Round the corners of each fieldstone using Lp norm distance from joint
    edge_dist = np.minimum(dh, dv) - (14.0 * np.exp(-(dh * dv) / 180.0))
    mortar_t = 4.5
    stone_mask = np.clip((edge_dist - mortar_t) / 10.0, 0.0, 1.0)
    stone_dome = np.sin(stone_mask * (math.pi * 0.5))

    max_id = 150
    s_val = rng.uniform(-1.0, 1.0, size=max_id).astype(np.float32)[stone_id]
    s_warm = rng.uniform(-1.0, 1.0, size=max_id).astype(np.float32)[stone_id]
    s_hoff = rng.uniform(-0.12, 0.12, size=max_id).astype(np.float32)[stone_id]

    f_macro = tileable_fbm((size, size), octaves=5, base_freq=7, seed=304)
    f_micro = tileable_fbm((size, size), octaves=4, base_freq=18, seed=305)

    height = (
        stone_dome * (0.62 + s_hoff + 0.25 * f_macro + 0.08 * f_micro)
        + (1.0 - stone_dome) * (0.18 * f_micro)
    )

    stone_r = 208.0 + s_val * 20.0 + s_warm * 12.0 + (f_macro - 0.5) * 30.0
    stone_g = 190.0 + s_val * 18.0 + s_warm * 5.0 + (f_macro - 0.5) * 26.0
    stone_b = 159.0 + s_val * 15.0 - s_warm * 8.0 + (f_macro - 0.5) * 22.0

    # Lime mortar with sand aggregate specks
    mortar_r = 166.0 + (f_micro - 0.5) * 32.0
    mortar_g = 154.0 + (f_micro - 0.5) * 28.0
    mortar_b = 132.0 + (f_micro - 0.5) * 24.0

    ao = 0.62 + 0.38 * stone_dome
    diff_r = (stone_r * stone_mask + mortar_r * (1.0 - stone_mask)) * ao
    diff_g = (stone_g * stone_mask + mortar_g * (1.0 - stone_mask)) * ao
    diff_b = (stone_b * stone_mask + mortar_b * (1.0 - stone_mask)) * ao

    diff = np.stack([diff_r, diff_g, diff_b], axis=-1)
    rough = np.clip(0.88 - stone_dome * 0.18 + (1.0 - f_macro) * 0.12, 0.58, 0.97)

    save_pbr_set("stone_rubble", diff, height, rough, norm_strength=15.0)


def gen_paving_setts(size=1024):
    """
    paving_setts_{diff,norm,rough}.jpg:
    Polished Jerusalem flagstone and staggered stone street cobble setts with
    rounded bevelled edges, worn shiny centres (lower roughness 0.45), and dark
    dusty mortar joints (high roughness 0.92).
    """
    rng = np.random.default_rng(401)
    rows = 10
    y = np.arange(size, dtype=np.float32)[:, None]
    x = np.arange(size, dtype=np.float32)[None, :]

    wav_x = (tileable_fbm((size, size), octaves=3, base_freq=6, seed=402) - 0.5) * 5.0
    wav_y = (tileable_fbm((size, size), octaves=3, base_freq=6, seed=403) - 0.5) * 4.0

    wy = (y + wav_y) % size
    wx = (x + wav_x) % size

    row_h = size / rows
    r_idx = np.floor(wy / row_h).astype(np.int32) % rows
    ly = (wy % row_h) / row_h
    dh = np.minimum(ly, 1.0 - ly) * row_h

    sett_id = np.zeros((size, size), dtype=np.int32)
    dv = np.ones((size, size), dtype=np.float32)
    center_r2 = np.zeros((size, size), dtype=np.float32)

    for r in range(rows):
        mask = (r_idx == r)
        cols = 6 if (r % 2 == 0) else 7
        cw = size / cols
        off = (r * 0.37) * size
        sx = (wx + off) % size
        c_idx = np.floor(sx / cw).astype(np.int32) % cols
        lx = (sx % cw) / cw
        dv_r = np.minimum(lx, 1.0 - lx) * cw
        # Normalized squared distance from sett centre (for foot-worn polish)
        cr2 = ((lx - 0.5) * 2.0) ** 2 + ((ly - 0.5) * 2.0) ** 2
        sett_id = np.where(mask, r * 10 + c_idx, sett_id)
        dv = np.where(mask, dv_r, dv)
        center_r2 = np.where(mask, cr2, center_r2)

    # Rounded bevelled stone corners
    edge_dist = np.minimum(dh, dv) - 8.5 * np.exp(-(dh * dv) / 90.0)
    joint_w = 3.8
    bevel_w = 11.0
    sett_mask = np.clip((edge_dist - joint_w) / bevel_w, 0.0, 1.0)
    sett_dome = np.sin(sett_mask * (math.pi * 0.5))

    # Foot-worn polished centre vs slightly rougher bevelled edge
    polish_center = np.clip(1.0 - center_r2 * 0.85, 0.0, 1.0) * sett_mask

    max_id = 120
    # Classic Jerusalem street paving mixes warm cream, pale rose-pink (Mizzi Ahmar), and cool grey-buff flagstones
    s_tone = rng.uniform(-1.0, 1.0, size=max_id).astype(np.float32)[sett_id]
    s_rose = rng.uniform(-0.8, 1.0, size=max_id).astype(np.float32)[sett_id]
    s_hoff = rng.uniform(-0.06, 0.06, size=max_id).astype(np.float32)[sett_id]

    f_macro = tileable_fbm((size, size), octaves=4, base_freq=6, seed=404)
    f_micro = tileable_fbm((size, size), octaves=4, base_freq=20, seed=405)

    height = (
        sett_dome * (0.72 + s_hoff + 0.14 * f_macro + 0.04 * (1.0 - polish_center) * f_micro)
        + (1.0 - sett_dome) * (0.10 * f_micro)
    )

    sett_r = 205.0 + s_tone * 18.0 + s_rose * 11.0 + (f_macro - 0.5) * 22.0 + polish_center * 10.0
    sett_g = 188.0 + s_tone * 16.0 - s_rose * 2.0 + (f_macro - 0.5) * 20.0 + polish_center * 9.0
    sett_b = 162.0 + s_tone * 14.0 - s_rose * 7.0 + (f_macro - 0.5) * 18.0 + polish_center * 8.0

    # Dark dusty mortar joints
    joint_r = 92.0 + (f_micro - 0.5) * 20.0
    joint_g = 84.0 + (f_micro - 0.5) * 18.0
    joint_b = 72.0 + (f_micro - 0.5) * 16.0

    ao = 0.58 + 0.42 * sett_dome
    diff_r = (sett_r * sett_mask + joint_r * (1.0 - sett_mask)) * ao
    diff_g = (sett_g * sett_mask + joint_g * (1.0 - sett_mask)) * ao
    diff_b = (sett_b * sett_mask + joint_b * (1.0 - sett_mask)) * ao

    diff = np.stack([diff_r, diff_g, diff_b], axis=-1)

    # Roughness: worn shiny centres (~0.40..0.48), bevels (~0.65), dark dusty mortar joints (~0.92..0.96)
    stone_rough = 0.64 - polish_center * 0.22 + (1.0 - f_macro) * 0.06
    rough = sett_mask * stone_rough + (1.0 - sett_mask) * (0.92 + 0.04 * f_micro)
    rough = np.clip(rough, 0.36, 0.97)

    save_pbr_set("paving_setts", diff, height, rough, norm_strength=14.0)


def gen_plaster_warm(size=1024):
    """
    plaster_warm_{diff,norm,rough}.jpg:
    Weathered lime-washed cream plaster for recessed upper walls and vaults.
    """
    f_broad = tileable_fbm((size, size), octaves=4, base_freq=3, seed=501)
    f_med = tileable_fbm((size, size), octaves=5, base_freq=8, seed=502)
    f_fine = tileable_fbm((size, size), octaves=4, base_freq=24, seed=503)

    # Subtle trowel sweep marks
    y = np.arange(size, dtype=np.float32)[:, None]
    x = np.arange(size, dtype=np.float32)[None, :]
    trowel = np.sin((x * 0.7 + y * 0.4 + f_broad * 60.0) * (2.0 * math.pi / 48.0)) * 0.04

    # Slight flaking patches where undercoat shows through
    flake = np.clip((f_med - 0.74) * 5.0, 0.0, 1.0)

    height = 0.55 * f_broad + 0.28 * f_med + 0.17 * f_fine + trowel - flake * 0.18

    r = 232.0 + (f_broad - 0.5) * 18.0 + (f_med - 0.5) * 14.0 - flake * 16.0
    g = 216.0 + (f_broad - 0.5) * 16.0 + (f_med - 0.5) * 13.0 - flake * 18.0
    b = 186.0 + (f_broad - 0.5) * 14.0 + (f_med - 0.5) * 12.0 - flake * 20.0

    diff = np.stack([r, g, b], axis=-1)
    rough = np.clip(0.76 + (1.0 - f_med) * 0.12 + flake * 0.08, 0.65, 0.95)

    save_pbr_set("plaster_warm", diff, height, rough, norm_strength=7.5)


def gen_roof_tile(size=1024):
    """
    roof_tile_{diff,norm,rough}.jpg:
    Mediterranean terracotta barrel roof tiles (pan-and-cover mission tiles) with
    warm baked clay variation and course shadow overlap.
    """
    rng = np.random.default_rng(601)
    cols = 12
    rows = 8
    y = np.arange(size, dtype=np.float32)[:, None]
    x = np.arange(size, dtype=np.float32)[None, :]

    col_w = size / cols
    row_h = size / rows

    c_idx = np.floor(x / col_w).astype(np.int32) % cols
    r_idx = np.floor(y / row_h).astype(np.int32) % rows

    lx = (x % col_w) / col_w  # [0, 1] across tile column
    ly = (y % row_h) / row_h  # [0, 1] along tile course

    # Alternating pan (concave valley) and cover (convex ridge) barrel profile
    barrel = np.sin(lx * math.pi)
    # Course overlap lip at bottom of each row (ly near 1.0)
    lip_shadow = np.exp(-(1.0 - ly) * 9.0)
    course_ramp = 0.82 + 0.18 * ly

    tile_id = r_idx * cols + c_idx
    t_hue = rng.uniform(-1.0, 1.0, size=cols * rows).astype(np.float32)[tile_id]
    t_val = rng.uniform(-1.0, 1.0, size=cols * rows).astype(np.float32)[tile_id]

    f_clay = tileable_fbm((size, size), octaves=4, base_freq=10, seed=602)
    lichen = np.clip(tileable_fbm((size, size), octaves=5, base_freq=8, seed=603) - 0.68, 0.0, 1.0) * 2.5

    height = barrel * course_ramp * 0.78 + 0.12 * f_clay - lip_shadow * 0.25

    ao = (0.55 + 0.45 * barrel) * (1.0 - 0.35 * lip_shadow)
    r = (186.0 + t_val * 16.0 + t_hue * 12.0 + (f_clay - 0.5) * 24.0 - lichen * 35.0) * ao
    g = (106.0 + t_val * 12.0 - t_hue * 4.0 + (f_clay - 0.5) * 18.0 + lichen * 12.0) * ao
    b = (72.0 + t_val * 9.0 - t_hue * 8.0 + (f_clay - 0.5) * 14.0 - lichen * 10.0) * ao

    diff = np.stack([r, g, b], axis=-1)
    rough = np.clip(0.68 + (1.0 - barrel) * 0.16 + lichen * 0.14, 0.50, 0.94)

    save_pbr_set("roof_tile", diff, height, rough, norm_strength=14.0)


# =============================================================================
# 2. Architectural Opening & Souq Shopfront Atlas (1024x1024: wood_doors_*)
# =============================================================================

def gen_wood_doors_atlas(size=1024):
    """
    2x2 quadrant atlas (u, v in [0..0.5] and [0.5..1.0]):
    NOTE on UV mapping:
      In OpenGL / Three.js (`flipY = true`, default):
      v in [0..0.5] is the BOTTOM half of the image (rows 512..1024)
      v in [0.5..1.0] is the TOP half of the image (rows 0..512)
      u in [0..0.5] is the LEFT half of the image (cols 0..512)
      u in [0.5..1.0] is the RIGHT half of the image (cols 512..1024)
    - Top-left [u: 0..0.5, v: 0.5..1.0] (rows 0..512, cols 0..512): window_glass
    - Top-right [u: 0.5..1.0, v: 0.5..1.0] (rows 0..512, cols 512..1024): souq_open
    - Bottom-left [u: 0..0.5, v: 0..0.5] (rows 512..1024, cols 0..512): door_wood
    - Bottom-right [u: 0.5..1.0, v: 0..0.5] (rows 512..1024, cols 512..1024): shutter_louvre
    """
    q = size // 2  # 512
    diff = np.zeros((size, size, 3), dtype=np.float32)
    height = np.zeros((size, size), dtype=np.float32)
    rough = np.ones((size, size), dtype=np.float32) * 0.7

    f_wood = tileable_fbm((q, q), octaves=4, base_freq=8, seed=701)
    # Stretch vertically for wood grain
    wood_grain = gaussian_filter(f_wood, sigma=(10.0, 1.2), mode="wrap")
    wood_grain = (wood_grain - wood_grain.min()) / (wood_grain.max() - wood_grain.min() + 1e-6)

    yy = np.linspace(0.0, 1.0, q, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, q, dtype=np.float32)[None, :]

    # -------------------------------------------------------------------------
    # Q1: Bottom-Left (rows q:size, cols 0:q) -> door_wood
    # Heavy panelled dark-green & walnut Jerusalem timber door with raised bevelled panels & brass studs
    # -------------------------------------------------------------------------
    d_rgb = np.zeros((q, q, 3), dtype=np.float32)
    d_h = np.ones((q, q), dtype=np.float32) * 0.45 + wood_grain * 0.06
    d_r = np.ones((q, q), dtype=np.float32) * 0.58

    # Rich aged walnut & Jerusalem dark-green painted frame
    # Outer stiles/rails in deep walnut-brown, inner panels in heritage dark olive-pine green
    d_rgb[:, :, 0] = 68.0 + wood_grain * 26.0
    d_rgb[:, :, 1] = 46.0 + wood_grain * 18.0
    d_rgb[:, :, 2] = 30.0 + wood_grain * 12.0

    # Center split between double door leaves
    center_seam = np.exp(-((xx - 0.5) / 0.008) ** 2)
    d_h -= center_seam * 0.35
    d_rgb *= (1.0 - center_seam[:, :, None] * 0.65)

    # 6 coffers / bevelled panels (2 columns x 3 rows)
    panel_boxes = [
        (0.09, 0.07, 0.44, 0.28), (0.56, 0.07, 0.91, 0.28),
        (0.09, 0.35, 0.44, 0.64), (0.56, 0.35, 0.91, 0.64),
        (0.09, 0.71, 0.44, 0.92), (0.56, 0.71, 0.91, 0.92),
    ]
    for (x0, y0, x1, y1) in panel_boxes:
        dx = np.minimum(xx - x0, x1 - xx)
        dy = np.minimum(yy - y0, y1 - yy)
        d_edge = np.minimum(dx, dy)
        in_box = (d_edge > 0).astype(np.float32)
        # Molding groove then raised central panel
        groove = np.exp(-((d_edge - 0.018) / 0.009) ** 2) * in_box
        raised = np.clip((d_edge - 0.032) / 0.022, 0.0, 1.0) * in_box
        d_h = d_h - groove * 0.28 + raised * 0.26
        # Heritage painted dark green-teal inner panel with warm wood grain showing through
        panel_r = 36.0 + wood_grain * 22.0
        panel_g = 66.0 + wood_grain * 26.0
        panel_b = 50.0 + wood_grain * 18.0
        blend = np.clip(d_edge / 0.015, 0.0, 1.0)
        d_rgb[:, :, 0] = d_rgb[:, :, 0] * (1.0 - blend) + panel_r * blend
        d_rgb[:, :, 1] = d_rgb[:, :, 1] * (1.0 - blend) + panel_g * blend
        d_rgb[:, :, 2] = d_rgb[:, :, 2] * (1.0 - blend) + panel_b * blend
        d_rgb *= (1.0 - groove[:, :, None] * 0.45)

    # Brass studs along cross rails and two ornate brass knockers
    stud_y_list = [0.035, 0.315, 0.675, 0.955]
    stud_x_list = np.linspace(0.08, 0.92, 10)
    for sy in stud_y_list:
        for sx in stud_x_list:
            if abs(sx - 0.5) < 0.03:
                continue
            r2 = (xx - sx) ** 2 + (yy - sy) ** 2
            stud = np.clip(1.0 - r2 / (0.011 ** 2), 0.0, 1.0)
            dome = np.sqrt(stud)
            d_h += dome * 0.32
            d_rgb[:, :, 0] = d_rgb[:, :, 0] * (1.0 - stud) + (215.0 * dome) * stud
            d_rgb[:, :, 1] = d_rgb[:, :, 1] * (1.0 - stud) + (175.0 * dome) * stud
            d_rgb[:, :, 2] = d_rgb[:, :, 2] * (1.0 - stud) + (75.0 * dome) * stud
            d_r = d_r * (1.0 - stud) + 0.25 * stud

    # Brass ring knockers on middle panels
    for kx in [0.265, 0.735]:
        ky = 0.49
        ring_r = np.sqrt((xx - kx) ** 2 + (yy - ky) ** 2)
        ring = np.exp(-((ring_r - 0.042) / 0.008) ** 2)
        plate = np.clip(1.0 - ((xx - kx) ** 2 + (yy - (ky - 0.03)) ** 2) / (0.022 ** 2), 0.0, 1.0)
        brass = np.clip(ring + plate, 0.0, 1.0)
        d_h += brass * 0.35
        d_rgb[:, :, 0] = d_rgb[:, :, 0] * (1.0 - brass) + 225.0 * brass
        d_rgb[:, :, 1] = d_rgb[:, :, 1] * (1.0 - brass) + 184.0 * brass
        d_rgb[:, :, 2] = d_rgb[:, :, 2] * (1.0 - brass) + 82.0 * brass
        d_r = d_r * (1.0 - brass) + 0.22 * brass

    diff[q:size, 0:q] = d_rgb
    height[q:size, 0:q] = d_h
    rough[q:size, 0:q] = d_r

    # -------------------------------------------------------------------------
    # Q2: Bottom-Right (rows q:size, cols q:size) -> shutter_louvre
    # Painted timber & dark-green louvred window shutters with crisp horizontal slats
    # -------------------------------------------------------------------------
    s_rgb = np.zeros((q, q, 3), dtype=np.float32)
    s_h = np.ones((q, q), dtype=np.float32) * 0.5
    s_r = np.ones((q, q), dtype=np.float32) * 0.55

    # Base Levant green shutter paint (#2a523be0) with weathered wood warm undertone
    s_rgb[:, :, 0] = 44.0 + wood_grain * 20.0
    s_rgb[:, :, 1] = 84.0 + wood_grain * 26.0
    s_rgb[:, :, 2] = 62.0 + wood_grain * 18.0

    # Louvred zones inside left and right shutter leaves
    slat_boxes = [
        (0.08, 0.07, 0.44, 0.46), (0.56, 0.07, 0.92, 0.46),
        (0.08, 0.53, 0.44, 0.92), (0.56, 0.53, 0.92, 0.92),
    ]
    n_slats = 28.0
    slat_phase = (yy * n_slats) % 1.0  # sawtooth ramp per slat
    for (x0, y0, x1, y1) in slat_boxes:
        in_slat = ((xx > x0) & (xx < x1) & (yy > y0) & (yy < y1)).astype(np.float32)
        # Crisp angled louvre slat height and underside shadow
        slat_h = (0.25 + 0.65 * slat_phase)
        slat_shade = 0.52 + 0.58 * slat_phase
        s_h = s_h * (1.0 - in_slat) + slat_h * in_slat
        s_rgb *= (1.0 - in_slat[:, :, None]) + (slat_shade * in_slat)[:, :, None]

    # Center meeting stile seam & wrought iron latch
    seam = np.exp(-((xx - 0.5) / 0.008) ** 2)
    s_h -= seam * 0.35
    s_rgb *= (1.0 - seam[:, :, None] * 0.6)

    diff[q:size, q:size] = s_rgb
    height[q:size, q:size] = s_h
    rough[q:size, q:size] = s_r

    # -------------------------------------------------------------------------
    # Q3: Top-Left (rows 0:q, cols 0:q) -> window_glass
    # Divided-light dark glazed window with warm interior room depth, subtle curtain/stone
    # reveal shadow, and specular glass smoothness (roughness = 0.15)
    # -------------------------------------------------------------------------
    w_rgb = np.zeros((q, q, 3), dtype=np.float32)
    w_h = np.ones((q, q), dtype=np.float32) * 0.18
    w_r = np.ones((q, q), dtype=np.float32) * 0.15  # Smooth specular glass

    # Warm interior room glow + subtle sky reflection gradient + soft linen curtain folds on sides
    curtain_fold = (0.5 + 0.5 * np.sin(xx * 28.0 * math.pi)) * np.exp(-((np.minimum(xx, 1.0 - xx)) / 0.22) ** 2)
    warm_glow = np.exp(-((xx - 0.5) ** 2 + (yy - 0.55) ** 2) / 0.28)
    sky_refl = (1.0 - yy) * 0.35

    w_rgb[:, :, 0] = 34.0 + warm_glow * 62.0 + curtain_fold * 55.0 + sky_refl * 35.0
    w_rgb[:, :, 1] = 38.0 + warm_glow * 46.0 + curtain_fold * 48.0 + sky_refl * 48.0
    w_rgb[:, :, 2] = 44.0 + warm_glow * 24.0 + curtain_fold * 38.0 + sky_refl * 64.0

    # Outer timber window frame + 3x4 divided-light muntin bars
    border = ((xx < 0.06) | (xx > 0.94) | (yy < 0.06) | (yy > 0.94)).astype(np.float32)
    v_bars = (np.abs(xx - 0.35) < 0.016) | (np.abs(xx - 0.65) < 0.016)
    h_bars = (np.abs(yy - 0.28) < 0.015) | (np.abs(yy - 0.52) < 0.015) | (np.abs(yy - 0.76) < 0.015)
    frame_mask = np.clip(border + v_bars.astype(np.float32) + h_bars.astype(np.float32), 0.0, 1.0)

    w_h = w_h * (1.0 - frame_mask) + (0.78 + 0.05 * wood_grain) * frame_mask
    w_r = w_r * (1.0 - frame_mask) + 0.62 * frame_mask
    w_rgb[:, :, 0] = w_rgb[:, :, 0] * (1.0 - frame_mask) + (78.0 + wood_grain * 20.0) * frame_mask
    w_rgb[:, :, 1] = w_rgb[:, :, 1] * (1.0 - frame_mask) + (56.0 + wood_grain * 15.0) * frame_mask
    w_rgb[:, :, 2] = w_rgb[:, :, 2] * (1.0 - frame_mask) + (38.0 + wood_grain * 10.0) * frame_mask

    diff[0:q, 0:q] = w_rgb
    height[0:q, 0:q] = w_h
    rough[0:q, 0:q] = w_r

    # -------------------------------------------------------------------------
    # Q4: Top-Right (rows 0:q, cols q:size) -> souq_open
    # Rich open bazaar shopfront interior showing hanging colorful patterned carpets/textiles,
    # brass lanterns, ceramic plates on shelves, and dark recessed shop depth!
    # Rendered with PIL + numpy for vivid detail.
    # -------------------------------------------------------------------------
    souq_img = Image.new("RGB", (q, q), (26, 20, 16))
    s_draw = ImageDraw.Draw(souq_img)
    souq_h = np.ones((q, q), dtype=np.float32) * 0.15
    souq_r = np.ones((q, q), dtype=np.float32) * 0.65

    # Warm vaulted ceiling glow at top-center of shop interior
    ceil_glow = np.exp(-((xx - 0.5) ** 2 / 0.16 + (yy - 0.28) ** 2 / 0.18))
    bg_arr = np.array(souq_img, dtype=np.float32)
    bg_arr[:, :, 0] += ceil_glow * 95.0
    bg_arr[:, :, 1] += ceil_glow * 62.0
    bg_arr[:, :, 2] += ceil_glow * 28.0
    souq_img = Image.fromarray(np.clip(bg_arr, 0, 255).astype(np.uint8))
    s_draw = ImageDraw.Draw(souq_img)

    # Left & Right hanging woven Anatolian/Persian kilim carpets on side jambs
    rugs = [
        # (x0, y0, x1, y1, base_color, border_color, diamond_color)
        (18, 48, 128, 370, (168, 34, 32), (218, 168, 64), (28, 64, 118)),
        (38, 180, 142, 465, (192, 92, 28), (235, 215, 175), (124, 24, 32)),
        (384, 48, 494, 370, (24, 78, 112), (212, 158, 52), (176, 38, 38)),
        (368, 185, 476, 468, (148, 28, 42), (224, 188, 92), (38, 96, 68)),
    ]
    for (rx0, ry0, rx1, ry1, c_base, c_bord, c_dia) in rugs:
        s_draw.rectangle([rx0, ry0, rx1, ry1], fill=c_base, outline=c_bord, width=5)
        s_draw.rectangle([rx0 + 9, ry0 + 9, rx1 - 9, ry1 - 9], outline=(240, 225, 190), width=2)
        # Geometric kilim medallions
        cx = (rx0 + rx1) // 2
        for cy in range(ry0 + 38, ry1 - 30, 56):
            pts = [(cx, cy - 20), (cx + 22, cy), (cx, cy + 20), (cx - 22, cy)]
            s_draw.polygon(pts, fill=c_dia, outline=c_bord)
            pts_in = [(cx, cy - 9), (cx + 10, cy), (cx, cy + 9), (cx - 10, cy)]
            s_draw.polygon(pts_in, fill=c_bord)
        # Tassels at bottom
        for tx in range(rx0 + 4, rx1 - 2, 6):
            s_draw.line([(tx, ry1), (tx, ry1 + 10)], fill=(230, 215, 180), width=2)
        souq_h[ry0:ry1, rx0:rx1] = 0.65

    # Wooden display shelves in the middle-back with Armenian ceramic plates & brassware
    for sy in [165, 255, 345]:
        s_draw.rectangle([145, sy, 367, sy + 9], fill=(92, 58, 34), outline=(55, 34, 18))
        souq_h[sy:sy + 9, 145:367] = 0.45

    # Colorful Armenian ceramic plates on shelves
    plate_specs = [
        (178, 142, 22, (24, 82, 168), (235, 240, 245)),
        (230, 140, 24, (32, 128, 96), (240, 230, 190)),
        (284, 142, 22, (184, 48, 38), (242, 222, 150)),
        (334, 143, 21, (28, 72, 156), (235, 240, 245)),
        (185, 232, 23, (176, 118, 32), (245, 232, 185)),
        (242, 230, 25, (22, 92, 168), (240, 245, 250)),
        (302, 232, 23, (42, 134, 78), (238, 228, 180)),
        (344, 234, 19, (168, 38, 48), (240, 220, 160)),
    ]
    for (px, py, pr, c_outer, c_inner) in plate_specs:
        s_draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=c_outer, outline=(245, 235, 200), width=2)
        s_draw.ellipse([px - pr // 2, py - pr // 2, px + pr // 2, py + pr // 2], fill=c_inner)
        s_draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=c_outer)
        souq_h[max(0, py - pr):min(q, py + pr), max(0, px - pr):min(q, px + pr)] = 0.52

    # Hanging glowing brass souq lanterns from top arch
    lanterns = [(195, 68, 16), (256, 54, 20), (318, 68, 16)]
    for (lx, ly, lr) in lanterns:
        s_draw.line([(lx, 0), (lx, ly - lr)], fill=(180, 140, 60), width=2)
        # Warm halo
        s_draw.ellipse([lx - lr * 2, ly - lr * 2, lx + lr * 2, ly + lr * 2], fill=(115, 78, 28))
        s_draw.polygon([
            (lx, ly - lr - 6), (lx + lr, ly), (lx, ly + lr + 6), (lx - lr, ly)
        ], fill=(252, 215, 110), outline=(185, 135, 45))
        souq_h[max(0, ly - lr):min(q, ly + lr), max(0, lx - lr):min(q, lx + lr)] = 0.72
        souq_r[max(0, ly - lr):min(q, ly + lr), max(0, lx - lr):min(q, lx + lr)] = 0.22

    # Spice & copperware baskets on lower counter (y: 380..500)
    s_draw.rectangle([135, 425, 377, 505], fill=(82, 52, 30), outline=(48, 30, 16), width=3)
    spice_mounds = [
        (170, 425, 28, (212, 68, 28)),   # Paprika / sumac
        (225, 425, 30, (228, 168, 24)),  # Turmeric / saffron
        (285, 425, 28, (112, 136, 48)),  # Za'atar
        (340, 425, 26, (148, 32, 52)),   # Deep crimson sumac
    ]
    for (sx, sy, sr, sc) in spice_mounds:
        s_draw.chord([sx - sr, sy - sr, sx + sr, sy + sr], start=180, end=360, fill=sc, outline=(90, 60, 30))
        souq_h[sy - sr:sy, sx - sr:sx + sr] = 0.58

    # Stone arch frame around the shopfront opening
    s_draw.rectangle([0, 0, q - 1, q - 1], outline=(158, 140, 112), width=10)
    souq_arr = np.array(souq_img, dtype=np.float32)
    # Add fine textile texture ripple to rugs
    souq_h += (f_wood - 0.5) * 0.08

    diff[0:q, q:size] = souq_arr
    height[0:q, q:size] = souq_h
    rough[0:q, q:size] = souq_r

    save_pbr_set("wood_doors", diff, height, rough, norm_strength=12.0)


# =============================================================================
# 3. Botanical Foliage, Bark & Wrought-Iron Alpha Cutouts
# =============================================================================

def gen_bark_pbr(size=512):
    """
    bark_{diff,norm,rough}.jpg (512x512):
    Deeply furrowed Mediterranean olive/cypress trunk bark.
    """
    f_vert = gaussian_filter(tileable_fbm((size, size), octaves=5, base_freq=6, seed=801), sigma=(14.0, 1.5), mode="wrap")
    f_vert = (f_vert - f_vert.min()) / (f_vert.max() - f_vert.min() + 1e-6)
    f_gnarl = tileable_fbm((size, size), octaves=4, base_freq=5, seed=802)
    f_micro = tileable_fbm((size, size), octaves=4, base_freq=16, seed=803)

    x = np.arange(size, dtype=np.float32)[None, :]
    y = np.arange(size, dtype=np.float32)[:, None]

    # Twisted vertical bark ridges (characteristic of ancient Jerusalem olive & cypress trunks)
    ridges = 0.5 + 0.5 * np.sin((x + (f_gnarl - 0.5) * 85.0) * (2.0 * math.pi / 28.0))
    furrows = np.power(ridges, 0.65) * 0.65 + f_vert * 0.25 + f_micro * 0.10

    height = furrows

    # Olive/cypress silvery grey-brown bark with dark fissured crevices
    r = 72.0 + furrows * 64.0 + (f_micro - 0.5) * 18.0
    g = 66.0 + furrows * 58.0 + (f_micro - 0.5) * 16.0
    b = 56.0 + furrows * 48.0 + (f_micro - 0.5) * 14.0

    diff = np.stack([r, g, b], axis=-1)
    rough = np.clip(0.94 - furrows * 0.14 + (1.0 - f_micro) * 0.08, 0.76, 0.98)

    save_pbr_set("bark", diff, height, rough, norm_strength=16.0)


def dilate_rgb_into_transparent(rgba_arr, fallback_rgb=(46, 62, 38)):
    """
    Ensure every pixel with alpha == 0 has its RGB channels filled by nearest opaque
    foliage/twig colour blended with dark olive/green fallback (NEVER white!) so
    mipmapping never creates white frosted fringes.
    """
    rgb = rgba_arr[:, :, :3].copy()
    alpha = rgba_arr[:, :, 3]
    mask = alpha > 24
    if np.any(mask):
        _, indices = distance_transform_edt(~mask, return_indices=True)
        nearest_rgb = rgb[indices[0], indices[1]]
        # For semi-transparent and zero-alpha pixels, un-premultiply or replace with nearest opaque RGB
        zero_mask = alpha < 25
        rgb[zero_mask] = (nearest_rgb[zero_mask].astype(np.float32) * 0.7 + np.array(fallback_rgb, dtype=np.float32) * 0.3).astype(np.uint8)
    else:
        rgb[:, :] = fallback_rgb
    out = np.dstack([rgb, alpha]).astype(np.uint8)
    return out


def gen_foliage_olive(size=1024):
    """
    foliage_olive.png (1024x1024 RGBA):
    RGBA alpha-cutout branch cluster with forked woody brown twigs and hundreds of
    overlapping silvery-sage & deep olive-green lance-shaped leaves.
    Background RGB is dark olive green so mipmaps never bleed white.
    """
    rng = np.random.default_rng(901)
    # Start with dark olive-green RGB and alpha=0
    base_bg = (52, 68, 42, 0)
    img = Image.new("RGBA", (size, size), base_bg)
    draw = ImageDraw.Draw(img)

    # Draw forked woody branches radiating from bottom-center & center
    def draw_branch(x0, y0, angle_deg, length, width, depth):
        if depth <= 0 or length < 22:
            return
        rad = math.radians(angle_deg)
        x1 = x0 + math.cos(rad) * length
        y1 = y0 - math.sin(rad) * length
        # Woody grey-brown twig
        c_twig = (
            int(88 + rng.integers(-10, 12)),
            int(76 + rng.integers(-8, 10)),
            int(58 + rng.integers(-8, 10)),
            255,
        )
        draw.line([(x0, y0), (x1, y1)], fill=c_twig, width=max(2, int(width)))
        # Spawn lance-shaped olive leaves along this branch segment
        n_leaves = int(length / 7.5)
        for i in range(n_leaves):
            t = (i + rng.uniform(0.1, 0.9)) / max(1, n_leaves)
            lx = x0 + (x1 - x0) * t + rng.uniform(-14, 14)
            ly = y0 + (y1 - y0) * t + rng.uniform(-14, 14)
            for side in (-1, 1):
                l_ang = angle_deg + side * rng.uniform(28, 68) + rng.uniform(-12, 12)
                l_len = rng.uniform(36, 68)
                l_wid = l_len * rng.uniform(0.20, 0.27)
                draw_olive_leaf(draw, lx, ly, l_ang, l_len, l_wid, rng)

        # Fork into sub-branches
        n_sub = 2 if depth > 2 else 3
        for s in range(n_sub):
            sub_ang = angle_deg + rng.uniform(-38, 38)
            sub_len = length * rng.uniform(0.62, 0.78)
            draw_branch(x1, y1, sub_ang, sub_len, width * 0.68, depth - 1)

    def draw_olive_leaf(d, x, y, angle_deg, length, width, r_gen):
        rad = math.radians(angle_deg)
        ca, sa = math.cos(rad), -math.sin(rad)
        px, py = -sa, ca  # perpendicular

        # Lanceolate leaf polygon (8 points from petiole to acute tip)
        pts = [
            (x, y),
            (x + ca * length * 0.22 + px * width * 0.55, y + sa * length * 0.22 + py * width * 0.55),
            (x + ca * length * 0.52 + px * width * 0.95, y + sa * length * 0.52 + py * width * 0.95),
            (x + ca * length * 0.80 + px * width * 0.58, y + sa * length * 0.80 + py * width * 0.58),
            (x + ca * length, y + sa * length),
            (x + ca * length * 0.80 - px * width * 0.58, y + sa * length * 0.80 - py * width * 0.58),
            (x + ca * length * 0.52 - px * width * 0.95, y + sa * length * 0.52 - py * width * 0.95),
            (x + ca * length * 0.22 - px * width * 0.55, y + sa * length * 0.22 - py * width * 0.55),
        ]
        # Olive trees have both deep olive-green upper leaf surfaces and silvery-sage undersides!
        is_silver_underside = r_gen.random() < 0.38
        if is_silver_underside:
            col = (
                int(134 + r_gen.integers(-16, 18)),
                int(152 + r_gen.integers(-16, 18)),
                int(118 + r_gen.integers(-14, 16)),
                255,
            )
            vein_col = (164, 178, 146, 255)
        else:
            col = (
                int(68 + r_gen.integers(-18, 20)),
                int(94 + r_gen.integers(-20, 22)),
                int(50 + r_gen.integers(-14, 16)),
                255,
            )
            vein_col = (98, 124, 72, 255)

        d.polygon(pts, fill=col)
        # Central midrib vein
        d.line([(x, y), (x + ca * length * 0.92, y + sa * length * 0.92)], fill=vein_col, width=1)

    # Radiating main boughs so the card looks lush from any angle
    main_angles = [90, 52, 128, 22, 158, 205, 335, 250, 290]
    for ang in main_angles:
        draw_branch(size * 0.5, size * 0.54, ang, size * 0.20, 11.0, 4)

    # Extra dense cluster of leaves around the crown core
    for _ in range(420):
        ang_r = rng.uniform(0, 2 * math.pi)
        rad_r = rng.uniform(25, size * 0.42) ** 0.95
        lx = size * 0.5 + math.cos(ang_r) * rad_r
        ly = size * 0.5 + math.sin(ang_r) * rad_r
        if 40 < lx < size - 40 and 40 < ly < size - 40:
            draw_olive_leaf(draw, lx, ly, rng.uniform(0, 360), rng.uniform(38, 66), rng.uniform(8.5, 15.0), rng)

    arr = np.array(img, dtype=np.uint8)
    arr = dilate_rgb_into_transparent(arr, fallback_rgb=(54, 72, 42))
    Image.fromarray(arr, mode="RGBA").save(OUT_DIR / "foliage_olive.png")
    print(f"  [OK] Saved alpha cutout: foliage_olive.png ({size}x{size})")


def gen_foliage_cypress(size=1024):
    """
    foliage_cypress.png (1024x1024 RGBA):
    RGBA alpha-cutout dense deep-green Mediterranean cypress (Cupressus sempervirens)
    scale-leaf sprays with dark green background RGB fill on alpha=0 pixels.
    """
    rng = np.random.default_rng(902)
    base_bg = (28, 48, 26, 0)
    img = Image.new("RGBA", (size, size), base_bg)
    draw = ImageDraw.Draw(img)

    def draw_cypress_spray(x0, y0, angle_deg, length, width, depth):
        if depth <= 0 or length < 14:
            return
        rad = math.radians(angle_deg)
        x1 = x0 + math.cos(rad) * length
        y1 = y0 - math.sin(rad) * length

        if depth >= 3:
            c_stem = (72, 54, 40, 255)
        else:
            c_stem = (
                int(32 + rng.integers(-8, 14)),
                int(64 + rng.integers(-12, 18)),
                int(30 + rng.integers(-8, 12)),
                255,
            )
        draw.line([(x0, y0), (x1, y1)], fill=c_stem, width=max(2, int(width)))

        # Dense scale-leaf branchlets along spray
        steps = int(length / 5.5)
        for i in range(steps):
            t = (i + 0.5) / max(1, steps)
            bx = x0 + (x1 - x0) * t
            by = y0 + (y1 - y0) * t
            for side in (-1, 1):
                s_ang = angle_deg + side * rng.uniform(22, 48)
                s_len = (1.0 - 0.45 * t) * rng.uniform(18, 38)
                s_rad = math.radians(s_ang)
                sx1 = bx + math.cos(s_rad) * s_len
                sy1 = by - math.sin(s_rad) * s_len
                c_leaf = (
                    int(28 + rng.integers(0, 26)),
                    int(56 + rng.integers(0, 36)),
                    int(24 + rng.integers(0, 20)),
                    255,
                )
                draw.line([(bx, by), (sx1, sy1)], fill=c_leaf, width=max(3, int(width * 0.85)))
                # Scale-leaf beaded tufts
                for k in range(3):
                    kt = (k + 1) / 3.0
                    tx = bx + (sx1 - bx) * kt
                    ty = by + (sy1 - by) * kt
                    r = rng.uniform(3.5, 6.5)
                    draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=c_leaf)

        for _ in range(3):
            sub_ang = angle_deg + rng.uniform(-26, 26)
            sub_len = length * rng.uniform(0.60, 0.78)
            draw_cypress_spray(x1, y1, sub_ang, sub_len, width * 0.72, depth - 1)

    # Upward fastigiate cypress sprays
    for i in range(18):
        ang = 90 + rng.uniform(-55, 55)
        sx = size * 0.5 + rng.uniform(-size * 0.18, size * 0.18)
        sy = size * 0.82 - rng.uniform(0, size * 0.35)
        draw_cypress_spray(sx, sy, ang, size * 0.22, 8.0, 4)

    arr = np.array(img, dtype=np.uint8)
    arr = dilate_rgb_into_transparent(arr, fallback_rgb=(28, 50, 26))
    Image.fromarray(arr, mode="RGBA").save(OUT_DIR / "foliage_cypress.png")
    print(f"  [OK] Saved alpha cutout: foliage_cypress.png ({size}x{size})")


def gen_iron_grille(size=512):
    """
    iron_grille.png (512x512 RGBA):
    RGBA alpha-cutout wrought-iron window security bars and ornate Jerusalem balcony scrollwork.
    """
    base_bg = (30, 32, 34, 0)
    img = Image.new("RGBA", (size, size), base_bg)
    draw = ImageDraw.Draw(img)

    iron_dark = (32, 34, 38, 255)
    iron_hi = (58, 62, 68, 255)

    # Outer border frame
    draw.rectangle([6, 6, size - 7, size - 7], outline=iron_dark, width=10)
    draw.rectangle([8, 8, size - 9, size - 9], outline=iron_hi, width=2)

    # Vertical wrought-iron bars
    n_bars = 6
    xs = [int(round((i + 1) * size / (n_bars + 1))) for i in range(n_bars)]
    for x in xs:
        draw.line([(x, 6), (x, size - 6)], fill=iron_dark, width=8)
        draw.line([(x - 1, 6), (x - 1, size - 6)], fill=iron_hi, width=2)

    # Horizontal tie bars
    ys = [int(size * 0.22), int(size * 0.50), int(size * 0.78)]
    for y in ys:
        draw.line([(6, y), (size - 6, y)], fill=iron_dark, width=8)
        draw.line([(6, y - 1), (size - 6, y - 1)], fill=iron_hi, width=2)

    # Ornate C-scrolls and S-scrolls between vertical bars
    for i in range(len(xs) - 1):
        cx = (xs[i] + xs[i + 1]) // 2
        w_span = xs[i + 1] - xs[i] - 12
        for cy in [int(size * 0.11), int(size * 0.36), int(size * 0.64), int(size * 0.89)]:
            rx = w_span // 2
            ry = 22
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=iron_dark, width=6)
            draw.ellipse([cx - rx // 2, cy - ry // 2, cx + rx // 2, cy + ry // 2], outline=iron_dark, width=5)
            # Forged collar clamp at centre
            draw.rectangle([cx - 6, cy - 6, cx + 6, cy + 6], fill=iron_hi, outline=iron_dark)

    arr = np.array(img, dtype=np.uint8)
    arr = dilate_rgb_into_transparent(arr, fallback_rgb=(32, 34, 38))
    Image.fromarray(arr, mode="RGBA").save(OUT_DIR / "iron_grille.png")
    print(f"  [OK] Saved alpha cutout: iron_grille.png ({size}x{size})")


# =============================================================================
# 4. Multilingual Store Names & Street Plaque Atlas (2048x1024 + JSON)
# =============================================================================

def load_font(size, bold=True, serif=False):
    candidates = []
    if serif:
        candidates = [
            "/usr/share/fonts/gnu-free/FreeSerifBold.otf" if bold else "/usr/share/fonts/gnu-free/FreeSerif.otf",
            "/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSerif.ttf",
        ]
    candidates += [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def fit_font(draw, text, max_w, start_size=34, min_size=11, bold=True, serif=False):
    for sz in range(start_size, min_size - 1, -1):
        f = load_font(sz, bold=bold, serif=serif)
        bbox = draw.textbbox((0, 0), text, font=f)
        if (bbox[2] - bbox[0]) <= max_w:
            return f
    return load_font(min_size, bold=bold, serif=serif)


def draw_centered_text(draw, cx, cy, text, font, fill, shadow=None):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = cx - tw / 2.0 - bbox[0]
    ty = cy - th / 2.0 - bbox[1]
    if shadow:
        draw.text((tx + 2, ty + 2), text, font=font, fill=shadow)
    draw.text((tx, ty), text, font=font, fill=fill)


def gen_sign_atlas(width=2048, height=1024):
    """
    Render a clean 4x4 grid of 16 authentic Jerusalem signs in public/textures/sign_atlas.png
    (2048x1024, each cell 512x256) and write their UV bounding boxes
    { id: { u0, v0, u1, v1, aspect } } to public/textures/sign_atlas.json.
    """
    cols, rows = 4, 4
    cw, ch = width // cols, height // rows  # 512 x 256

    atlas = Image.new("RGBA", (width, height), (28, 26, 24, 255))
    draw = ImageDraw.Draw(atlas)

    signs = [
        # 1..4: Cobalt-blue & white Armenian ceramic border Jerusalem municipal street plaques
        {
            "id": "jaffa_gate_plaque",
            "style": "ceramic_plaque",
            "he": "שער יפו",
            "ar": "باب الخليل",
            "en": "JAFFA GATE / BAB EL-KHALIL",
        },
        {
            "id": "omar_sq_plaque",
            "style": "ceramic_plaque",
            "he": "כיכר עומר אבן אל-ח׳טאב",
            "ar": "ميدان عمر بن الخطاب",
            "en": "OMAR IBN EL-KHATTAB SQ.",
        },
        {
            "id": "david_st_plaque",
            "style": "ceramic_plaque",
            "he": "רחוב דוד",
            "ar": "طريق البازار - سويقة علون",
            "en": "DAVID STREET / SUQ EL-BAZAR",
        },
        {
            "id": "latin_rd_plaque",
            "style": "ceramic_plaque",
            "he": "רחוב הפטריארכיה הלטינית",
            "ar": "طريق البطريركية اللاتينية",
            "en": "LATIN PATRIARCHATE RD.",
        },
        # 5..16: Historic hotels, institutions, and souq shopfront fascia boards
        {
            "id": "new_imperial_hotel",
            "style": "fascia",
            "bg": (242, 232, 205),
            "border": (178, 138, 52),
            "text_col": (32, 42, 68),
            "sub_col": (138, 96, 28),
            "title": "NEW IMPERIAL HOTEL — EST. 1884",
            "sub": "فندق امبريال الجديد  •  JAFFA GATE JERUSALEM",
            "serif": True,
        },
        {
            "id": "petra_hotel",
            "style": "fascia",
            "bg": (42, 28, 24),
            "border": (208, 166, 72),
            "text_col": (244, 214, 132),
            "sub_col": (218, 195, 152),
            "title": "HOTEL PETRA — ROOFTOP VIEW",
            "sub": "فندق البتراء  •  HERITAGE HOSPITALITY SINCE 1843",
            "serif": True,
        },
        {
            "id": "christian_info",
            "style": "fascia",
            "bg": (236, 226, 206),
            "border": (34, 78, 136),
            "text_col": (24, 58, 112),
            "sub_col": (88, 76, 62),
            "title": "CHRISTIAN INFORMATION CENTRE",
            "sub": "מרכז מידע נוצרי  •  OMAR IBN EL-KHATTAB SQUARE",
            "serif": False,
        },
        {
            "id": "tower_of_david_museum",
            "style": "fascia",
            "bg": (106, 24, 36),
            "border": (218, 178, 86),
            "text_col": (248, 236, 202),
            "sub_col": (228, 192, 112),
            "title": "TOWER OF DAVID JERUSALEM MUSEUM",
            "sub": "מוזיאון מגדל דוד  •  متحف قلعة داود",
            "serif": True,
        },
        {
            "id": "old_city_bazaar",
            "style": "fascia",
            "bg": (58, 32, 18),
            "border": (216, 162, 58),
            "text_col": (248, 218, 136),
            "sub_col": (230, 188, 128),
            "title": "OLD CITY ORIENTAL BAZAAR & ANTIQUES",
            "sub": "بازار البلدة القديمة للتحف والشرقيات  •  DAVID ST.",
            "serif": True,
        },
        {
            "id": "jerusalem_pottery",
            "style": "ceramic_shop",
            "bg": (245, 242, 232),
            "border": (22, 74, 158),
            "text_col": (18, 58, 134),
            "sub_col": (34, 122, 88),
            "title": "ARMENIAN CERAMICS & JERUSALEM POTTERY",
            "sub": "خزف القدس الأرمني التقليدي  •  HAND-PAINTED TILES",
            "serif": True,
        },
        {
            "id": "money_exchange",
            "style": "fascia",
            "bg": (20, 22, 26),
            "border": (228, 186, 64),
            "text_col": (246, 208, 78),
            "sub_col": (235, 238, 242),
            "title": "AUTHORIZED MONEY EXCHANGE • ATM",
            "sub": "صرافة العملات المعتمدة  •  המרת מט״ח  •  0% COMMISSION",
            "serif": False,
        },
        {
            "id": "falafel_cafe",
            "style": "fascia",
            "bg": (28, 78, 42),
            "border": (232, 184, 56),
            "text_col": (252, 242, 205),
            "sub_col": (244, 198, 78),
            "title": "JAFFA GATE FALAFEL, HUMMUS & FRESH POMEGRANATE JUICE",
            "sub": "فلافل وحمص وعصير رمان طازج  •  FRESH MINT TEA & ARABIC COFFEE",
            "serif": False,
        },
        {
            "id": "st_james_souvenirs",
            "style": "fascia",
            "bg": (24, 52, 102),
            "border": (216, 174, 72),
            "text_col": (248, 238, 212),
            "sub_col": (226, 192, 104),
            "title": "ST. JAMES SOUVENIRS, ICONS & GIFTS",
            "sub": "أيقونات وهدايا الشرق المقدس  •  OLIVE WOOD CARVINGS",
            "serif": True,
        },
        {
            "id": "holy_land_rugs",
            "style": "fascia",
            "bg": (128, 22, 28),
            "border": (224, 178, 68),
            "text_col": (250, 228, 162),
            "sub_col": (238, 208, 152),
            "title": "HOLY LAND CARPETS, KILIMS & TEXTILES",
            "sub": "سجاد وبسط الأراضي المقدسة  •  HANDWOVEN TRIBAL RUGS",
            "serif": True,
        },
        {
            "id": "pilgrim_books",
            "style": "fascia",
            "bg": (26, 62, 46),
            "border": (198, 164, 88),
            "text_col": (242, 234, 210),
            "sub_col": (214, 188, 126),
            "title": "OLD CITY BOOKSHOP & MAPS",
            "sub": "مكتبة وخرائط القدس القديمة  •  HISTORICAL PRINTS & GUIDES",
            "serif": True,
        },
        {
            "id": "swedish_study",
            "style": "fascia",
            "bg": (22, 82, 148),
            "border": (242, 202, 44),
            "text_col": (250, 252, 255),
            "sub_col": (248, 214, 72),
            "title": "SWEDISH CHRISTIAN STUDY CENTRE",
            "sub": "SCSC JERUSALEM  •  LATIN PATRIARCHATE ROAD",
            "serif": False,
        },
    ]

    meta = {}

    for idx, s in enumerate(signs):
        col = idx % cols
        row = idx // cols
        x0 = col * cw
        y0 = row * ch
        x1 = x0 + cw
        y1 = y0 + ch
        cx = x0 + cw / 2.0
        cy = y0 + ch / 2.0

        pad = 12
        bx0, by0, bx1, by1 = x0 + pad, y0 + pad, x1 - pad, y1 - pad

        if s["style"] == "ceramic_plaque":
            # Authentic Jerusalem street plaque: glazed white ceramic tiles framed by
            # cobalt-blue & turquoise Armenian floral border
            draw.rectangle([bx0, by0, bx1, by1], fill=(22, 68, 148), outline=(14, 42, 96), width=4)
            # Decorative corner rosettes & border diamonds
            for tx in range(bx0 + 18, bx1 - 12, 28):
                for ty in [by0 + 10, by1 - 10]:
                    draw.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=(48, 152, 168), outline=(245, 245, 240))
            for ty in range(by0 + 18, by1 - 12, 28):
                for tx in [bx0 + 10, bx1 - 10]:
                    draw.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=(48, 152, 168), outline=(245, 245, 240))

            # Inner white glazed tile bed
            ix0, iy0, ix1, iy1 = bx0 + 22, by0 + 22, bx1 - 22, by1 - 22
            draw.rectangle([ix0, iy0, ix1, iy1], fill=(246, 244, 236), outline=(22, 68, 148), width=3)
            # Subtle 3x2 ceramic tile joint grid lines
            for gx in [ix0 + (ix1 - ix0) // 3, ix0 + 2 * (ix1 - ix0) // 3]:
                draw.line([(gx, iy0), (gx, iy1)], fill=(220, 216, 204), width=1)
            gy = (iy0 + iy1) // 2
            draw.line([(ix0, gy), (ix1, gy)], fill=(220, 216, 204), width=1)

            # Trilingual text: Hebrew top, Arabic middle, English bottom
            f_he = fit_font(draw, s["he"], ix1 - ix0 - 28, start_size=32, bold=True)
            f_ar = fit_font(draw, s["ar"], ix1 - ix0 - 28, start_size=30, bold=True)
            f_en = fit_font(draw, s["en"], ix1 - ix0 - 28, start_size=26, bold=True)

            draw_centered_text(draw, cx, iy0 + (iy1 - iy0) * 0.23, s["he"], f_he, fill=(18, 52, 124))
            draw_centered_text(draw, cx, iy0 + (iy1 - iy0) * 0.52, s["ar"], f_ar, fill=(18, 52, 124))
            draw.line([(ix0 + 24, iy0 + (iy1 - iy0) * 0.69), (ix1 - 24, iy0 + (iy1 - iy0) * 0.69)], fill=(38, 118, 178), width=2)
            draw_centered_text(draw, cx, iy0 + (iy1 - iy0) * 0.83, s["en"], f_en, fill=(16, 44, 108))

        else:
            # Classic Jerusalem Shopfront Fascia / Ceramic Shop Board
            bg = s["bg"]
            border = s["border"]
            draw.rectangle([bx0, by0, bx1, by1], fill=bg, outline=border, width=6)
            draw.rectangle([bx0 + 12, by0 + 12, bx1 - 12, by1 - 12], outline=border, width=2)

            # Corner brass/ceramic medallions
            for (mx, my) in [(bx0 + 22, by0 + 22), (bx1 - 22, by0 + 22), (bx0 + 22, by1 - 22), (bx1 - 22, by1 - 22)]:
                draw.ellipse([mx - 6, my - 6, mx + 6, my + 6], fill=border)

            max_tw = (bx1 - bx0) - 52
            f_title = fit_font(draw, s["title"], max_tw, start_size=30, min_size=11, bold=True, serif=s.get("serif", False))
            f_sub = fit_font(draw, s["sub"], max_tw, start_size=21, min_size=11, bold=False, serif=False)

            shadow_col = (12, 12, 12) if sum(bg) < 380 else None
            draw_centered_text(draw, cx, y0 + ch * 0.38, s["title"], f_title, fill=s["text_col"], shadow=shadow_col)
            # Ornamental divider rule
            draw.line([(bx0 + 48, y0 + ch * 0.56), (bx1 - 48, y0 + ch * 0.56)], fill=border, width=2)
            draw.polygon([(cx, y0 + ch * 0.56 - 5), (cx + 6, y0 + ch * 0.56), (cx, y0 + ch * 0.56 + 5), (cx - 6, y0 + ch * 0.56)], fill=border)
            draw_centered_text(draw, cx, y0 + ch * 0.73, s["sub"], f_sub, fill=s["sub_col"], shadow=shadow_col)

        # Compute standard OpenGL UV coordinates (v=0 at bottom, v=1 at top) AND image-space bounds
        # Note: In Three.js TextureLoader (flipY = True by default), v=1.0 is top of image (y=0)
        # and v=0.0 is bottom of image (y=height).
        u0 = round((x0 + pad) / width, 5)
        u1 = round((x1 - pad) / width, 5)
        v0 = round(1.0 - (y1 - pad) / height, 5)
        v1 = round(1.0 - (y0 + pad) / height, 5)
        aspect = round(((x1 - x0 - 2 * pad) / (y1 - y0 - 2 * pad)), 3)

        meta[s["id"]] = {
            "u0": u0,
            "v0": v0,
            "u1": u1,
            "v1": v1,
            "aspect": aspect,
            "row": row,
            "col": col,
        }

    atlas.convert("RGB").save(OUT_DIR / "sign_atlas.png")
    with open(OUT_DIR / "sign_atlas.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Saved sign atlas: sign_atlas.png ({width}x{height}) + sign_atlas.json ({len(meta)} signs)")


# =============================================================================
# Main Entrypoint
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating procedural PBR textures & signage into {OUT_DIR} ...")

    gen_stone_ashlar(1024)
    gen_stone_boss(1024)
    gen_stone_rubble(1024)
    gen_paving_setts(1024)
    gen_plaster_warm(1024)
    gen_roof_tile(1024)

    gen_wood_doors_atlas(1024)

    gen_bark_pbr(512)
    gen_foliage_olive(1024)
    gen_foliage_cypress(1024)
    gen_iron_grille(512)

    gen_sign_atlas(2048, 1024)

    print("All procedural PBR textures, alpha cutouts, and signage generated successfully!")


if __name__ == "__main__":
    main()
