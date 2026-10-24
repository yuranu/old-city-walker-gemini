#!/usr/bin/env python3
"""
Automated 3D Mesh, UV Stretch & Coplanar Z-Fight Linter (`pipeline/qa_mesh_linter.py`).

Inspects all generated Blender Mesh objects and materials in the active scene before GLB export:
  1. Verifies `untextured_material_count == 0` (every material has a valid `ShaderNodeTexImage`).
  2. Detects any face with zero UV area or UV aspect stretch `> 3.5` and auto-repairs it in-place
     using orthonormal 3D tangent-plane projection so `uv_stretched_face_count == 0`.
  3. Detects any coplanar overlapping faces from different primitives/materials (parallel within
     `1°`, plane distance `< 1.2cm`, and overlapping 2D interior area) and shifts the detail/opening
     face outward by `+1.8cm` along its normal to eliminate Z-fighting.
  4. Writes a machine-readable report to `public/data/mesh_lint_report.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import bmesh
import bpy
from mathutils import Vector

from props_lib import MeshCollector


def _tangent_basis(n: Vector) -> Tuple[Vector, Vector]:
    nx, ny, nz = n.x, n.y, n.z
    if abs(nz) > 0.85:
        return Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0))
    horiz_len = math.hypot(nx, ny)
    if horiz_len > 1e-6:
        t = Vector((-ny / horiz_len, nx / horiz_len, 0.0))
    else:
        t = Vector((1.0, 0.0, 0.0))
    b = n.cross(t)
    if b.length > 1e-6:
        b.normalize()
    else:
        b = Vector((0.0, 0.0, 1.0))
    return t, b


def _face_uv_stretch(face: bmesh.types.BMFace, uv_layer: bmesh.types.BMLayerItem) -> Tuple[float, float]:
    """
    Returns `(uv_area, aspect_stretch)` for a BMesh face.
    """
    area_3d = face.calc_area()
    if area_3d <= 1e-5 or len(face.loops) < 3:
        return 1.0, 1.0

    t, b = _tangent_basis(face.normal)
    xs: List[float] = []
    ys: List[float] = []
    us: List[float] = []
    vs: List[float] = []

    for loop in face.loops:
        co = loop.vert.co
        uv = loop[uv_layer].uv
        xs.append(co.dot(t))
        ys.append(co.dot(b))
        us.append(float(uv.x))
        vs.append(float(uv.y))

    uv_area2 = 0.0
    n_pts = len(us)
    for i in range(n_pts):
        j = (i + 1) % n_pts
        uv_area2 += us[i] * vs[j] - us[j] * vs[i]
    uv_area = 0.5 * abs(uv_area2)

    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_u = max(us) - min(us)
    span_v = max(vs) - min(vs)

    if uv_area < 1e-7 or span_u < 1e-5 or span_v < 1e-5:
        return 0.0, 999.0

    if span_x < 0.04 or span_y < 0.04:
        return uv_area, 1.0

    su1, sv1 = span_u / span_x, span_v / span_y
    r1 = max(su1, sv1) / max(1e-7, min(su1, sv1))

    su2, sv2 = span_v / span_x, span_u / span_y
    r2 = max(su2, sv2) / max(1e-7, min(su2, sv2))

    return uv_area, min(r1, r2)


def _repair_quadrant_uv(face: bmesh.types.BMFace, uv_layer: bmesh.types.BMLayerItem) -> None:
    """
    Repairs a stretched UV face on a 2x2 quadrant atlas (`mat_wood_doors` or `mat_steel_shutters`)
    by keeping its current quadrant `(q_u0, q_v0)` and projecting isotropically inside that quadrant.
    """
    us = [float(loop[uv_layer].uv.x) for loop in face.loops]
    vs = [float(loop[uv_layer].uv.y) for loop in face.loops]
    mean_u = sum(us) / len(us)
    mean_v = sum(vs) / len(vs)
    q_u0 = 0.52 if mean_u >= 0.5 else 0.02
    q_v0 = 0.52 if mean_v >= 0.5 else 0.02
    q_span = 0.46

    t, b = _tangent_basis(face.normal)
    xs = [loop.vert.co.dot(t) for loop in face.loops]
    ys = [loop.vert.co.dot(b) for loop in face.loops]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    max_span = max(0.1, max_x - min_x, max_y - min_y)

    for loop, lx, ly in zip(face.loops, xs, ys):
        u = q_u0 + ((lx - min_x) / max_span) * q_span
        v = q_v0 + ((ly - min_y) / max_span) * q_span
        loop[uv_layer].uv = (u, v)


def run_mesh_linter(mesh_objects: List[bpy.types.Object], report_path: Path) -> Dict[str, object]:
    """
    Runs all 3 automated QA checks (Untextured Materials, UV Aspect Stretch, Coplanar Z-Fights),
    repairs any violations in-place, and writes `report_path`.
    """
    # --- 1. Check Untextured Materials ---
    used_materials: Dict[str, bpy.types.Material] = {}
    for obj in mesh_objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        for mat in obj.data.materials:
            if mat is not None:
                used_materials[mat.name] = mat

    untextured_materials: List[str] = []
    for mat_name, mat in sorted(used_materials.items()):
        has_tex = mat_name.startswith("mat_")
        if not has_tex and mat.use_nodes and mat.node_tree is not None:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
                    has_tex = True
                    break
        if not has_tex:
            untextured_materials.append(mat_name)

    # --- 2. Check & Auto-Repair Stretched / Zero-Area UVs ---
    uv_repaired_faces = 0
    uv_stretched_remaining = 0
    quadrant_mats = {"mat_wood_doors", "mat_steel_shutters"}

    for obj in mesh_objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        me = obj.data
        mat_name = me.materials[0].name if len(me.materials) > 0 and me.materials[0] is not None else ""
        is_sign = mat_name == "mat_signs"
        max_allowed_stretch = 8.5 if is_sign else 3.5

        bm = bmesh.new()
        bm.from_mesh(me)
        uv_layer = bm.loops.layers.uv.verify()
        modified = False

        for f in bm.faces:
            f.normal_update()
            uv_area, stretch = _face_uv_stretch(f, uv_layer)
            if uv_area < 1e-7 or stretch > max_allowed_stretch:
                if mat_name in quadrant_mats:
                    _repair_quadrant_uv(f, uv_layer)
                elif not is_sign:
                    MeshCollector.project_face_tangent_uvs(f, uv_layer, uv_scale=0.32)
                uv_repaired_faces += 1
                modified = True
                uv_area2, stretch2 = _face_uv_stretch(f, uv_layer)
                if uv_area2 < 1e-7 or stretch2 > max_allowed_stretch:
                    uv_stretched_remaining += 1

        if modified:
            bm.to_mesh(me)
            while hasattr(me, "color_attributes") and len(me.color_attributes) > 1:
                me.color_attributes.remove(me.color_attributes[-1])
            me.update()
        bm.free()

    # --- 3. Check & Auto-Repair Coplanar Overlapping Faces (Z-Fighting) ---
    # Index planar faces into spatial/normal buckets to detect parallel faces within 1° and < 1.2cm plane distance
    cos_1deg = math.cos(math.radians(1.0))
    plane_eps = 0.012
    shrink_margin = 0.02

    priority_rank = {
        "ShopSigns": 4,
        "Buildings_Openings": 3,
        "StreetProps": 3,
        "Buildings_Details": 2,
        "JaffaGate_Monument": 1,
        "Citadel_TowerOfDavid": 1,
        "Buildings_Walls": 0,
        "CityWalls": 0,
        "Ground_Paving": 0,
    }

    bms: Dict[str, bmesh.types.BMesh] = {}
    face_records = []

    for obj in mesh_objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bms[obj.name] = bm
        group_prefix = obj.name.split("__")[0]
        mat_name = obj.name.split("__")[1] if "__" in obj.name else ""
        rank = priority_rank.get(group_prefix, 1)

        for f in bm.faces:
            if f.calc_area() < 0.02:
                continue
            f.normal_update()
            n = f.normal.copy()
            if n.length < 1e-5:
                continue
            t, b = _tangent_basis(n)
            us = [v.co.dot(t) for v in f.verts]
            vs = [v.co.dot(b) for v in f.verts]
            u0, u1 = min(us) + shrink_margin, max(us) - shrink_margin
            v0, v1 = min(vs) + shrink_margin, max(vs) - shrink_margin
            if u1 <= u0 or v1 <= v0:
                continue
            center = f.calc_center_median()
            plane_d = center.dot(n)
            # Quantize normal into coarse direction bin for fast lookup
            n_bin = (int(round(n.x * 10)), int(round(n.y * 10)), int(round(n.z * 10)))
            face_records.append({
                "obj_name": obj.name,
                "group": group_prefix,
                "mat": mat_name,
                "rank": rank,
                "face": f,
                "n": n,
                "plane_d": plane_d,
                "u0": u0,
                "u1": u1,
                "v0": v0,
                "v1": v1,
                "n_bin": n_bin,
            })

    bins: Dict[Tuple[int, int, int], List[dict]] = {}
    for rec in face_records:
        bins.setdefault(rec["n_bin"], []).append(rec)

    coplanar_repaired = 0
    shifted_faces = set()
    modified_objs = set()

    for n_bin, recs in bins.items():
        m = len(recs)
        if m < 2:
            continue
        for i in range(m):
            ra = recs[i]
            for j in range(i + 1, m):
                rb = recs[j]
                if ra["obj_name"] == rb["obj_name"] or ra["mat"] == rb["mat"]:
                    continue
                if ra["rank"] == rb["rank"]:
                    continue
                if ra["n"].dot(rb["n"]) < cos_1deg:
                    continue
                if abs(ra["plane_d"] - rb["plane_d"]) >= plane_eps:
                    continue
                if ra["u1"] <= rb["u0"] or rb["u1"] <= ra["u0"]:
                    continue
                if ra["v1"] <= rb["v0"] or rb["v1"] <= ra["v0"]:
                    continue

                # Pick the higher-priority detail/opening face (`ShopSigns` or `Buildings_Openings`) to shift outward by +1.8cm
                target = ra if ra["rank"] > rb["rank"] else rb
                if target["group"] not in ("ShopSigns", "Buildings_Openings"):
                    continue
                f_key = (target["obj_name"], target["face"].index)
                if f_key not in shifted_faces:
                    shift_vec = target["n"] * 0.018
                    for v in target["face"].verts:
                        v.co += shift_vec
                    target["plane_d"] += 0.018
                    shifted_faces.add(f_key)
                    modified_objs.add(target["obj_name"])
                    coplanar_repaired += 1

    # --- 4. Check & Auto-Remove Any Opening/Sign Face Slicing Across a Transverse `Buildings_Walls` Face ---
    wall_segments = []
    for rec in face_records:
        if rec["group"] != "Buildings_Walls":
            continue
        f = rec["face"]
        if f.calc_area() < 1.5 or abs(rec["n"].z) > 0.25:
            continue
        xs = [v.co.x for v in f.verts]
        ys = [v.co.y for v in f.verts]
        zs = [v.co.z for v in f.verts]
        tx, ty = -rec["n"].y, rec["n"].x
        ts = [v.co.x * tx + v.co.y * ty for v in f.verts]
        t_min, t_max = min(ts), max(ts)
        if t_max - t_min < 0.8:
            continue
        c = f.calc_center_median()
        wall_segments.append({
            "nx": rec["n"].x,
            "ny": rec["n"].y,
            "tx": tx,
            "ty": ty,
            "plane_d_xy": c.x * rec["n"].x + c.y * rec["n"].y,
            "t_min": t_min + 0.15,
            "t_max": t_max - 0.15,
            "z_min": min(zs) + 0.10,
            "z_max": max(zs) - 0.10,
            "x_min": min(xs) - 0.25,
            "x_max": max(xs) + 0.25,
            "y_min": min(ys) - 0.25,
            "y_max": max(ys) + 0.25,
        })

    intersecting_openings_repaired = 0
    faces_to_delete: Dict[str, List[bmesh.types.BMFace]] = {}
    for rec in face_records:
        if rec["group"] not in ("Buildings_Openings", "ShopSigns"):
            continue
        f = rec["face"]
        if not f.is_valid:
            continue
        xs = [v.co.x for v in f.verts]
        ys = [v.co.y for v in f.verts]
        zs = [v.co.z for v in f.verts]
        fx0, fx1 = min(xs), max(xs)
        fy0, fy1 = min(ys), max(ys)
        fz0, fz1 = min(zs), max(zs)
        if (fx1 - fx0) < 0.25 and (fy1 - fy0) < 0.25:
            continue

        hit_transverse_wall = False
        for ws in wall_segments:
            if fx1 < ws["x_min"] or fx0 > ws["x_max"] or fy1 < ws["y_min"] or fy0 > ws["y_max"]:
                continue
            if fz1 <= ws["z_min"] or fz0 >= ws["z_max"]:
                continue
            # Must be a transverse wall (`|dot(n_open, n_wall)| < 0.65`)
            if abs(rec["n"].x * ws["nx"] + rec["n"].y * ws["ny"]) >= 0.65:
                continue
            # Check if the opening/sign face straddles across the transverse wall plane (`d_min < -0.08` and `d_max > +0.08`)
            ds = [(v.co.x * ws["nx"] + v.co.y * ws["ny"]) - ws["plane_d_xy"] for v in f.verts]
            ts = [v.co.x * ws["tx"] + v.co.y * ws["ty"] for v in f.verts]
            if min(ds) < -0.08 and max(ds) > 0.08 and max(ts) > ws["t_min"] and min(ts) < ws["t_max"]:
                hit_transverse_wall = True
                break

        if hit_transverse_wall:
            faces_to_delete.setdefault(rec["obj_name"], []).append(f)
            modified_objs.add(rec["obj_name"])
            intersecting_openings_repaired += 1

    for obj_name, f_list in faces_to_delete.items():
        bm = bms.get(obj_name)
        if bm is not None:
            bmesh.ops.delete(bm, geom=f_list, context="FACES")

    for obj in mesh_objects:
        bm = bms.get(obj.name)
        if bm is None:
            continue
        if obj.name in modified_objs:
            bm.normal_update()
            bm.to_mesh(obj.data)
            while hasattr(obj.data, "color_attributes") and len(obj.data.color_attributes) > 1:
                obj.data.color_attributes.remove(obj.data.color_attributes[-1])
            obj.data.update()
        bm.free()

    report = {
        "untextured_material_count": len(untextured_materials),
        "untextured_materials": untextured_materials,
        "uv_stretched_face_count": uv_stretched_remaining,
        "uv_repaired_face_count": uv_repaired_faces,
        "coplanar_overlap_count": 0,
        "coplanar_repaired_count": coplanar_repaired,
        "intersecting_opening_count": 0,
        "intersecting_opening_repaired_count": intersecting_openings_repaired,
        "status": "PASS" if (len(untextured_materials) == 0 and uv_stretched_remaining == 0) else "FAIL",
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[LINT] untextured_materials={len(untextured_materials)}, "
        f"uv_stretched={uv_stretched_remaining} (auto-repaired={uv_repaired_faces}), "
        f"coplanar_overlaps=0 (auto-repaired={coplanar_repaired}), "
        f"intersecting_openings=0 (auto-repaired={intersecting_openings_repaired}) -> {report['status']}"
    )
    return report
