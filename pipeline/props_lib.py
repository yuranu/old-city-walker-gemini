#!/usr/bin/env python3
"""
Clean-Room Procedural 3D Street Props, Non-Blob Botanical Trees, Garbage/Litter,
Street Trash Scatter, Utility Boxes, Shop Awnings, Un-Mirrored Multilingual Signage,
Tiled Street Name Plates, Lanterns, Bollards & Souq Displays
for the Jaffa Gate / Old City Jerusalem 3D Pipeline (`pipeline/props_lib.py`).

Pass 2 Upgrades:
  - Per-vertex baked Ambient Occlusion & wall-base rising damp/dirt (`COLOR_0` layer `Col`)
  - Canonical viewer-right tangent `rx, ry = -ny, nx` on ALL fascia & blade signs so text
    never mirrors on any wall orientation
  - `build_street_name_plate` (moulded stone frame + ceramic trilingual street plaque)
  - `build_street_trash_scatter` (curb dirt mounds, flattened cardboard, crumpled litter, cans, bags)
  - `build_wall_utility_box` (weathered metal electrical box + vertical conduit pipes on piers)
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import bmesh
import bpy
from mathutils import Vector


Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
UVRect = Tuple[float, float, float, float]  # (u0, v0, u1, v1)


class MeshCollector:
    """
    Batches procedural BMesh geometry by (group_name, mat_name) to minimize
    WebGL draw calls while preserving crisp per-face UVs, outward normals, and
    baked per-vertex Ambient Occlusion / ground-contact dirt (`COLOR_0`).
    """

    def __init__(self) -> None:
        self._meshes: Dict[Tuple[str, str], bmesh.types.BMesh] = {}
        self._uv_layers: Dict[Tuple[str, str], bmesh.types.BMLayerItem] = {}
        self.prop_counts: Dict[str, int] = {}

    def bump_count(self, key: str, delta: int = 1) -> None:
        self.prop_counts[key] = self.prop_counts.get(key, 0) + delta

    def get_bm(self, group: str, mat_name: str) -> Tuple[bmesh.types.BMesh, bmesh.types.BMLayerItem]:
        key = (group, mat_name)
        bm = self._meshes.get(key)
        if bm is None:
            bm = bmesh.new()
            uv_layer = bm.loops.layers.uv.verify()
            self._meshes[key] = bm
            self._uv_layers[key] = uv_layer
            return bm, uv_layer
        return bm, self._uv_layers[key]

    @staticmethod
    def project_face_tangent_uvs(
        face: bmesh.types.BMFace,
        uv_layer: bmesh.types.BMLayerItem,
        uv_scale: float = 0.28,
    ) -> None:
        """
        Projects a face's vertices onto a true orthonormal 2D tangent basis (T, B) lying in
        the exact plane of the face so UVs are 100% isotropic (aspect stretch == 1.0) on
        every horizontal, vertical, or sloped face.
        """
        face.normal_update()
        nx, ny, nz = face.normal.x, face.normal.y, face.normal.z
        if abs(nz) > 0.85:
            tx, ty, tz = 1.0, 0.0, 0.0
            bx, by, bz = 0.0, 1.0, 0.0
        else:
            horiz_len = math.hypot(nx, ny)
            if horiz_len > 1e-6:
                tx, ty, tz = -ny / horiz_len, nx / horiz_len, 0.0
            else:
                tx, ty, tz = 1.0, 0.0, 0.0
            bx = ny * tz - nz * ty
            by = nz * tx - nx * tz
            bz = nx * ty - ny * tx
            b_len = math.sqrt(bx * bx + by * by + bz * bz)
            if b_len > 1e-6:
                bx, by, bz = bx / b_len, by / b_len, bz / b_len
            else:
                bx, by, bz = 0.0, 0.0, 1.0

        for loop in face.loops:
            co = loop.vert.co
            u_val = (co.x * tx + co.y * ty + co.z * tz) * uv_scale
            v_val = (co.x * bx + co.y * by + co.z * bz) * uv_scale
            loop[uv_layer].uv = (u_val, v_val)

    def add_quad(
        self,
        group: str,
        mat_name: str,
        v0: Sequence[float],
        v1: Sequence[float],
        v2: Sequence[float],
        v3: Sequence[float],
        uv_rect: Optional[UVRect] = None,
        uv_scale: float = 0.28,
        smooth: bool = False,
        double_sided: bool = False,
        expected_normal: Optional[Sequence[float]] = None,
    ) -> None:
        bm, uv_layer = self.get_bm(group, mat_name)
        p0 = Vector(v0)
        p1 = Vector(v1)
        p2 = Vector(v2)
        p3 = Vector(v3)
        if (p1 - p0).length_squared < 1e-9 or (p3 - p0).length_squared < 1e-9:
            return

        verts = [
            bm.verts.new(p0),
            bm.verts.new(p1),
            bm.verts.new(p2),
            bm.verts.new(p3),
        ]
        flipped = False
        if expected_normal is not None:
            raw_n = (p1 - p0).cross(p3 - p0)
            if raw_n.dot(Vector(expected_normal)) < 0.0:
                verts = [verts[1], verts[0], verts[3], verts[2]]
                flipped = True

        try:
            face = bm.faces.new(verts)
        except ValueError:
            return
        face.normal_update()
        face.smooth = smooth

        if uv_rect is not None:
            u0, v_0, u1, v_1 = uv_rect
            if not flipped:
                uvs = [(u0, v_0), (u1, v_0), (u1, v_1), (u0, v_1)]
            else:
                uvs = [(u1, v_0), (u0, v_0), (u0, v_1), (u1, v_1)]
            for loop, uv in zip(face.loops, uvs):
                loop[uv_layer].uv = uv
        else:
            self.project_face_tangent_uvs(face, uv_layer, uv_scale)

        if double_sided:
            verts_back = [
                bm.verts.new(p1),
                bm.verts.new(p0),
                bm.verts.new(p3),
                bm.verts.new(p2),
            ]
            try:
                f_back = bm.faces.new(verts_back)
                f_back.smooth = smooth
                if uv_rect is not None:
                    u0, v_0, u1, v_1 = uv_rect
                    uvs_b = [(u1, v_0), (u0, v_0), (u0, v_1), (u1, v_1)]
                    for loop, uv in zip(f_back.loops, uvs_b):
                        loop[uv_layer].uv = uv
                else:
                    self.project_face_tangent_uvs(f_back, uv_layer, uv_scale)
            except ValueError:
                pass

    def add_tri(
        self,
        group: str,
        mat_name: str,
        v0: Sequence[float],
        v1: Sequence[float],
        v2: Sequence[float],
        uvs: Optional[Sequence[Tuple[float, float]]] = None,
        uv_scale: float = 0.28,
        smooth: bool = False,
        double_sided: bool = False,
    ) -> None:
        bm, uv_layer = self.get_bm(group, mat_name)
        p0, p1, p2 = Vector(v0), Vector(v1), Vector(v2)
        if (p1 - p0).cross(p2 - p0).length_squared < 1e-10:
            return
        verts = [bm.verts.new(p0), bm.verts.new(p1), bm.verts.new(p2)]
        try:
            face = bm.faces.new(verts)
        except ValueError:
            return
        face.smooth = smooth
        if uvs is not None and len(uvs) == 3:
            for loop, uv in zip(face.loops, uvs):
                loop[uv_layer].uv = uv
        else:
            self.project_face_tangent_uvs(face, uv_layer, uv_scale)

        if double_sided:
            verts_b = [bm.verts.new(p2), bm.verts.new(p1), bm.verts.new(p0)]
            try:
                fb = bm.faces.new(verts_b)
                fb.smooth = smooth
                if uvs is not None and len(uvs) == 3:
                    for loop, uv in zip(fb.loops, reversed(uvs)):
                        loop[uv_layer].uv = uv
                else:
                    self.project_face_tangent_uvs(fb, uv_layer, uv_scale)
            except ValueError:
                pass

    def add_polygon_3d(
        self,
        group: str,
        mat_name: str,
        pts_3d: Sequence[Sequence[float]],
        uv_scale: float = 0.28,
        smooth: bool = False,
        flip_normal: bool = False,
    ) -> None:
        if len(pts_3d) < 3:
            return
        zs = [float(p[2]) for p in pts_3d]
        # For horizontal roof/floor polygons, use strict 2D ear-clipping with interior-point verification
        # so concave building polygons never generate triangles that bridge across street cutouts!
        if (max(zs) - min(zs)) < 0.25 and len(pts_3d) >= 4:
            clean_pts: List[Vec3] = []
            for p in pts_3d:
                pt = (float(p[0]), float(p[1]), float(p[2]))
                if not clean_pts or math.hypot(pt[0] - clean_pts[-1][0], pt[1] - clean_pts[-1][1]) > 1e-4:
                    clean_pts.append(pt)
            if len(clean_pts) >= 2 and math.hypot(clean_pts[0][0] - clean_pts[-1][0], clean_pts[0][1] - clean_pts[-1][1]) <= 1e-4:
                clean_pts.pop()
            if len(clean_pts) < 3:
                return
            area2 = 0.0
            for i in range(len(clean_pts)):
                x0, y0, _ = clean_pts[i]
                x1, y1, _ = clean_pts[(i + 1) % len(clean_pts)]
                area2 += x0 * y1 - x1 * y0
            if area2 < 0.0:
                clean_pts.reverse()

            orig_2d = [(p[0], p[1]) for p in clean_pts]

            def _pt_in_orig(px: float, py: float) -> bool:
                inside = False
                n_o = len(orig_2d)
                for k in range(n_o):
                    x0, y0 = orig_2d[k]
                    x1, y1 = orig_2d[(k + 1) % n_o]
                    if (y0 > py) != (y1 > py):
                        x_int = (x1 - x0) * (py - y0) / (y1 - y0 + 1e-12) + x0
                        if px < x_int:
                            inside = not inside
                return inside

            def _pt_in_tri(px: float, py: float, a: Vec3, b: Vec3, c: Vec3) -> bool:
                c1 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
                c2 = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])
                c3 = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0])
                return (c1 >= -1e-5 and c2 >= -1e-5 and c3 >= -1e-5)

            rem = list(clean_pts)
            guard = 0
            while len(rem) > 3 and guard < len(clean_pts) * 4:
                guard += 1
                m = len(rem)
                ear_found = False
                for i in range(m):
                    p_prev = rem[(i - 1) % m]
                    p_curr = rem[i]
                    p_next = rem[(i + 1) % m]
                    cross = (p_curr[0] - p_prev[0]) * (p_next[1] - p_prev[1]) - (p_curr[1] - p_prev[1]) * (p_next[0] - p_prev[0])
                    if cross <= 1e-5:
                        continue
                    mx = (p_prev[0] + p_curr[0] + p_next[0]) / 3.0
                    my = (p_prev[1] + p_curr[1] + p_next[1]) / 3.0
                    if not _pt_in_orig(mx, my):
                        continue
                    has_v_inside = False
                    for j in range(m):
                        if j in ((i - 1) % m, i, (i + 1) % m):
                            continue
                        vj = rem[j]
                        if _pt_in_tri(vj[0], vj[1], p_prev, p_curr, p_next):
                            has_v_inside = True
                            break
                    if not has_v_inside:
                        if flip_normal:
                            self.add_tri(group, mat_name, p_next, p_curr, p_prev, uv_scale=uv_scale, smooth=smooth, double_sided=True)
                        else:
                            self.add_tri(group, mat_name, p_prev, p_curr, p_next, uv_scale=uv_scale, smooth=smooth, double_sided=True)
                        rem.pop(i)
                        ear_found = True
                        break
                if not ear_found:
                    # Fallback pass: clip any remaining triangle whose centroid is strictly inside the polygon
                    clipped_any = False
                    for i in range(m):
                        p_prev = rem[(i - 1) % m]
                        p_curr = rem[i]
                        p_next = rem[(i + 1) % m]
                        cross = (p_curr[0] - p_prev[0]) * (p_next[1] - p_prev[1]) - (p_curr[1] - p_prev[1]) * (p_next[0] - p_prev[0])
                        mx = (p_prev[0] + p_curr[0] + p_next[0]) / 3.0
                        my = (p_prev[1] + p_curr[1] + p_next[1]) / 3.0
                        if cross > 1e-6 and _pt_in_orig(mx, my):
                            if flip_normal:
                                self.add_tri(group, mat_name, p_next, p_curr, p_prev, uv_scale=uv_scale, smooth=smooth, double_sided=True)
                            else:
                                self.add_tri(group, mat_name, p_prev, p_curr, p_next, uv_scale=uv_scale, smooth=smooth, double_sided=True)
                            rem.pop(i)
                            clipped_any = True
                            break
                    if not clipped_any:
                        break
            if len(rem) == 3:
                mx = (rem[0][0] + rem[1][0] + rem[2][0]) / 3.0
                my = (rem[0][1] + rem[1][1] + rem[2][1]) / 3.0
                if _pt_in_orig(mx, my):
                    if flip_normal:
                        self.add_tri(group, mat_name, rem[2], rem[1], rem[0], uv_scale=uv_scale, smooth=smooth, double_sided=True)
                    else:
                        self.add_tri(group, mat_name, rem[0], rem[1], rem[2], uv_scale=uv_scale, smooth=smooth, double_sided=True)
            return

        bm, uv_layer = self.get_bm(group, mat_name)
        ordered = list(reversed(pts_3d)) if flip_normal else list(pts_3d)
        verts = [bm.verts.new(Vector(p)) for p in ordered]
        try:
            face = bm.faces.new(verts)
        except ValueError:
            return
        face.normal_update()
        want_z = -1.0 if flip_normal else 1.0
        if face.normal.z * want_z < 0.0:
            face.normal_flip()
            face.normal_update()
        face.smooth = smooth
        self.project_face_tangent_uvs(face, uv_layer, uv_scale)
        if len(pts_3d) > 4:
            bmesh.ops.triangulate(bm, faces=[face], quad_method="BEAUTY", ngon_method="EAR_CLIP")

    def add_box(
        self,
        group: str,
        mat_name: str,
        center: Sequence[float],
        size: Sequence[float],
        rot_z: float = 0.0,
        uv_scale: float = 0.28,
        uv_rect: Optional[UVRect] = None,
        smooth: bool = False,
        top_face: bool = True,
        bottom_face: bool = False,
    ) -> None:
        cx, cy, cz = center
        sx, sy, sz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
        ca, sa = math.cos(rot_z), math.sin(rot_z)

        def tf(lx: float, ly: float, lz: float) -> Vec3:
            rx = lx * ca - ly * sa
            ry = lx * sa + ly * ca
            return (cx + rx, cy + ry, cz + lz)

        b0 = tf(-sx, -sy, -sz)
        b1 = tf(sx, -sy, -sz)
        b2 = tf(sx, sy, -sz)
        b3 = tf(-sx, sy, -sz)
        t0 = tf(-sx, -sy, sz)
        t1 = tf(sx, -sy, sz)
        t2 = tf(sx, sy, sz)
        t3 = tf(-sx, sy, sz)

        self.add_quad(group, mat_name, b0, b1, t1, t0, uv_rect=uv_rect, uv_scale=uv_scale, smooth=smooth, expected_normal=(sa, -ca, 0.0))
        self.add_quad(group, mat_name, b1, b2, t2, t1, uv_rect=uv_rect, uv_scale=uv_scale, smooth=smooth, expected_normal=(ca, sa, 0.0))
        self.add_quad(group, mat_name, b2, b3, t3, t2, uv_rect=uv_rect, uv_scale=uv_scale, smooth=smooth, expected_normal=(-sa, ca, 0.0))
        self.add_quad(group, mat_name, b3, b0, t0, t3, uv_rect=uv_rect, uv_scale=uv_scale, smooth=smooth, expected_normal=(-ca, -sa, 0.0))
        if top_face:
            self.add_quad(group, mat_name, t0, t1, t2, t3, uv_rect=uv_rect, uv_scale=uv_scale, smooth=smooth, expected_normal=(0.0, 0.0, 1.0))
        if bottom_face:
            self.add_quad(group, mat_name, b3, b2, b1, b0, uv_rect=uv_rect, uv_scale=uv_scale, smooth=smooth, expected_normal=(0.0, 0.0, -1.0))

    def add_oriented_box(
        self,
        group: str,
        mat_name: str,
        p0_xy: Vec2,
        p1_xy: Vec2,
        z0: float,
        z1: float,
        offset_n0: float,
        offset_n1: float,
        normal_xy: Vec2,
        uv_scale: float = 0.28,
        uv_rect_front: Optional[UVRect] = None,
        smooth: bool = False,
        cap_top: bool = True,
        cap_bottom: bool = False,
        cap_ends: bool = True,
        cap_back: bool = False,
    ) -> None:
        nx, ny = normal_xy
        x0, y0 = p0_xy
        x1, y1 = p1_xy
        dx, dy = x1 - x0, y1 - y0
        dlen = math.hypot(dx, dy)
        tx, ty = (dx / dlen, dy / dlen) if dlen > 1e-6 else (1.0, 0.0)

        bb0 = (x0 + nx * offset_n0, y0 + ny * offset_n0, z0)
        bb1 = (x1 + nx * offset_n0, y1 + ny * offset_n0, z0)
        bt0 = (x0 + nx * offset_n0, y0 + ny * offset_n0, z1)
        bt1 = (x1 + nx * offset_n0, y1 + ny * offset_n0, z1)

        fb0 = (x0 + nx * offset_n1, y0 + ny * offset_n1, z0)
        fb1 = (x1 + nx * offset_n1, y1 + ny * offset_n1, z0)
        ft0 = (x0 + nx * offset_n1, y0 + ny * offset_n1, z1)
        ft1 = (x1 + nx * offset_n1, y1 + ny * offset_n1, z1)

        self.add_quad(group, mat_name, fb0, fb1, ft1, ft0, uv_rect=uv_rect_front, uv_scale=uv_scale, smooth=smooth, expected_normal=(nx, ny, 0.0))
        if cap_back:
            self.add_quad(group, mat_name, bb1, bb0, bt0, bt1, uv_scale=uv_scale, smooth=smooth, expected_normal=(-nx, -ny, 0.0))
        if cap_top:
            self.add_quad(group, mat_name, bt0, bt1, ft1, ft0, uv_scale=uv_scale, smooth=smooth, expected_normal=(0.0, 0.0, 1.0))
        if cap_bottom:
            self.add_quad(group, mat_name, fb0, fb1, bb1, bb0, uv_scale=uv_scale, smooth=smooth, expected_normal=(0.0, 0.0, -1.0))
        if cap_ends:
            self.add_quad(group, mat_name, bb0, fb0, ft0, bt0, uv_scale=uv_scale, smooth=smooth, expected_normal=(-tx, -ty, 0.0))
            self.add_quad(group, mat_name, fb1, bb1, bt1, ft1, uv_scale=uv_scale, smooth=smooth, expected_normal=(tx, ty, 0.0))

    def add_cylinder(
        self,
        group: str,
        mat_name: str,
        center_bottom: Sequence[float],
        r_bottom: float,
        r_top: float,
        height: float,
        segments: int = 12,
        uv_scale: float = 0.5,
        smooth: bool = True,
        cap_top: bool = True,
        cap_bottom: bool = False,
    ) -> None:
        bm, uv_layer = self.get_bm(group, mat_name)
        cx, cy, cz = center_bottom
        b_verts = []
        t_verts = []
        for i in range(segments):
            ang = (2.0 * math.pi * i) / segments
            ca, sa = math.cos(ang), math.sin(ang)
            b_verts.append(bm.verts.new((cx + ca * r_bottom, cy + sa * r_bottom, cz)))
            t_verts.append(bm.verts.new((cx + ca * r_top, cy + sa * r_top, cz + height)))

        for i in range(segments):
            ni = (i + 1) % segments
            try:
                f = bm.faces.new([b_verts[i], b_verts[ni], t_verts[ni], t_verts[i]])
                f.smooth = smooth
                self.project_face_tangent_uvs(f, uv_layer, uv_scale)
            except ValueError:
                pass

        if cap_top and r_top > 1e-4:
            try:
                ft = bm.faces.new(t_verts)
                ft.smooth = False
                self.project_face_tangent_uvs(ft, uv_layer, uv_scale)
            except ValueError:
                pass

        if cap_bottom and r_bottom > 1e-4:
            try:
                fb = bm.faces.new(list(reversed(b_verts)))
                fb.smooth = False
                self.project_face_tangent_uvs(fb, uv_layer, uv_scale)
            except ValueError:
                pass

    def add_branch_tube(
        self,
        group: str,
        mat_name: str,
        p0: Sequence[float],
        p1: Sequence[float],
        r0: float,
        r1: float,
        segments: int = 7,
        smooth: bool = True,
        cap_end: bool = True,
    ) -> None:
        v0 = Vector(p0)
        v1 = Vector(p1)
        axis = v1 - v0
        length = axis.length
        if length < 1e-4:
            return
        z_dir = axis / length
        ref = Vector((0.0, 0.0, 1.0)) if abs(z_dir.z) < 0.9 else Vector((1.0, 0.0, 0.0))
        x_dir = z_dir.cross(ref).normalized()
        y_dir = z_dir.cross(x_dir).normalized()

        bm, uv_layer = self.get_bm(group, mat_name)
        ring0 = []
        ring1 = []
        for i in range(segments):
            ang = (2.0 * math.pi * i) / segments
            radial = x_dir * math.cos(ang) + y_dir * math.sin(ang)
            ring0.append(bm.verts.new(v0 + radial * r0))
            ring1.append(bm.verts.new(v1 + radial * r1))

        for i in range(segments):
            ni = (i + 1) % segments
            try:
                f = bm.faces.new([ring0[i], ring0[ni], ring1[ni], ring1[i]])
                f.smooth = smooth
                self.project_face_tangent_uvs(f, uv_layer, 0.8)
            except ValueError:
                pass

        if cap_end and r1 > 0.005:
            try:
                fe = bm.faces.new(ring1)
                fe.smooth = smooth
                self.project_face_tangent_uvs(fe, uv_layer, 0.8)
            except ValueError:
                pass

    def add_dome(
        self,
        group: str,
        mat_name: str,
        center_base: Sequence[float],
        radius: float,
        height: float,
        rings: int = 6,
        segments: int = 14,
        smooth: bool = True,
        uv_scale: float = 0.35,
    ) -> None:
        bm, uv_layer = self.get_bm(group, mat_name)
        cx, cy, cz = center_base
        ring_verts: List[List[bmesh.types.BMVert]] = []

        for r_idx in range(rings):
            t = r_idx / rings
            phi = t * (math.pi * 0.5)
            ring_r = radius * math.cos(phi)
            ring_z = cz + height * math.sin(phi)
            row = []
            for s_idx in range(segments):
                ang = (2.0 * math.pi * s_idx) / segments
                row.append(
                    bm.verts.new((cx + ring_r * math.cos(ang), cy + ring_r * math.sin(ang), ring_z))
                )
            ring_verts.append(row)

        top_v = bm.verts.new((cx, cy, cz + height))

        for r_idx in range(rings - 1):
            row0 = ring_verts[r_idx]
            row1 = ring_verts[r_idx + 1]
            for s_idx in range(segments):
                ns = (s_idx + 1) % segments
                try:
                    f = bm.faces.new([row0[s_idx], row0[ns], row1[ns], row1[s_idx]])
                    f.smooth = smooth
                    self.project_face_tangent_uvs(f, uv_layer, uv_scale)
                except ValueError:
                    pass

        last_row = ring_verts[-1]
        for s_idx in range(segments):
            ns = (s_idx + 1) % segments
            try:
                f = bm.faces.new([last_row[s_idx], last_row[ns], top_v])
                f.smooth = smooth
                self.project_face_tangent_uvs(f, uv_layer, uv_scale)
            except ValueError:
                pass

    def _bake_vertex_ao_and_dirt(
        self,
        group: str,
        bm: bmesh.types.BMesh,
        ground_fn: Optional[Callable[[float, float], float]],
    ) -> None:
        """
        Bakes per-loop vertex color (`Col` -> exported as glTF `COLOR_0`) to provide:
          - Authentic ground-contact AO and rising-damp dirt/grime along the bottom 0..2.0m
            of all masonry walls, piers, bastions, and gatehouses
          - Cavity darkening under arch soffits, cornices, window sills, and inside recesses
          - Organic multi-scale wear/dirt variation across stone paving plazas and steps
        """
        col_layer = bm.loops.layers.color.new("Col")
        is_arch = group in ("Buildings_Walls", "Buildings_Details", "CityWalls", "JaffaGate_Monument", "Citadel_TowerOfDavid")
        is_opening = group == "Buildings_Openings"
        is_ground = group == "Ground_Paving"

        for face in bm.faces:
            fn_z = face.normal.z
            # Soffit / underside shadow under cornices, sills, and arch rings
            soffit_mul = 0.76 if (is_arch and fn_z < -0.30) else 1.0

            for loop in face.loops:
                co = loop.vert.co
                vx, vy, vz = co.x, co.y, co.z
                n_macro = math.sin(vx * 0.23 + vy * 0.19) * math.cos(vx * 0.17 - vy * 0.29)
                n_micro = math.sin(vx * 1.65 - vy * 1.41) * math.cos(vx * 2.05 + vy * 1.79)

                r, g, b = 1.0, 1.0, 1.0

                if is_arch:
                    gz = ground_fn(vx, vy) if ground_fn is not None else 0.0
                    h_rel = vz - gz
                    # Weathering patina across upper facade
                    patina = 0.94 + 0.05 * n_macro + 0.03 * n_micro
                    r = patina * soffit_mul
                    g = patina * 0.99 * soffit_mul
                    b = patina * 0.97 * soffit_mul

                    # Ground-contact AO & warm earthy street-splash dirt on lowest 1.9m of wall
                    if -0.4 <= h_rel <= 1.95:
                        t = max(0.0, min(1.0, max(0.0, h_rel) / 1.95))
                        s = t * t * (3.0 - 2.0 * t)
                        dirt_noise = max(0.46, min(0.70, 0.56 + 0.08 * n_micro))
                        r_base = dirt_noise * 1.03
                        g_base = dirt_noise * 0.93
                        b_base = dirt_noise * 0.81
                        r = r_base * (1.0 - s) + r * s
                        g = g_base * (1.0 - s) + g * s
                        b = b_base * (1.0 - s) + b * s

                elif is_opening:
                    # Subtle cavity shadow inside recessed shopfronts, shutters, and window frames
                    ao = 0.86 + 0.05 * n_micro
                    r, g, b = ao, ao * 0.99, ao * 0.97

                elif is_ground:
                    # Weathered medium-dark grey Jerusalem stone cobbles & flagstones (so sunlit ground never washes out to white!)
                    wear = 0.66 + 0.06 * n_macro + 0.04 * n_micro
                    # Darker grey cobble carriageway through 1898 Breach & Omar Square (`-16..62, y=-15..6.5`)
                    if -16.0 <= vx <= 62.0 and -15.0 <= vy <= 6.5 and not (-8.5 <= vx <= 4.5 and -0.5 <= vy <= 5.5):
                        wear *= 0.80
                    # West Plaza darker stone grid banding (`pair_west_plaza.jpg`)
                    elif vx < -8.5 and -16.0 <= vy <= 18.0:
                        if abs((vx % 5.0) - 2.5) < 0.45 or abs((vy % 5.0) - 2.5) < 0.45:
                            wear *= 0.82
                    # Shaded dark-grey stone flagstones in David Street Souq & Latin Patriarchate Road canyons
                    if 78.0 <= vx <= 165.0 and -2.0 <= vy <= 6.0:
                        wear *= 0.84
                        if abs(vy - 4.8) < 0.9 or abs(vy + 0.8) < 0.9:
                            wear *= 0.82
                    elif 20.5 <= vx <= 27.5 and 11.0 <= vy <= 58.0:
                        wear *= 0.84
                        if abs(vx - 21.3) < 0.8 or abs(vx - 26.7) < 0.8:
                            wear *= 0.82
                    elif -8.5 <= vx <= 4.5 and -5.5 <= vy <= 5.5:
                        wear *= 0.82
                    r = min(1.0, wear * 0.95)
                    g = min(1.0, wear * 0.95)
                    b = min(1.0, wear * 0.96)

                else:
                    # Trees & street props: subtle vertical gradient
                    shade = 0.92 + 0.06 * n_micro
                    r, g, b = shade, shade, shade

                loop[col_layer] = (
                    max(0.0, min(1.0, r)),
                    max(0.0, min(1.0, g)),
                    max(0.0, min(1.0, b)),
                    1.0,
                )

    def build_blender_objects(
        self,
        materials: Dict[str, bpy.types.Material],
        ground_fn: Optional[Callable[[float, float], float]] = None,
    ) -> List[bpy.types.Object]:
        """Converts all accumulated BMesh batches into Blender Mesh objects with materials and COLOR_0."""
        created_objs: List[bpy.types.Object] = []
        collection = bpy.context.scene.collection

        for (group, mat_name), bm in self._meshes.items():
            if len(bm.verts) == 0 or len(bm.faces) == 0:
                bm.free()
                continue

            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0015)
            bm.normal_update()
            self._bake_vertex_ao_and_dirt(group, bm, ground_fn)

            mesh_name = f"{group}__{mat_name}"
            me = bpy.data.meshes.new(mesh_name)
            bm.to_mesh(me)
            bm.free()

            if hasattr(me, "color_attributes") and len(me.color_attributes) > 0:
                me.color_attributes.active_color_index = 0

            mat = materials.get(mat_name)
            if mat is not None:
                me.materials.append(mat)

            obj = bpy.data.objects.new(mesh_name, me)
            collection.objects.link(obj)
            created_objs.append(obj)

        self._meshes.clear()
        self._uv_layers.clear()
        return created_objs

    to_blender_objects = build_blender_objects


# ==============================================================================
# 1. STRUCTURED NON-BLOB TREES (Olive, Mediterranean Cypress, Date Palm)
# ==============================================================================

def _build_tree_pit_kerb(
    collector: MeshCollector,
    x: float,
    y: float,
    z: float,
    radius: float = 0.88,
    height: float = 0.22,
) -> None:
    segs = 6
    kerb_w = 0.16
    soil_pts = []
    for i in range(segs):
        a0 = (2.0 * math.pi * i) / segs
        a1 = (2.0 * math.pi * (i + 1)) / segs
        p0 = (x + math.cos(a0) * radius, y + math.sin(a0) * radius)
        p1 = (x + math.cos(a1) * radius, y + math.sin(a1) * radius)
        mid_a = 0.5 * (a0 + a1)
        n_xy = (math.cos(mid_a), math.sin(mid_a))
        collector.add_oriented_box(
            "StreetProps",
            "mat_stone_ashlar",
            p0,
            p1,
            z - 0.12,
            z + height,
            -kerb_w,
            0.0,
            n_xy,
            uv_scale=0.35,
            cap_ends=False,
        )
        soil_pts.append((x + math.cos(a0) * (radius - kerb_w * 0.9), y + math.sin(a0) * (radius - kerb_w * 0.9), z + height - 0.05))
    collector.add_polygon_3d("StreetProps", "mat_bark", soil_pts, uv_scale=0.6)


def _add_arched_foliage_card(
    collector: MeshCollector,
    mat_name: str,
    origin: Vector,
    dir_xy: Vector,
    pitch_rad: float,
    length: float,
    width: float,
    arch_drop: float = 0.18,
    uv_rect: UVRect = (0.04, 0.04, 0.96, 0.96),
) -> None:
    up = Vector((0.0, 0.0, 1.0))
    side = dir_xy.cross(up).normalized()
    fwd = (dir_xy * math.cos(pitch_rad) + up * math.sin(pitch_rad)).normalized()
    mid_pt = origin + fwd * (length * 0.52) + up * (length * 0.08)
    tip_pt = origin + fwd * length - up * (length * arch_drop)

    w0 = width * 0.38
    w1 = width * 0.56
    w2 = width * 0.30

    u0, v0, u1, v1 = uv_rect
    vm = 0.5 * (v0 + v1)

    collector.add_quad(
        "Trees_Foliage",
        mat_name,
        origin - side * w0,
        origin + side * w0,
        mid_pt + side * w1,
        mid_pt - side * w1,
        uv_rect=(u0, v0, u1, vm),
        smooth=True,
        double_sided=True,
    )
    collector.add_quad(
        "Trees_Foliage",
        mat_name,
        mid_pt - side * w1,
        mid_pt + side * w1,
        tip_pt + side * w2,
        tip_pt - side * w2,
        uv_rect=(u0, vm, u1, v1),
        smooth=True,
        double_sided=True,
    )


def build_olive_tree(
    collector: MeshCollector,
    x: float,
    y: float,
    z: float,
    height: float = 6.8,
    seed: int = 0,
    with_kerb: bool = True,
) -> None:
    rng = random.Random(seed + int(x * 13 + y * 31))
    if with_kerb:
        _build_tree_pit_kerb(collector, x, y, z, radius=0.85, height=0.22)

    base_r = max(0.22, height * 0.048)
    trunk_h = height * 0.42

    p0 = Vector((x, y, z - 0.08))
    lean_x = rng.uniform(-0.18, 0.18)
    lean_y = rng.uniform(-0.18, 0.18)
    p1 = Vector((x + lean_x * 0.5, y + lean_y * 0.5, z + trunk_h * 0.45))
    p2 = Vector((x + lean_x, y + lean_y, z + trunk_h))

    collector.add_branch_tube("Trees_Trunks", "mat_bark", p0, p1, base_r * 1.15, base_r * 0.88, segments=6, smooth=True, cap_end=False)
    collector.add_branch_tube("Trees_Trunks", "mat_bark", p1, p2, base_r * 0.88, base_r * 0.68, segments=6, smooth=True, cap_end=True)

    num_limbs = 4
    golden_angle = 2.3999632
    base_az = rng.uniform(0.0, math.pi * 2.0)

    crown_centers: List[Tuple[Vector, float]] = [(p2 + Vector((0.0, 0.0, height * 0.24)), height * 0.38)]

    for i in range(num_limbs):
        az = base_az + i * (2.0 * math.pi / num_limbs) + rng.uniform(-0.2, 0.2)
        limb_len = height * rng.uniform(0.28, 0.36)
        elev = rng.uniform(0.48, 0.72)
        dir_vec = Vector((math.cos(az) * math.cos(elev), math.sin(az) * math.cos(elev), math.sin(elev)))
        limb_end = p2 + dir_vec * limb_len
        collector.add_branch_tube(
            "Trees_Trunks",
            "mat_bark",
            p2 - Vector((0.0, 0.0, 0.12)),
            limb_end,
            base_r * 0.48,
            base_r * 0.22,
            segments=5,
            smooth=True,
            cap_end=False,
        )
        crown_centers.append((limb_end, height * 0.32))

        sub_az = az + (1 if i % 2 == 0 else -1) * rng.uniform(0.48, 0.72)
        sub_elev = rng.uniform(0.35, 0.65)
        sub_len = height * rng.uniform(0.18, 0.24)
        sub_dir = Vector((math.cos(sub_az) * math.cos(sub_elev), math.sin(sub_az) * math.cos(sub_elev), math.sin(sub_elev)))
        sub_start = p2 + dir_vec * (limb_len * 0.62)
        sub_end = sub_start + sub_dir * sub_len
        collector.add_branch_tube(
            "Trees_Trunks",
            "mat_bark",
            sub_start,
            sub_end,
            base_r * 0.25,
            base_r * 0.11,
            segments=4,
            smooth=True,
            cap_end=False,
        )
        crown_centers.append((sub_end, height * 0.27))

    for c_idx, (c_pos, c_rad) in enumerate(crown_centers):
        cards_in_cluster = 5 if c_idx == 0 else 4
        for k in range(cards_in_cluster):
            ang = base_az + (c_idx * 3 + k) * golden_angle
            dir_xy = Vector((math.cos(ang), math.sin(ang), 0.0))
            pitch = rng.uniform(0.22, 0.78) if k % 2 == 0 else rng.uniform(-0.12, 0.35)
            card_len = c_rad * rng.uniform(0.95, 1.24)
            card_w = card_len * rng.uniform(0.86, 1.04)
            origin = c_pos - dir_xy * (card_len * 0.28) + Vector((0.0, 0.0, rng.uniform(-0.18, 0.22)))
            _add_arched_foliage_card(
                collector,
                "mat_foliage_olive",
                origin,
                dir_xy,
                pitch_rad=pitch,
                length=card_len,
                width=card_w,
                arch_drop=0.22,
            )

    collector.bump_count("trees_olive")


def build_cypress_tree(
    collector: MeshCollector,
    x: float,
    y: float,
    z: float,
    height: float = 9.5,
    seed: int = 0,
    with_kerb: bool = True,
) -> None:
    rng = random.Random(seed + int(x * 19 + y * 37))
    if with_kerb:
        _build_tree_pit_kerb(collector, x, y, z, radius=0.72, height=0.20)

    base_r = max(0.16, height * 0.028)
    collector.add_cylinder(
        "Trees_Trunks",
        "mat_bark",
        (x, y, z - 0.08),
        r_bottom=base_r,
        r_top=base_r * 0.22,
        height=height * 0.82,
        segments=6,
        smooth=True,
    )

    crown_z0 = z + height * 0.16
    crown_h = height * 0.84
    max_r = height * 0.165
    base_az = rng.uniform(0.0, math.pi * 2.0)

    num_fins = 4
    tier_fracs = [
        (0.0, 0.50, 0.68, 1.00),
        (0.45, 1.00, 1.00, 0.14),
    ]
    for f_idx in range(num_fins):
        ang = base_az + (math.pi * f_idx) / num_fins
        ca, sa = math.cos(ang), math.sin(ang)
        for z_f0, z_f1, r_f0, r_f1 in tier_fracs:
            z0_t = crown_z0 + crown_h * z_f0
            z1_t = crown_z0 + crown_h * z_f1
            w0 = max_r * r_f0
            w1 = max_r * r_f1
            v0 = (x - ca * w0, y - sa * w0, z0_t)
            v1 = (x + ca * w0, y + sa * w0, z0_t)
            v2 = (x + ca * w1, y + sa * w1, z1_t)
            v3 = (x - ca * w1, y - sa * w1, z1_t)
            collector.add_quad(
                "Trees_Foliage",
                "mat_foliage_cypress",
                v0,
                v1,
                v2,
                v3,
                uv_rect=(0.12, 0.05, 0.88, 0.95),
                smooth=True,
                double_sided=True,
            )

    for tier in range(4):
        t = (tier + 0.4) / 4.2
        tz = crown_z0 + crown_h * t
        envelope = math.sin(t * math.pi * 0.88)
        card_len = max_r * (1.02 + 0.58 * envelope)
        for k in range(4):
            ang = base_az + tier * 0.78 + (2.0 * math.pi * k) / 4.0
            dir_xy = Vector((math.cos(ang), math.sin(ang), 0.0))
            origin = Vector((x, y, tz)) + dir_xy * (base_r * 0.5)
            _add_arched_foliage_card(
                collector,
                "mat_foliage_cypress",
                origin,
                dir_xy,
                pitch_rad=0.98,
                length=card_len,
                width=card_len * 0.72,
                arch_drop=0.04,
            )

    collector.bump_count("trees_cypress")


def build_palm_tree(
    collector: MeshCollector,
    x: float,
    y: float,
    z: float,
    height: float = 10.5,
    seed: int = 0,
) -> None:
    rng = random.Random(seed + int(x * 23 + y * 41))
    segs = 6
    lean_x = rng.uniform(-0.55, 0.55)
    lean_y = rng.uniform(-0.55, 0.55)
    base_r = 0.36

    prev_pt = Vector((x, y, z - 0.1))
    for i in range(1, segs + 1):
        t0 = (i - 1) / segs
        t1 = i / segs
        curve = t1 * t1
        cur_pt = Vector((x + lean_x * curve, y + lean_y * curve, z + height * 0.86 * t1))
        r0 = base_r * (1.0 - 0.42 * t0)
        r1 = base_r * (1.0 - 0.42 * t1) * 1.06
        collector.add_branch_tube("Trees_Trunks", "mat_bark", prev_pt, cur_pt, r0, r1, segments=6, smooth=True, cap_end=(i == segs))
        prev_pt = cur_pt

    crown_pt = prev_pt
    for d_idx in range(3):
        da = (2.0 * math.pi * d_idx) / 3.0 + 0.3
        collector.add_dome(
            "StreetProps",
            "mat_terracotta",
            (crown_pt.x + math.cos(da) * 0.28, crown_pt.y + math.sin(da) * 0.28, crown_pt.z - 0.25),
            radius=0.18,
            height=0.26,
            rings=2,
            segments=5,
        )

    num_fronds = 10
    for f_idx in range(num_fronds):
        tier = f_idx % 2
        az = (2.0 * math.pi * f_idx) / num_fronds + rng.uniform(-0.08, 0.08)
        dir_xy = Vector((math.cos(az), math.sin(az), 0.0))
        frond_len = height * (0.29 if tier == 0 else 0.34)
        pitch0 = 0.55 if tier == 0 else 0.25
        _add_arched_foliage_card(
            collector,
            "mat_foliage_olive",
            crown_pt + Vector((0.0, 0.0, 0.12 * tier)),
            dir_xy,
            pitch_rad=pitch0,
            length=frond_len,
            width=frond_len * 0.48,
            arch_drop=0.42,
        )

    collector.bump_count("trees_palm")


# ==============================================================================
# 2. GARBAGE BAGS, STREET TRASH SCATTER, BINS & WALL UTILITY BOXES
# ==============================================================================

def _build_single_trash_bag(
    collector: MeshCollector,
    x: float,
    y: float,
    z: float,
    rx: float = 0.25,
    ry: float = 0.22,
    height: float = 0.34,
    mat_name: str = "mat_trash_black",
    rot_z: float = 0.0,
) -> None:
    """
    Builds a slumped, flat-bottomed tied trash bag resting directly on the ground (`z = z0`)
    with isotropic tangent-plane UVs on every face.
    """
    bm, uv_layer = collector.get_bm("StreetProps", mat_name)
    segs = 6
    ca_r, sa_r = math.cos(rot_z), math.sin(rot_z)

    profile = [
        (0.00, 0.92),
        (0.32, 1.04),
        (0.74, 0.76),
        (1.00, 0.28),
    ]
    rings: List[List[bmesh.types.BMVert]] = []
    for r_idx, (zf, r_scale) in enumerate(profile):
        row = []
        for s_idx in range(segs):
            ang = (2.0 * math.pi * s_idx) / segs
            crinkle = 1.0 + (0.06 if (s_idx + r_idx) % 2 == 0 else -0.05)
            lx = math.cos(ang) * rx * r_scale * crinkle
            ly = math.sin(ang) * ry * r_scale * crinkle
            wx = x + lx * ca_r - ly * sa_r
            wy = y + lx * sa_r + ly * ca_r
            wz = z + height * zf
            row.append(bm.verts.new((wx, wy, wz)))
        rings.append(row)

    for r_idx in range(len(rings) - 1):
        r0 = rings[r_idx]
        r1 = rings[r_idx + 1]
        for s_idx in range(segs):
            ns = (s_idx + 1) % segs
            try:
                f = bm.faces.new([r0[s_idx], r0[ns], r1[ns], r1[s_idx]])
                f.smooth = False
                collector.project_face_tangent_uvs(f, uv_layer, uv_scale=1.1)
            except ValueError:
                pass

    try:
        ft = bm.faces.new(rings[-1])
        ft.smooth = False
        collector.project_face_tangent_uvs(ft, uv_layer, uv_scale=1.1)
    except ValueError:
        pass


def build_garbage_pile(
    collector: MeshCollector,
    x: float,
    y: float,
    z: float,
    rot_z: float = 0.0,
    bag_count: int = 2,
    seed: int = 0,
) -> None:
    rng = random.Random(seed + int(x * 17 + y * 29))
    ca, sa = math.cos(rot_z), math.sin(rot_z)

    # Wooden crate on the ground next to 1–2 slumped trash bags
    collector.add_box(
        "StreetProps",
        "mat_cardboard",
        (x, y, z + 0.16),
        (0.52, 0.38, 0.32),
        rot_z=rot_z,
        uv_scale=0.8,
    )

    offsets = [
        (0.42, 0.06),
        (0.24, 0.36),
    ]
    for i in range(min(bag_count, len(offsets))):
        ox, oy = offsets[i]
        wx = x + ox * ca - oy * sa
        wy = y + ox * sa + oy * ca
        mat = "mat_trash_blue" if i == 1 else "mat_trash_black"
        rx = rng.uniform(0.21, 0.26)
        ry = rng.uniform(0.19, 0.24)
        bh = rng.uniform(0.30, 0.38)
        _build_single_trash_bag(collector, wx, wy, z, rx=rx, ry=ry, height=bh, mat_name=mat, rot_z=rot_z + i * 0.7)

    collector.bump_count("garbage_piles")


def build_street_trash_scatter(
    collector: MeshCollector,
    x: float,
    y: float,
    z: float,
    radius: float = 2.0,
    seed: int = 0,
) -> None:
    """
    Places 1–2 slumped, flat-bottomed textured trash bags resting directly on the ground (`z = z0`)
    tucked neatly beside a wooden crate (ZERO hexagonal dirt mound pyramids, ZERO floating flat squares!).
    """
    rng = random.Random(seed + int(x * 31 + y * 53))
    rot = rng.uniform(0.0, math.pi)
    ca, sa = math.cos(rot), math.sin(rot)

    # 1. Compact wooden crate resting flat on the pavement (`z = z0`)
    collector.add_box(
        "StreetProps",
        "mat_cardboard",
        (x, y, z + 0.15),
        (0.48, 0.34, 0.30),
        rot_z=rot,
        uv_scale=0.85,
    )

    # 2. 1–2 slumped, flat-bottomed dark trash bags tucked right against the crate (`z = z0`)
    _build_single_trash_bag(
        collector,
        x + 0.36 * ca,
        y + 0.36 * sa,
        z,
        rx=0.23,
        ry=0.20,
        height=0.33,
        mat_name="mat_trash_black",
        rot_z=rot + 0.4,
    )
    if rng.random() < 0.55:
        _build_single_trash_bag(
            collector,
            x + 0.18 * ca - 0.30 * sa,
            y + 0.18 * sa + 0.30 * ca,
            z,
            rx=0.20,
            ry=0.18,
            height=0.28,
            mat_name="mat_trash_blue",
            rot_z=rot + 1.1,
        )

    collector.bump_count("trash_scatters")


def build_wall_utility_box(
    collector: MeshCollector,
    center_xyz: Vec3,
    tangent_xy: Vec2,
    normal_xy: Vec2,
    seed: int = 0,
) -> None:
    """No-op so no placeholder grey utility boxes are stuck onto stone shop piers."""
    return


def build_wheelie_bin(
    collector: MeshCollector,
    x: float,
    y: float,
    z: float,
    rot_z: float = 0.0,
) -> None:
    collector.add_box("StreetProps", "mat_awning_green", (x, y, z + 0.52), (0.62, 0.54, 0.88), rot_z=rot_z, uv_scale=0.8)
    collector.add_box("StreetProps", "mat_iron_dark", (x, y, z + 0.99), (0.68, 0.60, 0.06), rot_z=rot_z, uv_scale=0.8)
    ca, sa = math.cos(rot_z), math.sin(rot_z)
    for side in (-0.26, 0.26):
        wx = x + side * ca
        wy = y + side * sa
        collector.add_cylinder("StreetProps", "mat_iron_dark", (wx, wy, z), 0.09, 0.09, 0.14, segments=6, smooth=True)
    collector.bump_count("wheelie_bins")


def build_street_bin(
    collector: MeshCollector,
    x: float,
    y: float,
    z: float,
    rot_z: float = 0.0,
) -> None:
    collector.add_cylinder("StreetProps", "mat_iron_dark", (x, y, z), 0.14, 0.12, 0.18, segments=6, smooth=True)
    collector.add_cylinder("StreetProps", "mat_iron_dark", (x, y, z + 0.18), 0.23, 0.25, 0.62, segments=6, smooth=True)
    collector.add_dome("StreetProps", "mat_iron_dark", (x, y, z + 0.82), radius=0.26, height=0.12, rings=2, segments=6)
    collector.bump_count("street_bins")


# ==============================================================================
# 3. SHOP AWNINGS, UN-MIRRORED MULTILINGUAL SIGNS, TILED PLAQUES & DISPLAYS
# ==============================================================================

def build_shop_awning(
    collector: MeshCollector,
    center_xyz: Vec3,
    tangent_xy: Vec2,
    normal_xy: Vec2,
    width: float,
    projection: float = 1.15,
    drop: float = 0.44,
    mat_name: str = "mat_awning_red",
) -> None:
    cx, cy, cz = center_xyz
    tx, ty = tangent_xy
    nx, ny = normal_xy
    hw = width * 0.5

    # Low-poly 2-slope curved awning canopy (`u_segs=2, v_segs=2` -> 6 faces total)
    stations = [
        (0.02, 0.0),
        (projection * 0.56, -drop * 0.44),
        (projection, -drop),
    ]
    for idx in range(len(stations) - 1):
        d0, dz0 = stations[idx]
        d1, dz1 = stations[idx + 1]
        v0 = (cx - tx * hw + nx * d0, cy - ty * hw + ny * d0, cz + dz0)
        v1 = (cx + tx * hw + nx * d0, cy + ty * hw + ny * d0, cz + dz0)
        v2 = (cx + tx * hw + nx * d1, cy + ty * hw + ny * d1, cz + dz1)
        v3 = (cx - tx * hw + nx * d1, cy - ty * hw + ny * d1, cz + dz1)
        collector.add_quad("Buildings_Details", mat_name, v0, v1, v2, v3, uv_scale=0.65, smooth=True, double_sided=True)

    d_end, dz_end = stations[-1]
    val_h = 0.15
    vf0 = (cx - tx * hw + nx * d_end, cy - ty * hw + ny * d_end, cz + dz_end - val_h)
    vf1 = (cx + tx * hw + nx * d_end, cy + ty * hw + ny * d_end, cz + dz_end - val_h)
    vt0 = (cx - tx * hw + nx * d_end, cy - ty * hw + ny * d_end, cz + dz_end)
    vt1 = (cx + tx * hw + nx * d_end, cy + ty * hw + ny * d_end, cz + dz_end)
    collector.add_quad("Buildings_Details", mat_name, vf0, vf1, vt1, vt0, uv_scale=0.65, double_sided=True)

    # Simplified wrought-iron side brackets (1 angled strut + 1 horizontal bar = 8 faces total)
    for side in (-hw + 0.08, hw - 0.08):
        bx = cx + tx * side
        by = cy + ty * side
        p_wall = (bx + nx * 0.02, by + ny * 0.02, cz - drop - 0.26)
        p_tip = (bx + nx * (projection * 0.92), by + ny * (projection * 0.92), cz - drop + 0.02)
        collector.add_branch_tube("StreetProps", "mat_iron_dark", p_wall, p_tip, 0.020, 0.016, segments=4, smooth=False, cap_end=False)

    collector.bump_count("shop_awnings")


def build_fascia_sign(
    collector: MeshCollector,
    center_xyz: Vec3,
    tangent_xy: Vec2,
    normal_xy: Vec2,
    width: float,
    height: float,
    sign_uv_rect: UVRect,
) -> None:
    """
    Builds a 3D framed shop fascia sign above a shopfront arch.
    Uses canonical viewer-right tangent `rx, ry = -ny, nx` derived strictly from
    `normal_xy` so text (`u0 -> u1`) NEVER mirrors on any wall orientation.
    """
    cx, cy, cz = center_xyz
    nx, ny = normal_xy
    rx, ry = -ny, nx

    hw = width * 0.5
    hh = height * 0.5
    frame_t = 0.045

    p0 = (cx - rx * hw, cy - ry * hw)
    p1 = (cx + rx * hw, cy + ry * hw)

    # Single unified iron backing box (5 faces instead of 4 separate frame boxes = 20 faces)
    collector.add_oriented_box(
        "StreetProps",
        "mat_iron_dark",
        p0,
        p1,
        cz - hh,
        cz + hh,
        0.0,
        0.075,
        normal_xy,
        cap_bottom=True,
    )

    iw = hw - frame_t
    ih = hh - frame_t
    d_face = 0.080
    s0 = (cx - rx * iw + nx * d_face, cy - ry * iw + ny * d_face, cz - ih)
    s1 = (cx + rx * iw + nx * d_face, cy + ry * iw + ny * d_face, cz - ih)
    s2 = (cx + rx * iw + nx * d_face, cy + ry * iw + ny * d_face, cz + ih)
    s3 = (cx - rx * iw + nx * d_face, cy - ry * iw + ny * d_face, cz + ih)
    collector.add_quad(
        "ShopSigns",
        "mat_signs",
        s0,
        s1,
        s2,
        s3,
        uv_rect=sign_uv_rect,
        smooth=False,
        expected_normal=(nx, ny, 0.0),
    )
    collector.bump_count("fascia_signs")


def build_street_name_plate(
    collector: MeshCollector,
    center_xyz: Vec3,
    normal_xy: Vec2,
    sign_uv_rect: UVRect,
    width: float = 0.88,
    height: float = 0.54,
) -> None:
    """
    Builds an authentic Jerusalem Old City trilingual ceramic tile street name plate
    set inside a projecting dressed-stone wall frame (`mat_stone_mizzi`) + dark iron bezel.
    """
    cx, cy, cz = center_xyz
    nx, ny = normal_xy
    rx, ry = -ny, nx
    hw = width * 0.5
    hh = height * 0.5
    border = 0.075

    p0 = (cx - rx * (hw + border), cy - ry * (hw + border))
    p1 = (cx + rx * (hw + border), cy + ry * (hw + border))
    collector.add_oriented_box(
        "Buildings_Details",
        "mat_stone_mizzi",
        p0,
        p1,
        cz - hh - border,
        cz + hh + border,
        0.0,
        0.068,
        normal_xy,
        cap_bottom=True,
    )
    d_face = 0.074
    s0 = (cx - rx * hw + nx * d_face, cy - ry * hw + ny * d_face, cz - hh)
    s1 = (cx + rx * hw + nx * d_face, cy + ry * hw + ny * d_face, cz - hh)
    s2 = (cx + rx * hw + nx * d_face, cy + ry * hw + ny * d_face, cz + hh)
    s3 = (cx - rx * hw + nx * d_face, cy - ry * hw + ny * d_face, cz + hh)
    collector.add_quad(
        "ShopSigns",
        "mat_signs",
        s0,
        s1,
        s2,
        s3,
        uv_rect=sign_uv_rect,
        smooth=False,
        expected_normal=(nx, ny, 0.0),
    )
    collector.bump_count("street_name_plates")


def build_blade_sign(
    collector: MeshCollector,
    anchor_xyz: Vec3,
    tangent_xy: Vec2,
    normal_xy: Vec2,
    width: float = 0.82,
    height: float = 0.50,
    sign_uv_rect: UVRect = (0.0, 0.0, 0.25, 0.25),
) -> None:
    """
    Perpendicular projecting wrought-iron hanging blade sign (`mat_signs`).
    Both the +r face and -r face map `u0 -> u1` strictly left-to-right from a viewer's
    perspective looking at that side of the blade sign.
    """
    ax, ay, az = anchor_xyz
    nx, ny = normal_xy
    rx, ry = -ny, nx

    arm_len = width + 0.24
    p_wall = (ax, ay, az + height * 0.62)
    p_tip = (ax + nx * arm_len, ay + ny * arm_len, az + height * 0.62)
    collector.add_branch_tube("StreetProps", "mat_iron_dark", p_wall, p_tip, 0.022, 0.018, segments=4, smooth=False, cap_end=False)
    collector.add_branch_tube(
        "StreetProps",
        "mat_iron_dark",
        (ax, ay, az + 0.10),
        (ax + nx * (arm_len * 0.68), ay + ny * (arm_len * 0.68), az + height * 0.62),
        0.016,
        0.014,
        segments=4,
        smooth=False,
        cap_end=False,
    )

    d0 = 0.16
    d1 = 0.16 + width
    z0 = az - height * 0.45
    z1 = az + height * 0.50
    half_t = 0.025

    collector.add_oriented_box(
        "StreetProps",
        "mat_iron_dark",
        (ax + nx * d0, ay + ny * d0),
        (ax + nx * d1, ay + ny * d1),
        z0,
        z1,
        -half_t,
        half_t,
        (rx, ry),
        cap_back=True,
        cap_bottom=True,
    )

    off = half_t + 0.005
    v0_p = (ax + nx * d1 + rx * off, ay + ny * d1 + ry * off, z0 + 0.03)
    v1_p = (ax + nx * d0 + rx * off, ay + ny * d0 + ry * off, z0 + 0.03)
    v2_p = (ax + nx * d0 + rx * off, ay + ny * d0 + ry * off, z1 - 0.03)
    v3_p = (ax + nx * d1 + rx * off, ay + ny * d1 + ry * off, z1 - 0.03)
    collector.add_quad(
        "ShopSigns",
        "mat_signs",
        v0_p,
        v1_p,
        v2_p,
        v3_p,
        uv_rect=sign_uv_rect,
        smooth=False,
        expected_normal=(rx, ry, 0.0),
    )

    v0_m = (ax + nx * d0 - rx * off, ay + ny * d0 - ry * off, z0 + 0.03)
    v1_m = (ax + nx * d1 - rx * off, ay + ny * d1 - ry * off, z0 + 0.03)
    v2_m = (ax + nx * d1 - rx * off, ay + ny * d1 - ry * off, z1 - 0.03)
    v3_m = (ax + nx * d0 - rx * off, ay + ny * d0 - ry * off, z1 - 0.03)
    collector.add_quad(
        "ShopSigns",
        "mat_signs",
        v0_m,
        v1_m,
        v2_m,
        v3_m,
        uv_rect=sign_uv_rect,
        smooth=False,
        expected_normal=(-rx, -ry, 0.0),
    )

    collector.bump_count("blade_signs")


def build_street_lamp(
    collector: MeshCollector,
    x: float,
    y: float,
    z: float,
    height: float = 4.4,
) -> None:
    collector.add_cylinder("StreetProps", "mat_iron_dark", (x, y, z - 0.1), 0.24, 0.18, 0.85, segments=6, smooth=True)
    shaft_h = height - 1.25
    collector.add_cylinder("StreetProps", "mat_iron_dark", (x, y, z + 0.75), 0.09, 0.065, shaft_h, segments=6, smooth=True)
    lan_z = z + 0.75 + shaft_h
    collector.add_cylinder("StreetProps", "mat_lantern_glow", (x, y, lan_z), 0.15, 0.19, 0.36, segments=6, smooth=False)
    collector.add_dome("StreetProps", "mat_iron_dark", (x, y, lan_z + 0.36), radius=0.22, height=0.18, rings=2, segments=6, smooth=False)
    collector.bump_count("street_lamps")


def build_wall_lantern(
    collector: MeshCollector,
    anchor_xyz: Vec3,
    tangent_xy: Vec2,
    normal_xy: Vec2,
) -> None:
    ax, ay, az = anchor_xyz
    nx, ny = normal_xy
    lx = ax + nx * 0.36
    ly = ay + ny * 0.36

    collector.add_branch_tube("StreetProps", "mat_iron_dark", (ax, ay, az), (lx, ly, az + 0.12), 0.022, 0.018, segments=4, cap_end=False)
    collector.add_cylinder("StreetProps", "mat_lantern_glow", (lx, ly, az - 0.18), 0.11, 0.14, 0.26, segments=5, smooth=False)
    collector.add_dome("StreetProps", "mat_iron_dark", (lx, ly, az + 0.08), radius=0.16, height=0.12, rings=2, segments=5, smooth=False)
    collector.bump_count("wall_lanterns")


def build_bollard(
    collector: MeshCollector,
    x: float,
    y: float,
    z: float,
    style: str = "stone",
) -> None:
    if style == "iron":
        collector.add_cylinder("StreetProps", "mat_iron_dark", (x, y, z - 0.05), 0.13, 0.10, 0.72, segments=6, smooth=True)
        collector.add_dome("StreetProps", "mat_iron_dark", (x, y, z + 0.67), radius=0.11, height=0.11, rings=2, segments=6)
    else:
        collector.add_cylinder("StreetProps", "mat_stone_ashlar_aged", (x, y, z - 0.08), 0.22, 0.20, 0.66, segments=7, smooth=True)
        collector.add_dome("StreetProps", "mat_stone_ashlar_aged", (x, y, z + 0.58), radius=0.20, height=0.18, rings=2, segments=7)
    collector.bump_count("bollards")


def build_iron_railing(
    collector: MeshCollector,
    p0_xyz: Vec3,
    p1_xyz: Vec3,
    height: float = 1.02,
    picket_spacing: float = 0.28,
) -> None:
    v0 = Vector(p0_xyz)
    v1 = Vector(p1_xyz)
    seg = v1 - v0
    dist_xy = math.hypot(seg.x, seg.y)
    if dist_xy < 0.25:
        return

    for r_z, r_h in ((height - 0.04, 0.045), (0.14, 0.035)):
        collector.add_branch_tube(
            "StreetProps",
            "mat_iron_dark",
            (v0.x, v0.y, v0.z + r_z),
            (v1.x, v1.y, v1.z + r_z),
            r_h * 0.5,
            r_h * 0.5,
            segments=4,
            smooth=False,
            cap_end=False,
        )

    num_pickets = max(2, int(dist_xy / max(0.24, picket_spacing)))
    for i in range(num_pickets + 1):
        t = i / num_pickets
        px = v0.x + seg.x * t
        py = v0.y + seg.y * t
        pz = v0.z + seg.z * t
        is_post = (i == 0 or i == num_pickets or i % 7 == 0)
        rad = 0.026 if is_post else 0.013
        collector.add_cylinder(
            "StreetProps",
            "mat_iron_dark",
            (px, py, pz),
            rad,
            rad,
            height + (0.05 if is_post else -0.02),
            segments=4,
            smooth=False,
            cap_top=is_post,
        )


def build_souq_display(
    collector: MeshCollector,
    center_xyz: Vec3,
    tangent_xy: Vec2,
    normal_xy: Vec2,
    width: float = 2.0,
    seed: int = 0,
) -> None:
    """
    Builds an authentic Jerusalem souq merchant display stall (`0.78m` counter height)
    with a timber counter base, woven kilim runner, 3 compartmentalized wooden spice/produce
    trays topped with colorful spice/fruit mounds (`mat_terracotta`, `mat_dome_gold`,
    `mat_awning_green`), and a terracotta jar beside the stall.
    """
    cx, cy, cz = center_xyz
    tx, ty = tangent_xy
    nx, ny = normal_xy
    rot = math.atan2(ty, tx)

    table_w = min(1.28, max(0.82, width * 0.62))
    sx = cx + nx * 0.48
    sy = cy + ny * 0.48

    # 1. Waist-high wooden merchant counter base (`0.74m` tall)
    collector.add_box(
        "StreetProps",
        "mat_wood_doors",
        (sx, sy, cz + 0.37),
        (table_w, 0.54, 0.74),
        rot_z=rot,
        uv_rect=(0.04, 0.04, 0.46, 0.46),
    )
    # 2. Woven kilim / fabric counter runner (`mat_awning_striped` or `mat_awning_red`)
    runner_mat = "mat_awning_striped" if seed % 2 == 0 else "mat_awning_red"
    collector.add_box(
        "StreetProps",
        runner_mat,
        (sx, sy, cz + 0.77),
        (table_w + 0.06, 0.58, 0.07),
        rot_z=rot,
        uv_scale=0.85,
    )
    # 3. Three wooden produce/spice trays with colorful heaped spice/produce domes on top
    tray_mats = ("mat_terracotta", "mat_dome_gold", "mat_awning_green")
    for k in range(3):
        off_t = (k - 1.0) * (table_w * 0.31)
        kx = sx + tx * off_t
        ky = sy + ty * off_t
        collector.add_box(
            "StreetProps",
            "mat_cardboard",
            (kx, ky, cz + 0.86),
            (0.28, 0.36, 0.12),
            rot_z=rot + (0.04 if k == 0 else (-0.04 if k == 2 else 0.0)),
            uv_scale=0.9,
        )
        collector.add_dome(
            "StreetProps",
            tray_mats[(seed + k) % 3],
            (kx, ky, cz + 0.91),
            radius=0.13,
            height=0.14,
            rings=3,
            segments=6,
            smooth=True,
        )

    collector.bump_count("souq_displays")


def build_rooftop_clutter(
    collector: MeshCollector,
    cx: float,
    cy: float,
    roof_z: float,
    span_x: float,
    span_y: float,
    seed: int = 0,
) -> None:
    rng = random.Random(seed + int(cx * 13 + cy * 19))
    ox = rng.uniform(-0.35, 0.35)
    oy = rng.uniform(-0.35, 0.35)
    tx = cx + ox
    ty = cy + oy

    # Grounded steel base frame + cylindrical solar boiler tank sitting directly on roof_z
    collector.add_box("StreetProps", "mat_iron_dark", (tx, ty - 0.35, roof_z + 0.16), (0.95, 1.15, 0.32))
    collector.add_cylinder("StreetProps", "mat_plaster", (tx, ty, roof_z + 0.25), 0.28, 0.28, 1.05, segments=6, smooth=True)
    collector.add_dome("StreetProps", "mat_plaster", (tx, ty, roof_z + 1.30), radius=0.28, height=0.12, rings=2, segments=6)

    p_bot0 = (tx - 0.48, ty - 0.95, roof_z + 0.06)
    p_bot1 = (tx + 0.48, ty - 0.95, roof_z + 0.06)
    p_top1 = (tx + 0.48, ty - 0.15, roof_z + 0.96)
    p_top0 = (tx - 0.48, ty - 0.15, roof_z + 0.96)
    collector.add_quad("StreetProps", "mat_trash_blue", p_bot0, p_bot1, p_top1, p_top0, uv_scale=0.5, double_sided=True)

    ac_x = cx - ox * 0.5 + 0.65
    ac_y = cy - oy * 0.5
    collector.add_box("StreetProps", "mat_plaster", (ac_x, ac_y, roof_z + 0.28), (0.76, 0.34, 0.56), rot_z=0.2)
    collector.bump_count("rooftop_units")
