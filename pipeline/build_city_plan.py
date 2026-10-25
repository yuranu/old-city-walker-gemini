#!/usr/bin/env python3
"""
Generates a crisp, lightweight vector Streets & Quarters Plan (`public/data/city_plan.json`)
for the interactive 2D Minimap (`src/minimap.js`), replacing the old scene-mesh AABB blobs
with an authentic cartographic Old City of Jerusalem street plan.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

Vec2 = Tuple[float, float]

OLD_CITY_WALL_POLYGON: List[List[float]] = [
    [-22.0, 5.0],
    [-188.0, 195.0],
    [-175.0, 292.0],
    [-135.0, 305.0],
    [-75.0, 335.0],
    [-20.0, 372.0],
    [40.0, 372.0],
    [175.0, 505.0],
    [252.0, 565.0],
    [310.0, 622.0],
    [430.0, 652.0],
    [575.0, 706.0],
    [708.0, 750.0],
    [820.0, 778.0],
    [865.0, 692.0],
    [876.0, 560.0],
    [888.0, 408.0],
    [904.0, 260.0],
    [935.0, 58.0],
    [768.0, -86.0],
    [746.0, -178.0],
    [584.0, -204.0],
    [526.0, -258.0],
    [441.0, -360.0],
    [286.0, -410.0],
    [180.0, -416.0],
    [5.0, -390.0],
    [6.0, -115.0],
    [-10.0, -15.0],
]


def point_in_polygon_2d(pt: Vec2, poly: Sequence[Sequence[float]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0):
            inside = not inside
    return inside


def is_near_old_city(x: float, y: float, margin: float = 40.0) -> bool:
    if point_in_polygon_2d((x, y), OLD_CITY_WALL_POLYGON):
        return True
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


def regular_polygon(cx: float, cy: float, radius: float, sides: int, rot_deg: float = 0.0) -> List[List[float]]:
    pts = []
    for i in range(sides):
        ang = math.radians(rot_deg + (360.0 * i) / sides)
        pts.append([round(cx + radius * math.cos(ang), 1), round(cy + radius * math.sin(ang), 1)])
    return pts


def build_city_plan() -> dict:
    root = Path(__file__).resolve().parent.parent
    osm_path = root / "data" / "osm_old_city.json"
    with open(osm_path, "r", encoding="utf-8") as f:
        osm_data = json.load(f)

    # 1. Named Primary & Secondary Thoroughfares of the Old City (continuous, well-labeled)
    curated_streets = [
        {
            "name": "Jaffa Rd / West Approach",
            "tier": "primary",
            "pts": [[-95.0, 14.0], [-42.0, 6.0], [-16.0, 2.5], [0.0, 2.0], [22.0, -2.0], [75.0, 2.0]],
        },
        {
            "name": "David St (Suq el-Bazar)",
            "tier": "primary",
            "pts": [[75.0, 2.0], [155.0, 2.0], [235.0, 8.0], [313.0, 14.0]],
        },
        {
            "name": "Street of the Chain",
            "tier": "primary",
            "pts": [[313.0, 14.0], [385.0, 24.0], [455.0, 35.0], [555.0, 35.0], [662.0, 35.0]],
        },
        {
            "name": "Khan el-Zeit",
            "tier": "primary",
            "pts": [[254.0, 545.0], [275.0, 440.0], [292.0, 330.0], [306.0, 210.0], [313.0, 108.0], [313.0, 14.0]],
        },
        {
            "name": "Triple Souq / Cardo",
            "tier": "primary",
            "pts": [[313.0, 108.0], [313.0, 14.0], [308.0, -75.0], [308.0, -185.0], [308.0, -265.0]],
        },
        {
            "name": "El-Wad (HaGai St)",
            "tier": "primary",
            "pts": [[262.0, 540.0], [345.0, 465.0], [430.0, 412.0], [468.0, 310.0], [512.0, 195.0], [555.0, 85.0], [595.0, 12.0]],
        },
        {
            "name": "Via Dolorosa",
            "tier": "primary",
            "pts": [[875.0, 467.0], [812.0, 528.0], [685.0, 485.0], [565.0, 428.0], [478.0, 413.0], [430.0, 412.0], [395.0, 335.0], [296.0, 255.0], [235.0, 185.0], [197.0, 158.0]],
        },
        {
            "name": "Christian Quarter Rd",
            "tier": "secondary",
            "pts": [[75.0, 12.0], [92.0, 95.0], [112.0, 175.0], [132.0, 255.0], [148.0, 315.0]],
        },
        {
            "name": "Latin Patriarchate Rd",
            "tier": "secondary",
            "pts": [[24.0, 6.0], [24.0, 65.0], [8.0, 145.0], [-45.0, 215.0], [-138.0, 294.0]],
        },
        {
            "name": "Greek Patriarchate St",
            "tier": "secondary",
            "pts": [[8.0, 145.0], [65.0, 155.0], [112.0, 175.0], [168.0, 175.0]],
        },
        {
            "name": "St. Francis St",
            "tier": "secondary",
            "pts": [[-138.0, 294.0], [-55.0, 285.0], [35.0, 285.0], [132.0, 255.0]],
        },
        {
            "name": "Muristan (Suq Aftimos)",
            "tier": "secondary",
            "pts": [[168.0, 124.0], [222.0, 124.0], [268.0, 124.0], [306.0, 124.0]],
        },
        {
            "name": "St. Helena Rd",
            "tier": "secondary",
            "pts": [[197.0, 124.0], [197.0, 158.0], [235.0, 185.0]],
        },
        {
            "name": "Armenian Patriarchate Rd",
            "tier": "primary",
            "pts": [[58.0, -4.0], [56.0, -95.0], [54.0, -210.0], [58.0, -325.0], [125.0, -385.0], [186.0, -410.0]],
        },
        {
            "name": "Moat Road (1898 Breach)",
            "tier": "secondary",
            "pts": [[-16.0, -6.0], [8.0, -12.0], [10.0, -85.0], [10.0, -210.0]],
        },
        {
            "name": "St. James St",
            "tier": "secondary",
            "pts": [[54.0, -210.0], [132.0, -242.0], [215.0, -235.0], [308.0, -225.0]],
        },
        {
            "name": "Ararat St",
            "tier": "secondary",
            "pts": [[185.0, -85.0], [185.0, -175.0], [185.0, -285.0], [186.0, -410.0]],
        },
        {
            "name": "Jewish Quarter Rd",
            "tier": "secondary",
            "pts": [[346.0, -45.0], [346.0, -110.0], [348.0, -195.0], [355.0, -285.0], [315.0, -355.0]],
        },
        {
            "name": "Tiferet Yisrael St",
            "tier": "secondary",
            "pts": [[308.0, -145.0], [365.0, -145.0], [435.0, -155.0], [495.0, -155.0], [565.0, -95.0]],
        },
        {
            "name": "Misgav Ladach St",
            "tier": "secondary",
            "pts": [[445.0, -75.0], [455.0, -155.0], [448.0, -245.0], [435.0, -310.0]],
        },
        {
            "name": "Batei Mahaseh & Dung Gate Rd",
            "tier": "primary",
            "pts": [[435.0, -310.0], [525.0, -240.0], [612.0, -200.0], [615.0, -95.0], [615.0, -25.0]],
        },
        {
            "name": "Suq al-Qattanin (Cotton Market)",
            "tier": "secondary",
            "pts": [[512.0, 145.0], [585.0, 145.0], [668.0, 145.0]],
        },
        {
            "name": "Aqabat al-Khalidiya",
            "tier": "secondary",
            "pts": [[306.0, 185.0], [410.0, 185.0], [512.0, 195.0]],
        },
        {
            "name": "Aqabat al-Saraya",
            "tier": "secondary",
            "pts": [[295.0, 295.0], [390.0, 295.0], [468.0, 310.0]],
        },
        {
            "name": "Tariq Bab al-Nazir",
            "tier": "secondary",
            "pts": [[485.0, 265.0], [585.0, 265.0], [668.0, 265.0]],
        },
        {
            "name": "Al-Qadisiya / Herod's Gate Rd",
            "tier": "secondary",
            "pts": [[568.0, 702.0], [555.0, 585.0], [545.0, 485.0], [565.0, 428.0]],
        },
        {
            "name": "Bab Hutta Rd",
            "tier": "secondary",
            "pts": [[568.0, 650.0], [685.0, 625.0], [745.0, 545.0], [745.0, 365.0]],
        },
    ]

    # 2. Extract OSM streets & alleys inside/around the Old City Walls
    osm_streets = []
    for s in osm_data.get("streets", []):
        pts3d = s.get("pts", [])
        if len(pts3d) < 2:
            continue
        pts2d = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts3d]
        mid = pts2d[len(pts2d) // 2]
        if not is_near_old_city(mid[0], mid[1], margin=55.0):
            continue
        hw = s.get("highway", "footway")
        name = (s.get("name") or "").strip()
        if hw in ("primary", "secondary", "tertiary"):
            tier = "secondary"
        elif hw == "steps" or s.get("is_steps"):
            tier = "steps"
        else:
            tier = "alley"
        osm_streets.append(
            {
                "name": name,
                "tier": tier,
                "pts": pts2d,
            }
        )

    # 3. 8 Gates
    gates = [
        {"id": "jaffa_gate", "name": "Jaffa Gate", "x": -6.0, "y": 3.0},
        {"id": "new_gate", "name": "New Gate", "x": -140.0, "y": 298.0},
        {"id": "damascus_gate", "name": "Damascus Gate", "x": 254.0, "y": 557.0},
        {"id": "herods_gate", "name": "Herod's Gate", "x": 568.0, "y": 702.0},
        {"id": "lions_gate", "name": "Lions' Gate", "x": 876.0, "y": 467.0},
        {"id": "golden_gate", "name": "Golden Gate", "x": 888.0, "y": 260.0},
        {"id": "dung_gate", "name": "Dung Gate", "x": 614.0, "y": -201.0},
        {"id": "zion_gate", "name": "Zion Gate", "x": 186.0, "y": -413.0},
    ]

    # 4. 5 Historic Quarters
    quarters = [
        {"id": "christian", "name": "CHRISTIAN QUARTER", "x": 85.0, "y": 235.0, "color": "#60a5fa"},
        {"id": "muslim", "name": "MUSLIM QUARTER", "x": 520.0, "y": 490.0, "color": "#34d399"},
        {"id": "armenian", "name": "ARMENIAN QUARTER", "x": 135.0, "y": -225.0, "color": "#f59e0b"},
        {"id": "jewish", "name": "JEWISH QUARTER", "x": 415.0, "y": -175.0, "color": "#a78bfa"},
        {"id": "temple_mount", "name": "TEMPLE MOUNT / HARAM", "x": 778.0, "y": 225.0, "color": "#fbbf24"},
    ]

    # 5. Clean Monument/Landmark Outlines (only key architectural monuments, zero blob clutter!)
    landmarks = [
        {
            "id": "citadel",
            "name": "Tower of David & Citadel",
            "poly": [[12.0, -14.0], [62.0, -14.0], [62.0, -82.0], [12.0, -82.0]],
        },
        {
            "id": "holy_sepulchre",
            "name": "Holy Sepulchre",
            "poly": [[162.0, 162.0], [234.0, 162.0], [234.0, 218.0], [162.0, 218.0]],
        },
        {
            "id": "redeemer",
            "name": "Church of the Redeemer",
            "poly": [[232.0, 118.0], [268.0, 118.0], [268.0, 152.0], [232.0, 152.0]],
        },
        {
            "id": "st_james",
            "name": "St. James Cathedral",
            "poly": [[110.0, -242.0], [156.0, -242.0], [156.0, -202.0], [110.0, -202.0]],
        },
        {
            "id": "hurva",
            "name": "Hurva Synagogue",
            "poly": [[330.0, -146.0], [364.0, -146.0], [364.0, -114.0], [330.0, -114.0]],
        },
        {
            "id": "sephardic_synagogues",
            "name": "Four Sephardic Synagogues",
            "poly": [[354.0, -228.0], [386.0, -228.0], [386.0, -198.0], [354.0, -198.0]],
        },
        {
            "id": "tiferet_yisrael",
            "name": "Tiferet Yisrael",
            "poly": [[468.0, -168.0], [496.0, -168.0], [496.0, -142.0], [468.0, -142.0]],
        },
        {
            "id": "western_wall",
            "name": "Western Wall Plaza",
            "poly": [[595.0, -45.0], [668.0, -45.0], [668.0, 48.0], [595.0, 48.0]],
        },
        {
            "id": "dome_of_the_rock",
            "name": "Dome of the Rock",
            "poly": regular_polygon(740.0, 158.0, 27.5, 8, 22.5),
        },
        {
            "id": "dome_of_the_chain",
            "name": "Dome of the Chain",
            "poly": regular_polygon(774.0, 158.0, 7.5, 8, 0.0),
        },
        {
            "id": "al_aqsa",
            "name": "Al-Aqsa Mosque",
            "poly": [[734.0, -84.0], [790.0, -84.0], [790.0, -4.0], [734.0, -4.0]],
        },
        {
            "id": "st_anne",
            "name": "Church of St. Anne",
            "poly": [[788.0, 482.0], [826.0, 482.0], [826.0, 514.0], [788.0, 514.0]],
        },
        {
            "id": "austrian_hospice",
            "name": "Austrian Hospice",
            "poly": [[416.0, 405.0], [452.0, 405.0], [452.0, 438.0], [416.0, 438.0]],
        },
    ]

    return {
        "cityWall": OLD_CITY_WALL_POLYGON,
        "templeMountWall": [[668.0, 365.0], [885.0, 365.0], [895.0, -185.0], [668.0, -185.0]],
        "domePlatform": [[698.0, 224.0], [782.0, 224.0], [782.0, 96.0], [698.0, 96.0]],
        "quarters": quarters,
        "gates": gates,
        "landmarks": landmarks,
        "curatedStreets": curated_streets,
        "osmStreets": osm_streets,
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    plan = build_city_plan()
    out_paths = [
        root / "public" / "data" / "city_plan.json",
        root / "data" / "city_plan.json",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(plan, f, separators=(",", ":"), ensure_ascii=False)
        print(f"[OK] Wrote {p} ({p.stat().st_size:,} bytes)")
    print(
        f"[PLAN] Curated streets: {len(plan['curatedStreets'])}, "
        f"OSM streets: {len(plan['osmStreets'])}, "
        f"Landmarks: {len(plan['landmarks'])}, Gates: {len(plan['gates'])}"
    )


if __name__ == "__main__":
    main()
