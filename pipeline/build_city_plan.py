#!/usr/bin/env python3
"""
Generates a crisp, lightweight vector Streets & Quarters Plan (`public/data/city_plan.json`)
for the interactive 2D Minimap (`src/minimap.js`), using exact OpenStreetMap geometry for
the Ottoman Old City Walls, the Temple Mount / Haram al-Sharif retaining walls, the Upper
Dome of the Rock platform, and all named historic thoroughfares.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Vec2 = Tuple[float, float]

REF_LAT = 31.7767
REF_LON = 35.2278
M_PER_LAT = 110880.0
M_PER_LON = 111320.0 * math.cos(math.radians(REF_LAT))

# Exact fallback Ottoman City Wall loop (overridden dynamically when osm_old_city_raw.json is present)
OLD_CITY_WALL_POLYGON: List[List[float]] = [
    [-3.4, -2.7], [-7.5, -1.8], [-4.8, 7.6], [-21.8, 3.6], [-186.1, 198.1],
    [-174.1, 290.8], [-152.9, 304.3], [-125.9, 297.2], [-77.4, 336.1],
    [-21.2, 372.4], [38.0, 369.5], [169.4, 502.3], [229.9, 560.4], [259.6, 557.8],
    [308.4, 622.6], [428.0, 650.2], [568.5, 706.5], [705.2, 749.3], [817.4, 777.9],
    [864.5, 691.5], [873.5, 564.1], [879.9, 468.5], [891.5, 344.2], [891.6, 257.1],
    [905.3, 152.6], [931.7, 56.8], [766.9, -86.6], [744.7, -177.9], [666.0, -189.2],
    [619.6, -201.4], [583.0, -203.7], [527.1, -257.5], [440.1, -359.7],
    [285.6, -408.7], [186.8, -414.4], [5.8, -389.0], [6.4, -11.3],
]


def _ll_to_xy(lat: float, lon: float) -> List[float]:
    return [
        round((lon - REF_LON) * M_PER_LON, 1),
        round((lat - REF_LAT) * M_PER_LAT, 1),
    ]


def _extract_exact_walls_from_raw(raw_path: Path) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    """Extract exact Ottoman City Wall, Temple Mount Wall, Dome Platform, and Western Wall Plaza from osm_old_city_raw.json."""
    default_ww = [[560.5, -44.0], [593.2, -56.2], [594.5, -32.0], [636.5, -13.3], [642.7, -21.0], [632.5, 37.1], [549.7, 13.2]]
    if not raw_path.exists():
        return (
            OLD_CITY_WALL_POLYGON,
            [[887.3, 407.5], [569.5, 378.1], [614.7, 159.4], [627.0, 67.6], [657.0, -104.1], [768.1, -84.8], [934.6, -58.4], [891.6, 257.1]],
            [[685.6, 266.3], [772.9, 250.3], [804.2, 111.7], [697.8, 92.2], [646.7, 224.2]],
            default_ww,
        )

    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    walls = raw.get("walls", [])
    # 1. Exact Temple Mount / Haram al-Sharif Retaining Wall
    # walls[0]: North & NW wall ([887.3, 407.5] -> [569.5, 378.1] -> [627.0, 67.6])
    # reversed(walls[6]): SW Western Wall & Southern Wall ([627.0, 67.6] -> [657.0, -104.1] -> [768.1, -84.8])
    # Plus SE corner ([934.6, -58.4]) & Eastern Wall back to [887.3, 407.5]
    tm_wall: List[List[float]] = []
    if len(walls) > 6:
        w0 = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in walls[0].get("pts", [])]
        w6_rev = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in reversed(walls[6].get("pts", []))]
        east_pts = [[934.6, -58.4], [905.3, 152.6], [891.6, 257.1], [891.5, 344.2]]
        for seg in (w0, w6_rev, east_pts):
            for p in seg:
                if not tm_wall or math.hypot(p[0] - tm_wall[-1][0], p[1] - tm_wall[-1][1]) > 0.5:
                    tm_wall.append(p)

    # 2. Exact Upper Dome of the Rock Platform (Wall of the Rock Square, walls[5])
    dome_platform: List[List[float]] = []
    for w in walls:
        if w.get("name") == "Wall of the Rock Square":
            dome_platform = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in w.get("pts", [])]
            if len(dome_platform) > 2 and dome_platform[0] == dome_platform[-1]:
                dome_platform = dome_platform[:-1]
            break

    # 3. Exact Western Wall Plaza polygon from streets
    ww_plaza: List[List[float]] = default_ww
    for s in raw.get("streets", []):
        if s.get("name") == "Western Wall Plaza" and len(s.get("pts", [])) >= 10:
            ww_plaza = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in s["pts"][:-1]]
            break

    return (
        OLD_CITY_WALL_POLYGON,
        tm_wall,
        dome_platform,
        ww_plaza,
    )


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


def is_near_old_city(x: float, y: float, wall_poly: Sequence[Sequence[float]], margin: float = 40.0) -> bool:
    if point_in_polygon_2d((x, y), wall_poly):
        return True
    n = len(wall_poly)
    for i in range(n):
        x0, y0 = wall_poly[i]
        x1, y1 = wall_poly[(i + 1) % n]
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
    raw_path = root / "data" / "osm_old_city_raw.json"

    with open(osm_path, "r", encoding="utf-8") as f:
        osm_data = json.load(f)

    city_wall, tm_wall, dome_platform, ww_plaza = _extract_exact_walls_from_raw(raw_path)
    global OLD_CITY_WALL_POLYGON
    OLD_CITY_WALL_POLYGON = city_wall

    # 1. Exact OpenStreetMap Named Primary & Secondary Thoroughfares of the Old City
    curated_streets = [
        {
            "name": "Jaffa Gate & Omar ibn al-Khattab Sq",
            "tier": "primary",
            "pts": [
                [-95.0, 14.0], [-42.0, 6.0], [-16.0, 2.5], [-6.2, 2.7],
                [8.1, -7.7], [34.8, 1.8], [62.5, 10.6], [83.7, 4.8],
            ],
        },
        {
            "name": "David St (Suq el-Bazar)",
            "tier": "primary",
            "pts": [
                [83.7, 4.8], [93.4, 11.1], [122.8, 10.6], [145.9, 10.1],
                [162.5, 11.4], [179.8, 15.0], [198.7, 15.0], [230.6, 15.7],
                [255.4, 18.0], [285.8, 19.2], [302.4, 20.4], [313.6, 20.2], [329.2, 19.4],
            ],
        },
        {
            "name": "Street of the Chain (Bab al-Silsila)",
            "tier": "primary",
            "pts": [
                [329.2, 19.4], [331.8, 11.1], [352.1, 12.0], [379.8, 14.1],
                [404.2, 18.6], [440.9, 24.2], [472.6, 29.2], [505.5, 33.9],
                [521.3, 36.7], [552.6, 46.4], [587.2, 56.7], [616.5, 65.4], [626.2, 73.2],
            ],
        },
        {
            "name": "Khan el-Zeit (Suq Khan al-Zeit)",
            "tier": "primary",
            "pts": [
                [254.8, 555.9], [263.7, 550.5], [271.4, 545.2], [284.4, 527.4],
                [290.7, 513.0], [295.1, 496.6], [290.5, 493.1], [290.6, 470.3],
                [292.7, 444.4], [294.2, 403.5], [296.2, 398.5], [298.1, 349.2],
                [299.5, 329.4], [300.5, 300.8], [304.0, 293.2], [305.1, 264.9],
                [306.7, 237.1], [306.8, 215.2], [307.7, 193.8], [307.9, 172.5],
            ],
        },
        {
            "name": "Triple Souq & Cardo (HaKardo)",
            "tier": "primary",
            "pts": [
                [307.9, 172.5], [313.8, 138.7], [319.1, 20.2], [320.2, 3.0],
                [315.4, -46.5], [312.7, -80.0], [310.7, -113.3], [307.7, -170.2], [306.0, -193.5],
            ],
        },
        {
            "name": "Al-Wad (HaGai St)",
            "tier": "primary",
            "pts": [
                [295.1, 496.6], [298.5, 496.8], [308.5, 489.4], [328.9, 463.3],
                [350.3, 435.8], [359.0, 425.3], [365.4, 425.4], [380.8, 412.0],
                [388.1, 404.6], [412.0, 374.7], [431.3, 347.7], [444.4, 330.0],
                [450.0, 325.4], [460.3, 315.8], [475.1, 305.2], [486.3, 286.1],
                [498.3, 269.1], [502.5, 247.1], [509.6, 218.3], [511.9, 196.8],
                [513.3, 176.9], [520.4, 142.6], [522.8, 129.6], [525.4, 109.1],
                [526.8, 83.3], [537.2, 66.3], [541.5, 35.8],
            ],
        },
        {
            "name": "Via Dolorosa",
            "tier": "primary",
            "pts": [
                [879.9, 468.5], [868.6, 466.8], [830.7, 459.6], [792.8, 453.8],
                [746.3, 444.0], [701.4, 436.9], [674.9, 431.2], [665.3, 429.5],
                [626.7, 421.6], [592.3, 414.4], [556.6, 407.2], [536.2, 403.0],
                [509.1, 397.1], [481.1, 392.1], [458.6, 387.8], [432.1, 382.6],
                [412.0, 374.7], [431.3, 347.7], [450.0, 325.4], [391.3, 311.3],
                [342.4, 301.4], [304.0, 293.2],
            ],
        },
        {
            "name": "Christian Quarter St",
            "tier": "secondary",
            "pts": [
                [162.5, 11.4], [161.5, 21.2], [160.0, 56.0], [157.4, 101.2],
                [154.7, 134.5], [153.7, 150.0], [152.2, 182.9], [147.2, 222.0], [146.3, 262.7],
            ],
        },
        {
            "name": "Latin Patriarchate & New Gate Rd",
            "tier": "secondary",
            "pts": [
                [34.8, 1.8], [23.3, 16.8], [-0.7, 46.1], [-32.6, 71.9],
                [-61.4, 96.1], [-40.6, 121.4], [-60.8, 141.0], [-51.1, 148.6],
                [6.3, 168.4], [-10.5, 195.8], [-33.3, 220.1], [-56.3, 231.1],
                [-95.6, 208.0], [-125.3, 247.8], [-135.3, 273.8], [-139.7, 294.8],
            ],
        },
        {
            "name": "Greek Patriarchate St",
            "tier": "secondary",
            "pts": [[6.3, 168.4], [89.5, 200.9], [114.4, 210.6], [147.2, 222.0]],
        },
        {
            "name": "St. Francis St & Aqabat el-Khanqa",
            "tier": "secondary",
            "pts": [
                [-33.3, 220.1], [-24.9, 235.1], [-0.9, 239.9], [33.2, 246.9],
                [95.0, 257.4], [146.3, 262.7], [193.7, 270.8], [253.3, 281.1], [304.0, 293.2],
            ],
        },
        {
            "name": "Muristan (Suq Aftimos)",
            "tier": "secondary",
            "pts": [
                [153.7, 150.0], [180.5, 155.2], [211.4, 157.9],
                [250.7, 154.8], [298.6, 155.4], [307.9, 172.5],
            ],
        },
        {
            "name": "Muristan St",
            "tier": "secondary",
            "pts": [[250.7, 154.8], [251.2, 120.5], [251.6, 99.8], [254.1, 60.0], [255.4, 18.0]],
        },
        {
            "name": "Armenian Patriarchate Rd",
            "tier": "primary",
            "pts": [
                [83.7, 4.8], [93.8, -8.2], [100.3, -33.5], [97.6, -54.9],
                [95.1, -76.2], [100.1, -81.1], [106.3, -104.7], [106.4, -130.4],
                [100.3, -180.3], [94.0, -220.1], [92.7, -256.3], [94.0, -290.1],
                [95.3, -342.4], [96.1, -382.0], [104.3, -401.2], [159.1, -404.7],
                [189.1, -399.7], [207.5, -406.5],
            ],
        },
        {
            "name": "Citadel Moat & Maronite Convent St",
            "tier": "secondary",
            "pts": [[-6.2, 2.7], [8.1, -7.7], [10.8, -46.8], [15.5, -80.2], [12.3, -139.2], [16.5, -173.4]],
        },
        {
            "name": "St. James & Or HaHayim St",
            "tier": "secondary",
            "pts": [
                [100.3, -180.3], [141.4, -179.9], [143.2, -163.4], [175.8, -161.5],
                [204.6, -149.9], [233.3, -148.8], [262.8, -152.7], [295.4, -165.8],
            ],
        },
        {
            "name": "Ararat St",
            "tier": "secondary",
            "pts": [
                [264.6, -57.9], [239.8, -77.0], [220.5, -104.8], [204.6, -149.9],
                [205.6, -186.1], [203.2, -223.7], [197.9, -260.3], [219.0, -276.0],
                [209.1, -305.9], [211.0, -338.1], [211.0, -379.4], [213.0, -407.5],
            ],
        },
        {
            "name": "Chabad St",
            "tier": "secondary",
            "pts": [
                [302.4, 20.4], [303.0, -41.9], [299.6, -98.5], [295.4, -165.8],
                [291.2, -230.9], [290.1, -299.0], [285.7, -320.7],
            ],
        },
        {
            "name": "Jewish Quarter St",
            "tier": "secondary",
            "pts": [
                [329.2, 19.4], [331.8, 11.1], [329.2, -43.8], [322.6, -91.0],
                [322.7, -124.7], [314.9, -167.1], [311.3, -216.1], [313.8, -245.0], [316.4, -264.7],
            ],
        },
        {
            "name": "Tiferet Israel St",
            "tier": "secondary",
            "pts": [
                [322.7, -124.7], [377.1, -115.5], [400.2, -122.1], [427.7, -124.5],
                [447.3, -122.5], [465.3, -136.2], [490.1, -132.1],
            ],
        },
        {
            "name": "Misgav Ladach St",
            "tier": "secondary",
            "pts": [
                [472.6, 29.2], [482.5, -9.2], [484.3, -60.2], [483.6, -87.0],
                [490.1, -132.1], [487.5, -165.1], [483.1, -198.9], [478.4, -235.5], [491.6, -252.5],
            ],
        },
        {
            "name": "Batei Mahase & Dung Gate Rd",
            "tier": "primary",
            "pts": [
                [296.0, -320.5], [349.1, -314.7], [399.3, -312.1], [441.2, -293.6],
                [504.8, -262.4], [533.7, -237.1], [562.2, -188.1], [568.0, -150.2],
                [574.5, -139.3], [597.0, -148.0], [607.4, -177.7], [614.5, -197.2], [621.2, -218.3],
            ],
        },
        {
            "name": "Suq al-Qattanin (Cotton Market)",
            "tier": "secondary",
            "pts": [[520.4, 142.6], [537.0, 145.2], [567.7, 150.7], [586.3, 152.6], [613.5, 151.9]],
        },
        {
            "name": "Aqabat al-Khalidiya",
            "tier": "secondary",
            "pts": [[307.9, 172.5], [376.9, 176.1], [412.9, 175.5], [468.3, 181.7], [482.8, 121.9], [522.8, 129.6]],
        },
        {
            "name": "Aqabat el-Saraya (Al-malik Faisal)",
            "tier": "secondary",
            "pts": [[306.7, 237.1], [390.3, 247.3], [411.7, 251.0], [498.3, 269.1], [590.2, 277.9]],
        },
        {
            "name": "Bab el-Hadid St",
            "tier": "secondary",
            "pts": [[511.9, 196.8], [556.7, 189.3], [605.1, 197.9]],
        },
        {
            "name": "Al-Qadisiya (Herod's Gate Rd)",
            "tier": "secondary",
            "pts": [
                [569.3, 697.4], [595.0, 696.0], [615.1, 683.4], [626.1, 653.6],
                [637.3, 582.7], [660.9, 448.8], [665.3, 429.5],
            ],
        },
        {
            "name": "Bab Hutta Rd",
            "tier": "secondary",
            "pts": [[708.5, 628.3], [713.3, 596.7], [723.7, 560.3], [729.1, 529.3], [743.7, 460.3], [752.1, 399.8]],
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
        if not is_near_old_city(mid[0], mid[1], city_wall, margin=55.0):
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

    # 3. 8 Gates (exact OSM gate node coordinates)
    gates = [
        {"id": "jaffa_gate", "name": "Jaffa Gate", "x": -6.2, "y": 2.7},
        {"id": "new_gate", "name": "New Gate", "x": -139.7, "y": 296.6},
        {"id": "damascus_gate", "name": "Damascus Gate", "x": 254.2, "y": 556.8},
        {"id": "herods_gate", "name": "Herod's Gate", "x": 569.1, "y": 698.6},
        {"id": "lions_gate", "name": "Lions' Gate", "x": 879.9, "y": 468.5},
        {"id": "golden_gate", "name": "Golden Gate", "x": 891.6, "y": 257.1},
        {"id": "dung_gate", "name": "Dung Gate", "x": 616.7, "y": -204.0},
        {"id": "zion_gate", "name": "Zion Gate", "x": 186.8, "y": -414.4},
    ]

    # 4. 5 Historic Quarters
    quarters = [
        {"id": "christian", "name": "CHRISTIAN QUARTER", "x": 85.0, "y": 235.0, "color": "#60a5fa"},
        {"id": "muslim", "name": "MUSLIM QUARTER", "x": 520.0, "y": 490.0, "color": "#34d399"},
        {"id": "armenian", "name": "ARMENIAN QUARTER", "x": 145.0, "y": -225.0, "color": "#f59e0b"},
        {"id": "jewish", "name": "JEWISH QUARTER", "x": 415.0, "y": -175.0, "color": "#a78bfa"},
        {"id": "temple_mount", "name": "TEMPLE MOUNT / HARAM", "x": 745.0, "y": 175.0, "color": "#fbbf24"},
    ]

    # 5. Clean Monument/Landmark Outlines (aligned to exact OSM polygons)
    landmarks = [
        {
            "id": "citadel",
            "name": "Tower of David & Citadel",
            "poly": [[6.3, -15.9], [77.6, -15.9], [77.6, -86.4], [6.3, -86.4]],
        },
        {
            "id": "holy_sepulchre",
            "name": "Holy Sepulchre",
            "poly": [[160.1, 160.0], [250.3, 160.0], [250.3, 232.1], [160.1, 232.1]],
        },
        {
            "id": "redeemer",
            "name": "Church of the Redeemer",
            "poly": [[250.7, 118.2], [298.6, 118.2], [298.6, 155.4], [250.7, 155.4]],
        },
        {
            "id": "st_james",
            "name": "St. James Cathedral",
            "poly": [[112.1, -242.5], [156.9, -242.5], [156.9, -193.3], [112.1, -193.3]],
        },
        {
            "id": "hurva",
            "name": "Hurva Synagogue",
            "poly": [[335.4, -152.7], [357.1, -152.7], [357.1, -125.1], [335.4, -125.1]],
        },
        {
            "id": "sephardic_synagogues",
            "name": "Four Sephardic Synagogues",
            "poly": [[344.2, -242.3], [389.0, -242.3], [389.0, -197.4], [344.2, -197.4]],
        },
        {
            "id": "tiferet_yisrael",
            "name": "Tiferet Yisrael",
            "poly": [[430.8, -146.8], [447.6, -146.8], [447.6, -125.0], [430.8, -125.0]],
        },
        {
            "id": "western_wall",
            "name": "Western Wall Plaza",
            "poly": ww_plaza,
        },
        {
            "id": "dome_of_the_rock",
            "name": "Dome of the Rock",
            "poly": regular_polygon(720.7, 155.7, 26.8, 8, 22.5),
        },
        {
            "id": "dome_of_the_chain",
            "name": "Dome of the Chain",
            "poly": regular_polygon(762.4, 162.4, 7.5, 8, 0.0),
        },
        {
            "id": "al_aqsa",
            "name": "Al-Aqsa Mosque",
            "poly": [[713.9, -86.6], [790.2, -86.6], [790.2, -2.5], [713.9, -2.5]],
        },
        {
            "id": "st_anne",
            "name": "Church of St. Anne",
            "poly": [[822.4, 519.4], [851.9, 519.4], [851.9, 537.7], [822.4, 537.7]],
        },
        {
            "id": "austrian_hospice",
            "name": "Austrian Hospice",
            "poly": [[396.0, 392.7], [447.2, 392.7], [447.2, 428.3], [396.0, 428.3]],
        },
    ]

    return {
        "cityWall": city_wall,
        "templeMountWall": tm_wall,
        "domePlatform": dome_platform,
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
        f"[PLAN] Wall pts: {len(plan['cityWall'])}, TM wall pts: {len(plan['templeMountWall'])}, "
        f"Curated streets: {len(plan['curatedStreets'])}, OSM streets: {len(plan['osmStreets'])}, "
        f"Landmarks: {len(plan['landmarks'])}, Gates: {len(plan['gates'])}"
    )


if __name__ == "__main__":
    main()
