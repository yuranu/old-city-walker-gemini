#!/usr/bin/env node
/**
 * Automated Headless Browser QA Capture & Self-Critique Defect Detection Harness
 * Uses google-chrome-stable via raw Chrome DevTools WebSocket protocol (zero extra npm dependencies).
 *
 * Captures 30 Viewpoints:
 *  - 10 recognizable reference places from public/data/places.json
 *  - 10 multi-angle walker viewpoints across the Jaffa Gate district
 *  - 10 diagnostic look-up / look-down / close-up inspection viewpoints (diag_21 .. diag_30)
 *
 * For each view, captures:
 *  - <id>_model.png   (normal PBR render)
 *  - <id>_jitter.png  (+1cm camera jitter frame for localized 32x32 block z-fighting detection)
 *  - <id>_skykey.png  (bright magenta below-horizon skykey for ground-hole detection)
 *  - traceView(6, 4)  (scene manifest object/rule/material coverage)
 *
 * Runs automated image-level defect detectors:
 *  - maxBlockFlickerPct (32x32 local block Z-fighting detector with 1px parallax compensation)
 *  - flatPatchPct       (16x16 untextured solid-color polygon detector)
 *  - uvStreakScore      (32x32 directional Sobel 1D UV stripe/streak detector)
 *  - groundHoleFraction (below-horizon magenta sky-leak detector)
 *
 * Outputs:
 *  - captured.json & qa_automated_audit.json
 *  - pair_<id>.jpg (REAL | MODEL side-by-side comparisons)
 *  - contact_sheet_pairs.jpg
 *  - contact_sheet_20views.jpg
 *  - contact_sheet_diagnostics.jpg (10 diagnostic close-up/look-up views with defect metrics)
 */

import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');

// 10 additional multi-angle walker viewpoints across the district
const EXTRA_WALKER_VIEWPOINTS = [
  {
    id: 'walker_11_mamilla_steps_up',
    name: 'Mamilla Approach Promenade Looking East to Walls',
    x: -48.0,
    y: 12.0,
    z_eye: 1.7,
    heading: 102.0,
    pitch: 5.0,
    fov: 70.0
  },
  {
    id: 'walker_12_west_plaza_south_angle',
    name: 'Jaffa Gate West Plaza Looking NE at Gatehouse & Bastion',
    x: -26.0,
    y: -8.0,
    z_eye: 1.7,
    heading: 52.0,
    pitch: 7.0,
    fov: 70.0
  },
  {
    id: 'walker_13_gate_inner_exit',
    name: 'Jaffa Gate Inner Threshold Exiting into Old City',
    x: 6.5,
    y: 1.5,
    z_eye: 1.7,
    heading: 115.0,
    pitch: 3.0,
    fov: 72.0
  },
  {
    id: 'walker_14_citadel_moat_bridge',
    name: 'Citadel Moat Bridge Looking Toward Phasael & Hippicus Towers',
    x: 43.8,
    y: -14.5,
    z_eye: 1.7,
    heading: 205.0,
    pitch: 8.0,
    fov: 70.0
  },
  {
    id: 'walker_15_armenian_road_south',
    name: 'Armenian Patriarchate Road Along Citadel East Glacis',
    x: 68.0,
    y: -18.0,
    z_eye: 1.7,
    heading: 185.0,
    pitch: 4.0,
    fov: 70.0
  },
  {
    id: 'walker_16_armenian_road_north_lookback',
    name: 'Armenian Patriarchate Road Looking North to Omar Square',
    x: 68.0,
    y: -34.0,
    z_eye: 1.7,
    heading: 355.0,
    pitch: 5.0,
    fov: 70.0
  },
  {
    id: 'walker_17_imperial_hotel_arcade',
    name: 'Grand New Imperial Hotel Ground-Floor Shopfront Arcade',
    x: 34.0,
    y: -6.0,
    z_eye: 1.7,
    heading: 25.0,
    pitch: 6.0,
    fov: 70.0
  },
  {
    id: 'walker_18_petra_hotel_balconies',
    name: 'Hotel Petra Corner Facade & Christian Quarter Entry',
    x: 58.0,
    y: -4.0,
    z_eye: 1.7,
    heading: 348.0,
    pitch: 8.0,
    fov: 70.0
  },
  {
    id: 'walker_19_david_street_lower_steps',
    name: 'David Street Souq Looking West Up the Market Steps',
    x: 106.0,
    y: 2.0,
    z_eye: 1.7,
    heading: 270.0,
    pitch: 3.0,
    fov: 70.0
  },
  {
    id: 'walker_20_latin_patriarchate_north',
    name: 'Latin Patriarchate Lane Looking South to Jaffa Gate Plaza',
    x: 26.5,
    y: 44.0,
    z_eye: 1.7,
    heading: 178.0,
    pitch: 2.0,
    fov: 70.0
  }
];

// 10 Diagnostic Look-Up, Look-Down & Close-Up Inspection Viewpoints (diag_21 .. diag_30)
const DIAGNOSTIC_VIEWPOINTS = [
  {
    id: 'diag_21_gate_portal_lookup',
    name: 'DIAG: Jaffa Gate Outer Portal Look-Up (Arch & Machicolations)',
    x: -14.4,
    y: 0.6,
    z_eye: 1.7,
    heading: 80.0,
    pitch: 38.0,
    fov: 65.0
  },
  {
    id: 'diag_22_gate_passage_ceiling',
    name: 'DIAG: Jaffa Gate Interior Vault Ceiling & Doors Look-Up',
    x: -6.5,
    y: 1.7,
    z_eye: 1.7,
    heading: 85.0,
    pitch: 38.0,
    fov: 68.0
  },
  {
    id: 'diag_23_gate_curb_clutter',
    name: 'DIAG: Jaffa Gate Portal Curb & Corner Clutter Close-Up',
    x: -13.0,
    y: 3.0,
    z_eye: 1.7,
    heading: 55.0,
    pitch: -24.0,
    fov: 50.0
  },
  {
    id: 'diag_24_david_st_pier_closeup',
    name: 'DIAG: David Street Shop Pier, Awning & Display Close-Up',
    x: 86.0,
    y: 2.0,
    z_eye: 1.7,
    heading: 35.0,
    pitch: -6.0,
    fov: 50.0
  },
  {
    id: 'diag_25_petra_corner_door',
    name: 'DIAG: Petra Hotel Corner Arched Door, Shutters & Kerb Close-Up',
    x: 55.5,
    y: -1.5,
    z_eye: 1.7,
    heading: 15.0,
    pitch: -4.0,
    fov: 50.0
  },
  {
    id: 'diag_26_imperial_arcade_closeup',
    name: 'DIAG: New Imperial Hotel Ground Arcade & Balconies Close-Up',
    x: 34.0,
    y: 8.5,
    z_eye: 1.7,
    heading: 15.0,
    pitch: 18.0,
    fov: 55.0
  },
  {
    id: 'diag_27_moat_bridge_portal_closeup',
    name: 'DIAG: Citadel Moat Bridge Entrance Portal & Railing Close-Up',
    x: 43.8,
    y: -22.0,
    z_eye: 1.7,
    heading: 195.0,
    pitch: 6.0,
    fov: 55.0
  },
  {
    id: 'diag_28_latin_lane_arch_lookup',
    name: 'DIAG: Latin Patriarchate Flying-Buttress Arch Underside Look-Up',
    x: 26.5,
    y: 32.0,
    z_eye: 1.7,
    heading: 0.0,
    pitch: 35.0,
    fov: 65.0
  },
  {
    id: 'diag_29_david_steps_down_closeup',
    name: 'DIAG: David Street Stepped Paving & Souq Stalls Look-Down',
    x: 92.0,
    y: 2.0,
    z_eye: 1.7,
    heading: 90.0,
    pitch: -22.0,
    fov: 55.0
  },
  {
    id: 'diag_30_citadel_glacis_corner',
    name: 'DIAG: Citadel Sloped Stone Glacis & Moat Floor Close-Up',
    x: 14.0,
    y: -14.0,
    z_eye: 1.7,
    heading: 155.0,
    pitch: -16.0,
    fov: 55.0
  },
  {
    id: 'diag_31_plaza_floating_strips',
    name: 'DIAG: Jaffa Gate West Plaza Paving & Lamppost Bases Look-Down',
    x: -18.0,
    y: 3.5,
    z_eye: 1.6,
    heading: 95.0,
    pitch: -18.0,
    fov: 60.0
  },
  {
    id: 'diag_32_gate_south_exit_arch',
    name: 'DIAG: Jaffa Gate South Exit Portal Arch & Spandrel Look-Up',
    x: 1.5,
    y: -10.5,
    z_eye: 1.7,
    heading: 15.0,
    pitch: 18.0,
    fov: 65.0
  },
  {
    id: 'diag_33_swedish_study_keystone',
    name: 'DIAG: Swedish Study Centre Sign vs Arch Keystone Close-Up',
    x: 49.0,
    y: -1.0,
    z_eye: 1.7,
    heading: 35.0,
    pitch: 12.0,
    fov: 55.0
  },
  {
    id: 'diag_34_concave_corner_arches',
    name: 'DIAG: Concave Building Corner Shop Bay Junction (E+52.5, N+13.9)',
    x: 52.5,
    y: 13.9,
    z_eye: 1.7,
    heading: 36.0,
    pitch: 0.0,
    fov: 65.0
  },
  {
    id: 'diag_35_east_omar_wall_junction',
    name: 'DIAG: East Omar Square Building Junction (E+76.8, N+7.9)',
    x: 76.8,
    y: 7.9,
    z_eye: 1.7,
    heading: 66.0,
    pitch: 2.0,
    fov: 65.0
  }
];

function parseArgs() {
  const args = process.argv.slice(2);
  let outDir = path.join(ROOT_DIR, 'qa');
  let baseUrl = 'http://127.0.0.1:5173';
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--out' && args[i + 1]) {
      outDir = path.resolve(args[++i]);
    } else if (args[i] === '--url' && args[i + 1]) {
      baseUrl = args[++i];
    } else if (!args[i].startsWith('--')) {
      outDir = path.resolve(args[i]);
    }
  }
  return { outDir, baseUrl };
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
    srv.on('error', reject);
  });
}

async function isUrlReachable(url) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(1500) });
    return res.ok;
  } catch (_) {
    return false;
  }
}

async function ensureViteServer(baseUrl) {
  if (await isUrlReachable(baseUrl)) {
    console.log(`[QA] Using existing server at ${baseUrl}`);
    return null;
  }

  console.log(`[QA] Starting local Vite server at ${baseUrl}...`);
  const u = new URL(baseUrl);
  const port = u.port || '5173';
  const viteProc = spawn('npx', ['vite', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {
    cwd: ROOT_DIR,
    stdio: ['ignore', 'pipe', 'pipe']
  });

  const start = Date.now();
  while (Date.now() - start < 15000) {
    if (await isUrlReachable(baseUrl)) {
      console.log(`[QA] Vite server ready at ${baseUrl}`);
      return viteProc;
    }
    await new Promise((r) => setTimeout(r, 250));
  }

  viteProc.kill('SIGTERM');
  throw new Error(`Timed out waiting for Vite server at ${baseUrl}`);
}

class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.nextId = 1;
    this.pending = new Map();
  }

  async connect() {
    await new Promise((resolve, reject) => {
      const ws = new WebSocket(this.wsUrl);
      this.ws = ws;
      ws.onopen = () => resolve();
      ws.onerror = (err) => reject(err);
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(typeof event.data === 'string' ? event.data : event.data.toString());
          if (msg.id && this.pending.has(msg.id)) {
            const { resolve: res, reject: rej } = this.pending.get(msg.id);
            this.pending.delete(msg.id);
            if (msg.error) {
              rej(new Error(msg.error.message || JSON.stringify(msg.error)));
            } else {
              res(msg.result);
            }
          }
        } catch (_) {}
      };
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const res = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true
    });
    if (res?.exceptionDetails) {
      throw new Error(
        `Runtime.evaluate exception: ${res.exceptionDetails.text || ''} ${
          res.exceptionDetails.exception?.description || ''
        }`
      );
    }
    return res?.result?.value;
  }

  async screenshotPng(filePath) {
    const res = await this.send('Page.captureScreenshot', {
      format: 'png',
      fromSurface: true
    });
    const buf = Buffer.from(res.data, 'base64');
    fs.writeFileSync(filePath, buf);
  }

  close() {
    if (this.ws) {
      try {
        this.ws.close();
      } catch (_) {}
    }
  }
}

async function launchHeadlessChrome(debugPort) {
  const userDataDir = path.join('/tmp', `chrome-qa-${process.pid}-${Date.now()}`);
  const chromeArgs = [
    '--headless=new',
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',
    '--enable-webgl',
    '--ignore-gpu-blocklist',
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--window-size=1280,800',
    '--hide-scrollbars',
    '--mute-audio',
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${userDataDir}`,
    'about:blank'
  ];

  const chromeBin = fs.existsSync('/usr/bin/google-chrome-stable')
    ? '/usr/bin/google-chrome-stable'
    : 'google-chrome-stable';

  const proc = spawn(chromeBin, chromeArgs, {
    stdio: ['ignore', 'pipe', 'pipe']
  });

  const versionUrl = `http://127.0.0.1:${debugPort}/json/list`;
  const start = Date.now();
  let pageTarget = null;

  while (Date.now() - start < 15000) {
    try {
      const res = await fetch(versionUrl, { signal: AbortSignal.timeout(1000) });
      if (res.ok) {
        const list = await res.json();
        pageTarget = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
        if (pageTarget) break;
      }
    } catch (_) {}
    await new Promise((r) => setTimeout(r, 200));
  }

  if (!pageTarget) {
    proc.kill('SIGKILL');
    throw new Error('Failed to obtain DevTools page WebSocket URL from headless Chrome');
  }

  return {
    proc,
    userDataDir,
    wsUrl: pageTarget.webSocketDebuggerUrl
  };
}

function buildCompositesAndMetrics(outDir) {
  const pyBin = fs.existsSync(path.join(ROOT_DIR, '.venv/bin/python'))
    ? path.join(ROOT_DIR, '.venv/bin/python')
    : 'python3';

  const pyCode = `
import sys, os, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageOps

out_dir = sys.argv[1]
root_dir = sys.argv[2]
captured_path = os.path.join(out_dir, "captured.json")

with open(captured_path, "r", encoding="utf-8") as f:
    data = json.load(f)

views = data.get("views", [])
pair_images = []
model_thumbs = []
diag_thumbs = []
flagged_issues = []

for v in views:
    vid = v["id"]
    model_p = os.path.join(out_dir, f"{vid}_model.png")
    jitter_p = os.path.join(out_dir, f"{vid}_jitter.png")
    skykey_p = os.path.join(out_dir, f"{vid}_skykey.png")

    if not os.path.exists(model_p):
        continue

    im_model = Image.open(model_p).convert("RGB")
    arr_m = np.asarray(im_model, dtype=np.float32) # (800, 1280, 3)
    H, W, _ = arr_m.shape
    luma_m = 0.299 * arr_m[:, :, 0] + 0.587 * arr_m[:, :, 1] + 0.114 * arr_m[:, :, 2]
    mean_luma = float(np.mean(luma_m))

    # Identify non-sky mask (either from skykey or top-sky blue/horizon heuristic)
    # In skykey image, below-horizon holes are pure magenta (R>220, G<35, B>220)
    hole_fraction = 0.0
    if os.path.exists(skykey_p):
        arr_sky = np.asarray(Image.open(skykey_p).convert("RGB"), dtype=np.uint8)
        lower_sky = arr_sky[H // 2 :, :, :]
        mag_mask = (lower_sky[:, :, 0] > 220) & (lower_sky[:, :, 1] < 35) & (lower_sky[:, :, 2] > 220)
        hole_fraction = round(float(np.mean(mag_mask)), 5)

    # 1. Local 32x32 Block Z-Fighting Detector (maxBlockFlickerPct)
    # Uses 1-pixel neighborhood shift compensation so normal 1px parallax on edges is ignored,
    # while true coplanar Z-fighting flicker inside 32x32 blocks is caught accurately.
    jitter_diff = 0.0
    max_block_flicker_pct = 0.0
    max_block_coords = [0, 0]

    if os.path.exists(jitter_p):
        arr_j = np.asarray(Image.open(jitter_p).convert("RGB"), dtype=np.float32)
        raw_diff = np.mean(np.abs(arr_m - arr_j), axis=2)
        jitter_diff = round(float(np.mean(raw_diff) / 255.0), 5)

        # Compute spatial gradient magnitude on luma_m to mask out sharp silhouette edges
        gx_m = np.zeros_like(luma_m)
        gy_m = np.zeros_like(luma_m)
        gx_m[:, 1:-1] = np.abs(luma_m[:, 2:] - luma_m[:, :-2]) * 0.5
        gy_m[1:-1, :] = np.abs(luma_m[2:, :] - luma_m[:-2, :]) * 0.5
        edge_mag = np.hypot(gx_m, gy_m)

        # Shift-compensated minimum difference over 3x3 neighborhood
        min_nbr_diff = raw_diff.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = np.roll(np.roll(arr_m, dy, axis=0), dx, axis=1)
                d_nbr = np.mean(np.abs(arr_j - shifted), axis=2)
                min_nbr_diff = np.minimum(min_nbr_diff, d_nbr)

        # A pixel is flickering (Z-fighting) if its shift-compensated diff > 18.0
        # OR (raw_diff > 18.0 and it is NOT on a high-contrast silhouette edge: edge_mag < 14.0)
        flicker_mask = ((min_nbr_diff > 18.0) | ((raw_diff > 18.0) & (edge_mag < 14.0))).astype(np.float32)
        # Zero out 2px image border from np.roll
        flicker_mask[:2, :] = 0.0
        flicker_mask[-2:, :] = 0.0
        flicker_mask[:, :2] = 0.0
        flicker_mask[:, -2:] = 0.0

        # Reshape into 32x32 blocks (25 rows x 40 cols for 800x1280)
        bh, bw = 32, 32
        grid_h, grid_w = H // bh, W // bw
        blocks = flicker_mask[: grid_h * bh, : grid_w * bw].reshape(grid_h, bh, grid_w, bw)
        block_means = blocks.mean(axis=(1, 3)) # (25, 40)
        max_idx = int(np.argmax(block_means))
        by, bx = divmod(max_idx, grid_w)
        max_block_flicker_pct = round(float(block_means[by, bx] * 100.0), 2)
        max_block_coords = [int(bx), int(by)]

    # 2. Untextured / Flat Solid Patch Detector (flatPatchPct) in lower 75% of frame (16x16 blocks)
    y_start = H // 4
    sub_luma = luma_m[y_start:, :]
    sub_rgb = arr_m[y_start:, :, :]
    fb_s = 16
    f_gh = sub_luma.shape[0] // fb_s
    f_gw = sub_luma.shape[1] // fb_s
    luma_blocks = sub_luma[: f_gh * fb_s, : f_gw * fb_s].reshape(f_gh, fb_s, f_gw, fb_s)
    b_std = luma_blocks.std(axis=(1, 3))
    b_mean = luma_blocks.mean(axis=(1, 3))

    # Exclude sky pixels (blue > red + 22 and luma > 135) and deep shadow / blown highlights
    r_blocks = sub_rgb[: f_gh * fb_s, : f_gw * fb_s, 0].reshape(f_gh, fb_s, f_gw, fb_s).mean(axis=(1, 3))
    b_blocks = sub_rgb[: f_gh * fb_s, : f_gw * fb_s, 2].reshape(f_gh, fb_s, f_gw, fb_s).mean(axis=(1, 3))
    is_sky_block = (b_blocks > r_blocks + 22.0) & (b_mean > 135.0)
    valid_geo_block = (~is_sky_block) & (b_mean > 24.0) & (b_mean < 242.0)

    valid_count = int(np.sum(valid_geo_block))
    if valid_count > 0:
        flat_blocks = valid_geo_block & (b_std < 1.1)
        flat_patch_pct = round(float(np.sum(flat_blocks) / valid_count * 100.0), 2)
    else:
        flat_patch_pct = 0.0

    # 3. 1D UV Streak / Stripe Detector (uvStreakScore) in 32x32 blocks
    gx_full = np.abs(luma_m[:, 2:] - luma_m[:, :-2]) * 0.5
    gy_full = np.abs(luma_m[2:, :] - luma_m[:-2, :]) * 0.5
    gx_pad = np.pad(gx_full, ((0, 0), (1, 1)), mode="edge")
    gy_pad = np.pad(gy_full, ((1, 1), (0, 0)), mode="edge")

    bh, bw = 32, 32
    grid_h, grid_w = H // bh, W // bw
    gx_blk = gx_pad[: grid_h * bh, : grid_w * bw].reshape(grid_h, bh, grid_w, bw).mean(axis=(1, 3))
    gy_blk = gy_pad[: grid_h * bh, : grid_w * bw].reshape(grid_h, bh, grid_w, bw).mean(axis=(1, 3))
    energy_blk = gx_blk + gy_blk
    active_tex = energy_blk > 6.0
    if np.any(active_tex):
        aniso = np.maximum(gx_blk, gy_blk) / (np.minimum(gx_blk, gy_blk) + 0.75)
        # Blocks with extreme 1D stripe ratio > 5.5
        streak_blocks = active_tex & (aniso > 5.5)
        uv_streak_score = round(float(np.sum(streak_blocks) / np.sum(active_tex) * 100.0), 2)
    else:
        uv_streak_score = 0.0

    v["metrics"] = {
        "meanLuma": round(mean_luma, 2),
        "jitterDiff": jitter_diff,
        "maxBlockFlickerPct": max_block_flicker_pct,
        "maxBlockFlickerCoords": max_block_coords,
        "flatPatchPct": flat_patch_pct,
        "uvStreakScore": uv_streak_score,
        "groundHoleFraction": hole_fraction
    }

    if max_block_flicker_pct > 15.0 or flat_patch_pct > 8.0 or hole_fraction > 0.001 or uv_streak_score > 12.0:
        flagged_issues.append({
            "id": vid,
            "name": v["name"],
            "metrics": v["metrics"]
        })

    # Build annotated thumbnail
    thumb = im_model.resize((420, 262), Image.Resampling.LANCZOS)
    draw_t = ImageDraw.Draw(thumb)
    draw_t.rectangle([0, 0, 420, 44], fill=(14, 16, 22))
    label1 = f"{v['index']:02d}. {vid} ({v['pose']['x']:.1f},{v['pose']['y']:.1f} hdg {v['pose']['heading']:.0f} p {v['pose']['pitch']:.0f})"
    label2 = f"Flicker:{max_block_flicker_pct:.1f}% | Flat:{flat_patch_pct:.1f}% | UV:{uv_streak_score:.1f}% | Hole:{hole_fraction*100:.2f}%"
    status_col = (110, 240, 145) if (max_block_flicker_pct <= 15.0 and flat_patch_pct <= 8.0 and hole_fraction <= 0.001) else (255, 130, 110)
    draw_t.text((8, 5), label1[:54], fill=(243, 200, 120))
    draw_t.text((8, 23), label2[:58], fill=status_col)

    if v.get("isDiagnostic"):
        diag_thumbs.append(thumb)
    else:
        model_thumbs.append(thumb)

    # If reference photo exists, build side-by-side REAL | MODEL pair_<id>.jpg
    ref_candidates = [
        os.path.join(root_dir, "public", "references", f"{vid}.jpg"),
        os.path.join(root_dir, "data", "references", f"{vid}.jpg"),
    ]
    ref_path = next((p for p in ref_candidates if os.path.exists(p)), None)
    if ref_path:
        im_ref = Image.open(ref_path).convert("RGB")
        target_w, target_h = 640, 420
        ref_fit = ImageOps.fit(im_ref, (target_w, target_h), Image.Resampling.LANCZOS)
        mod_fit = ImageOps.fit(im_model, (target_w, target_h), Image.Resampling.LANCZOS)

        pair = Image.new("RGB", (target_w * 2 + 12, target_h + 40), (18, 20, 26))
        pair.paste(ref_fit, (0, 40))
        pair.paste(mod_fit, (target_w + 12, 40))

        d = ImageDraw.Draw(pair)
        d.text((14, 12), f"REAL PHOTOGRAPH — {v['name']}", fill=(255, 226, 166))
        d.text((target_w + 26, 12), f"3D WEBGL MODEL — ({v['pose']['x']:.1f}m, {v['pose']['y']:.1f}m, HDG {v['pose']['heading']:.0f} deg)", fill=(120, 220, 255))

        pair_out = os.path.join(out_dir, f"pair_{vid}.jpg")
        pair.save(pair_out, quality=90)
        v["pairImage"] = f"pair_{vid}.jpg"
        pair_images.append(pair)

# Save updated captured.json with image metrics
with open(captured_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

# Write qa_automated_audit.json
audit_summary = {
    "auditedAt": data.get("capturedAt"),
    "totalViews": len(views),
    "diagnosticViewsCount": len(diag_thumbs),
    "sceneAudit": data.get("sceneAudit", {}),
    "maxBlockFlickerAcrossAllViews": max((v.get("metrics", {}).get("maxBlockFlickerPct", 0.0) for v in views), default=0.0),
    "maxFlatPatchAcrossAllViews": max((v.get("metrics", {}).get("flatPatchPct", 0.0) for v in views), default=0.0),
    "maxGroundHoleAcrossAllViews": max((v.get("metrics", {}).get("groundHoleFraction", 0.0) for v in views), default=0.0),
    "flaggedIssuesCount": len(flagged_issues),
    "flaggedIssues": flagged_issues,
    "views": [
        {
            "index": v["index"],
            "id": v["id"],
            "name": v["name"],
            "isDiagnostic": v.get("isDiagnostic", False),
            "pose": v["pose"],
            "metrics": v.get("metrics", {})
        }
        for v in views
    ]
}
audit_path = os.path.join(out_dir, "qa_automated_audit.json")
with open(audit_path, "w", encoding="utf-8") as f:
    json.dump(audit_summary, f, indent=2)

# Also copy qa_automated_audit.json to root_dir/qa_automated_audit.json for easy pipeline inspection
try:
    with open(os.path.join(root_dir, "qa_automated_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
except Exception:
    pass

# Build contact_sheet_pairs.jpg (2 columns x 5 rows of REAL | MODEL pairs)
if pair_images:
    pw, ph = pair_images[0].size
    scale = 0.5
    sw, sh = int(pw * scale), int(ph * scale)
    cols = 2
    rows = math.ceil(len(pair_images) / cols)
    sheet_pairs = Image.new("RGB", (cols * sw + 16, rows * sh + 16), (12, 14, 18))
    for i, pim in enumerate(pair_images):
        r, c = divmod(i, cols)
        resized = pim.resize((sw, sh), Image.Resampling.LANCZOS)
        sheet_pairs.paste(resized, (8 + c * sw, 8 + r * sh))
    sheet_pairs.save(os.path.join(out_dir, "contact_sheet_pairs.jpg"), quality=88)

# Build contact_sheet_20views.jpg (4 columns x 5 rows of the 20 standard views)
if model_thumbs:
    tw, th = model_thumbs[0].size
    cols = 4
    rows = math.ceil(len(model_thumbs) / cols)
    sheet_all = Image.new("RGB", (cols * (tw + 8) + 8, rows * (th + 8) + 8), (12, 14, 18))
    for i, tim in enumerate(model_thumbs):
        r, c = divmod(i, cols)
        sheet_all.paste(tim, (8 + c * (tw + 8), 8 + r * (th + 8)))
    sheet_all.save(os.path.join(out_dir, "contact_sheet_20views.jpg"), quality=88)

# Build contact_sheet_diagnostics.jpg (2 columns x 5 rows of the 10 diagnostic close-up/look-up views)
if diag_thumbs:
    tw, th = diag_thumbs[0].size
    cols = 2
    rows = math.ceil(len(diag_thumbs) / cols)
    sheet_diag = Image.new("RGB", (cols * (tw + 10) + 10, rows * (th + 10) + 10), (12, 14, 18))
    for i, dim in enumerate(diag_thumbs):
        r, c = divmod(i, cols)
        sheet_diag.paste(dim, (10 + c * (tw + 10), 10 + r * (th + 10)))
    sheet_diag.save(os.path.join(out_dir, "contact_sheet_diagnostics.jpg"), quality=90)
`;

  execFileSync(pyBin, ['-c', pyCode, outDir, ROOT_DIR], { stdio: 'inherit' });
}

async function runCapture() {
  const { outDir, baseUrl } = parseArgs();
  fs.mkdirSync(outDir, { recursive: true });

  const placesPath = path.join(ROOT_DIR, 'public/data/places.json');
  const referencePlaces = fs.existsSync(placesPath)
    ? JSON.parse(fs.readFileSync(placesPath, 'utf-8'))
    : [];

  const allViewpoints = [
    ...referencePlaces.slice(0, 10).map((p) => ({ ...p, isReferencePlace: true, isDiagnostic: false })),
    ...EXTRA_WALKER_VIEWPOINTS.map((p) => ({ ...p, isReferencePlace: false, isDiagnostic: false })),
    ...DIAGNOSTIC_VIEWPOINTS.map((p) => ({ ...p, isReferencePlace: false, isDiagnostic: true }))
  ];

  const viteProc = await ensureViteServer(baseUrl);
  const debugPort = await getFreePort();
  let chrome = null;
  let cdp = null;

  try {
    console.log(`[QA] Launching headless Chrome on DevTools port ${debugPort}...`);
    chrome = await launchHeadlessChrome(debugPort);
    cdp = new CDPClient(chrome.wsUrl);
    await cdp.connect();

    await cdp.send('Page.enable');
    await cdp.send('Runtime.enable');
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: 1280,
      height: 800,
      deviceScaleFactor: 1,
      mobile: false
    });

    const targetUrl = `${baseUrl.replace(/\/$/, '')}/?clean=1`;
    console.log(`[QA] Navigating to ${targetUrl}...`);
    await cdp.send('Page.navigate', { url: targetUrl });

    // Wait for window.__QA__.ready === true
    const waitStart = Date.now();
    let isReady = false;
    while (Date.now() - waitStart < 45000) {
      try {
        isReady = await cdp.evaluate('Boolean(window.__QA__ && window.__QA__.ready === true)');
        if (isReady) break;
      } catch (_) {}
      await new Promise((r) => setTimeout(r, 300));
    }

    if (!isReady) {
      throw new Error('Timed out waiting for window.__QA__.ready === true');
    }
    console.log(`[QA] window.__QA__.ready is true. Auditing scene & capturing ${allViewpoints.length} viewpoints...`);

    const sceneAudit = await cdp.evaluate('window.__QA__.auditScene ? window.__QA__.auditScene() : {}');
    const capturedViews = [];

    for (let i = 0; i < allViewpoints.length; i++) {
      const vp = allViewpoints[i];
      const poseArgs = {
        x: Number(vp.x ?? 0),
        y: Number(vp.y ?? 0),
        z_eye: Number(vp.z_eye ?? 1.7),
        heading: Number(vp.heading ?? 90),
        pitch: Number(vp.pitch ?? 0),
        fov: Number(vp.fov ?? 70)
      };

      // 1. Normal PBR frame + traceView(6, 4)
      await cdp.evaluate('window.__QA__.setSkyKey(false)');
      const actualPose = await cdp.evaluate(`window.__QA__.setPose(${JSON.stringify(poseArgs)})`);
      await new Promise((r) => setTimeout(r, 60));

      const modelFile = `${vp.id}_model.png`;
      await cdp.screenshotPng(path.join(outDir, modelFile));
      const trace = await cdp.evaluate('window.__QA__.traceView(6, 4)');

      // 2. +0.5mm (0.0005m) camera depth perturbation for true coplanar 32x32 block z-fighting detection
      const jitterArgs = {
        ...poseArgs,
        x: Number((poseArgs.x + 0.0005).toFixed(5)),
        y: Number((poseArgs.y + 0.0005).toFixed(5))
      };
      await cdp.evaluate(`window.__QA__.setPose(${JSON.stringify(jitterArgs)})`);
      await new Promise((r) => setTimeout(r, 45));

      const jitterFile = `${vp.id}_jitter.png`;
      await cdp.screenshotPng(path.join(outDir, jitterFile));

      // 3. Sky-key frame (magenta below horizon) for ground-hole detection
      await cdp.evaluate(`window.__QA__.setPose(${JSON.stringify(poseArgs)})`);
      await cdp.evaluate('window.__QA__.setSkyKey(true)');
      await new Promise((r) => setTimeout(r, 45));

      const skykeyFile = `${vp.id}_skykey.png`;
      await cdp.screenshotPng(path.join(outDir, skykeyFile));
      await cdp.evaluate('window.__QA__.setSkyKey(false)');

      capturedViews.push({
        index: i + 1,
        id: vp.id,
        name: vp.name,
        isReferencePlace: vp.isReferencePlace,
        isDiagnostic: vp.isDiagnostic,
        pose: actualPose,
        files: {
          model: modelFile,
          jitter: jitterFile,
          skykey: skykeyFile
        },
        trace
      });

      console.log(
        `[QA] [${i + 1}/${allViewpoints.length}] ${vp.id}: groundZ=${actualPose.groundZ?.toFixed(2)}m, visibleIds=${trace.visibleIds.length}, sky=${trace.skyFraction}`
      );
    }

    const report = {
      capturedAt: new Date().toISOString(),
      totalViews: capturedViews.length,
      referencePlacesCount: referencePlaces.length,
      diagnosticViewsCount: DIAGNOSTIC_VIEWPOINTS.length,
      sceneAudit,
      views: capturedViews
    };

    const capturedJsonPath = path.join(outDir, 'captured.json');
    fs.writeFileSync(capturedJsonPath, JSON.stringify(report, null, 2));

    console.log('[QA] Running automated defect detectors & building contact sheets...');
    buildCompositesAndMetrics(outDir);
    console.log(`[QA] Done! Outputs & qa_automated_audit.json written to ${outDir}`);
  } finally {
    if (cdp) cdp.close();
    if (chrome?.proc) {
      chrome.proc.kill('SIGKILL');
      try {
        fs.rmSync(chrome.userDataDir, { recursive: true, force: true });
      } catch (_) {}
    }
    if (viteProc) {
      viteProc.kill('SIGTERM');
    }
  }
}

runCapture().catch((err) => {
  console.error('[QA] Fatal error:', err);
  process.exit(1);
});
