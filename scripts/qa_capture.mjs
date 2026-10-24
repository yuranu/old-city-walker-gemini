#!/usr/bin/env node
/**
 * Automated Headless Browser QA Capture & Side-by-Side Comparison Harness
 * Uses google-chrome-stable via raw Chrome DevTools WebSocket protocol (zero extra npm dependencies).
 *
 * Captures:
 *  - 10 recognizable reference places from public/data/places.json
 *  - 10 additional multi-angle walker viewpoints across the Jaffa Gate district (20 views total)
 *  - For each view:
 *      <id>_model.png   (normal PBR render)
 *      <id>_jitter.png  (+1cm camera jitter frame for z-fighting detection)
 *      <id>_skykey.png  (bright magenta below-horizon skykey for ground-hole detection)
 *      traceView(6, 4)  (scene manifest object/rule/material coverage)
 *  - Builds side-by-side REAL | MODEL comparison images (pair_<id>.jpg) and contact sheets!
 */

import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');

// 10 additional multi-angle walker viewpoints across the district (complementing the 10 places in places.json)
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
    x: 59.5,
    y: -32.0,
    z_eye: 1.7,
    heading: 185.0,
    pitch: 4.0,
    fov: 70.0
  },
  {
    id: 'walker_16_armenian_road_north_lookback',
    name: 'Armenian Patriarchate Road Looking North to Omar Square',
    x: 59.5,
    y: -48.0,
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
    '--enable-gpu',
    '--ignore-gpu-blocklist',
    '--use-angle=gl',
    '--window-size=1280,800',
    '--hide-scrollbars',
    '--mute-audio',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${userDataDir}`,
    'about:blank'
  ];

  const chromeBin = fs.existsSync('/usr/bin/google-chrome-stable')
    ? '/usr/bin/google-chrome-stable'
    : 'google-chrome-stable';

  const proc = spawn(chromeBin, chromeArgs, {
    env: { ...process.env, DRI_PRIME: '1' },
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

function buildCompositesAndMetrics(outDir, placesPath) {
  const pyBin = fs.existsSync(path.join(ROOT_DIR, '.venv/bin/python'))
    ? path.join(ROOT_DIR, '.venv/bin/python')
    : 'python3';

  const pyCode = `
import sys, os, json, math
from PIL import Image, ImageDraw, ImageOps

out_dir = sys.argv[1]
root_dir = sys.argv[2]
captured_path = os.path.join(out_dir, "captured.json")

with open(captured_path, "r", encoding="utf-8") as f:
    data = json.load(f)

views = data.get("views", [])
pair_images = []
model_thumbs = []

for v in views:
    vid = v["id"]
    model_p = os.path.join(out_dir, f"{vid}_model.png")
    jitter_p = os.path.join(out_dir, f"{vid}_jitter.png")
    skykey_p = os.path.join(out_dir, f"{vid}_skykey.png")

    if not os.path.exists(model_p):
        continue

    im_model = Image.open(model_p).convert("RGB")
    w, h = im_model.size

    # Compute brightness & RGB variance
    small = im_model.resize((160, 100), Image.Resampling.BILINEAR)
    px = list(small.getdata())
    mean_r = sum(p[0] for p in px) / len(px)
    mean_g = sum(p[1] for p in px) / len(px)
    mean_b = sum(p[2] for p in px) / len(px)
    mean_luma = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b

    # Compute 1cm jitter diff (z-fighting / shimmer check)
    jitter_diff = 0.0
    if os.path.exists(jitter_p):
        im_jit = Image.open(jitter_p).convert("RGB").resize((160, 100), Image.Resampling.BILINEAR)
        px_j = list(im_jit.getdata())
        diffs = [abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2]) for a, b in zip(px, px_j)]
        jitter_diff = round(sum(diffs) / (len(diffs) * 3.0 * 255.0), 5)

    # Compute skykey below-horizon magenta hole fraction
    hole_fraction = 0.0
    if os.path.exists(skykey_p):
        im_sky = Image.open(skykey_p).convert("RGB").resize((160, 100), Image.Resampling.NEAREST)
        px_s = list(im_sky.getdata())
        # Check lower half of image for bright magenta (R > 220, G < 35, B > 220)
        lower_px = px_s[len(px_s)//2:]
        magenta_hits = sum(1 for p in lower_px if p[0] > 220 and p[1] < 35 and p[2] > 220)
        hole_fraction = round(magenta_hits / max(1, len(lower_px)), 5)

    v["metrics"] = {
        "meanLuma": round(mean_luma, 2),
        "jitterDiff": jitter_diff,
        "groundHoleFraction": hole_fraction
    }

    # Build annotated thumbnail for 20-view contact sheet
    thumb = im_model.resize((420, 262), Image.Resampling.LANCZOS)
    draw_t = ImageDraw.Draw(thumb)
    draw_t.rectangle([0, 0, 420, 28], fill=(16, 18, 24))
    label = f"{v['index']:02d}. {vid} ({v['pose']['x']:.0f},{v['pose']['y']:.0f} hdg {v['pose']['heading']:.0f}deg)"
    draw_t.text((8, 7), label[:52], fill=(243, 200, 120))
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

# Build contact_sheet_20views.jpg (4 columns x 5 rows of all 20 views)
if model_thumbs:
    tw, th = model_thumbs[0].size
    cols = 4
    rows = math.ceil(len(model_thumbs) / cols)
    sheet_all = Image.new("RGB", (cols * (tw + 8) + 8, rows * (th + 8) + 8), (12, 14, 18))
    for i, tim in enumerate(model_thumbs):
        r, c = divmod(i, cols)
        sheet_all.paste(tim, (8 + c * (tw + 8), 8 + r * (th + 8)))
    sheet_all.save(os.path.join(out_dir, "contact_sheet_20views.jpg"), quality=88)
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
    ...referencePlaces.slice(0, 10).map((p) => ({ ...p, isReferencePlace: true })),
    ...EXTRA_WALKER_VIEWPOINTS.map((p) => ({ ...p, isReferencePlace: false }))
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
    console.log(`[QA] window.__QA__.ready is true. Capturing ${allViewpoints.length} viewpoints...`);

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
      await new Promise((r) => setTimeout(r, 80));

      const modelFile = `${vp.id}_model.png`;
      await cdp.screenshotPng(path.join(outDir, modelFile));
      const trace = await cdp.evaluate('window.__QA__.traceView(6, 4)');

      // 2. +1cm (0.01m) camera jitter frame for z-fighting detection
      const jitterArgs = {
        ...poseArgs,
        x: Number((poseArgs.x + 0.01).toFixed(4)),
        y: Number((poseArgs.y + 0.01).toFixed(4))
      };
      await cdp.evaluate(`window.__QA__.setPose(${JSON.stringify(jitterArgs)})`);
      await new Promise((r) => setTimeout(r, 50));

      const jitterFile = `${vp.id}_jitter.png`;
      await cdp.screenshotPng(path.join(outDir, jitterFile));

      // 3. Sky-key frame (magenta below horizon) for ground-hole detection
      await cdp.evaluate(`window.__QA__.setPose(${JSON.stringify(poseArgs)})`);
      await cdp.evaluate('window.__QA__.setSkyKey(true)');
      await new Promise((r) => setTimeout(r, 50));

      const skykeyFile = `${vp.id}_skykey.png`;
      await cdp.screenshotPng(path.join(outDir, skykeyFile));
      await cdp.evaluate('window.__QA__.setSkyKey(false)');

      capturedViews.push({
        index: i + 1,
        id: vp.id,
        name: vp.name,
        isReferencePlace: vp.isReferencePlace,
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
      views: capturedViews
    };

    const capturedJsonPath = path.join(outDir, 'captured.json');
    fs.writeFileSync(capturedJsonPath, JSON.stringify(report, null, 2));

    console.log('[QA] Building REAL | MODEL comparison pairs and contact sheets...');
    buildCompositesAndMetrics(outDir, placesPath);
    console.log(`[QA] Done! Outputs written to ${outDir}`);
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
