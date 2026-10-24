import * as THREE from 'three';

export class Minimap {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas ? canvas.getContext('2d') : null;
    this.places = [];
    this.footprints = {
      streets: [],
      buildings: [],
      walls: [],
      water: []
    };
    this.worldHalfSpan = options.worldHalfSpan ?? 115; // metres around origin
    this.visible = true;
    this.onSelectPlace = options.onSelectPlace || null;
    this.onTeleportXY = options.onTeleportXY || null;

    if (this.canvas) {
      this._bindEvents();
    }
  }

  _bindEvents() {
    this.canvas.addEventListener('click', (e) => {
      e.stopPropagation();
      const rect = this.canvas.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * this.canvas.width;
      const py = ((e.clientY - rect.top) / rect.height) * this.canvas.height;
      const world = this._canvasToENU(px, py);

      // Check if user clicked near a place marker first
      let closestPlace = null;
      let closestDist = 12; // metres threshold
      for (const place of this.places) {
        const pxENU = Number(place.x ?? place.camera?.x ?? 0);
        const pyENU = Number(place.y ?? place.camera?.y ?? 0);
        const d = Math.hypot(world.x - pxENU, world.y - pyENU);
        if (d < closestDist) {
          closestDist = d;
          closestPlace = place;
        }
      }

      if (closestPlace && this.onSelectPlace) {
        this.onSelectPlace(closestPlace);
      } else if (this.onTeleportXY) {
        this.onTeleportXY(world.x, world.y);
      }
    });
  }

  setPlaces(places) {
    this.places = Array.isArray(places) ? places : [];
  }

  /**
   * Populates footprints from scene meshes and optional scene_manifest.json data.
   */
  buildFromScene(sceneRoot, manifestData = null) {
    this.footprints = {
      streets: [],
      buildings: [],
      walls: [],
      water: []
    };

    // 1. Extract explicit polygon footprints from manifest if present
    const manifestItems = Array.isArray(manifestData)
      ? manifestData
      : Array.isArray(manifestData?.objects)
      ? manifestData.objects
      : Array.isArray(manifestData?.features)
      ? manifestData.features
      : [];

    for (const item of manifestItems) {
      if (Array.isArray(item.footprint) && item.footprint.length >= 3) {
        const category = (item.category || item.rule || item.id || '').toLowerCase();
        const poly = item.footprint.map((pt) =>
          Array.isArray(pt) ? { x: Number(pt[0]), y: Number(pt[1]) } : { x: Number(pt.x), y: Number(pt.y) }
        );
        if (category.includes('wall') || category.includes('citadel') || category.includes('gate') || category.includes('tower')) {
          this.footprints.walls.push({ type: 'poly', points: poly });
        } else if (category.includes('street') || category.includes('paving') || category.includes('plaza') || category.includes('step')) {
          this.footprints.streets.push({ type: 'poly', points: poly });
        } else {
          this.footprints.buildings.push({ type: 'poly', points: poly });
        }
      }
    }

    // 2. Also extract XZ footprints from actual 3D meshes in the scene so everything is visible
    if (sceneRoot) {
      const box = new THREE.Box3();
      sceneRoot.traverse((obj) => {
        if (!obj.isMesh || !obj.geometry) return;
        const name = `${obj.name || ''} ${obj.parent?.name || ''}`.toLowerCase();
        if (
          name.includes('ground_terrain') ||
          name.includes('sky') ||
          name.includes('foliage') ||
          name.includes('cable')
        ) {
          return;
        }

        box.setFromObject(obj);
        if (box.isEmpty()) return;

        const sizeX = box.max.x - box.min.x;
        const sizeZ = box.max.z - box.min.z;
        if (sizeX < 1.2 && sizeZ < 1.2) return; // skip tiny props
        if (sizeX > 240 && sizeZ > 240) return; // skip whole-terrain base mesh

        // Convert Three.js (x, z) bounding box to ENU (x = threeX, y = -threeZ)
        const rect = {
          type: 'rect',
          minX: box.min.x,
          maxX: box.max.x,
          minY: -box.max.z,
          maxY: -box.min.z
        };

        if (
          name.includes('wall') ||
          name.includes('citadel') ||
          name.includes('tower') ||
          name.includes('bastion') ||
          name.includes('gatehouse') ||
          name.includes('jaffa_gate')
        ) {
          this.footprints.walls.push(rect);
        } else if (
          name.includes('street') ||
          name.includes('paving') ||
          name.includes('step') ||
          name.includes('plaza') ||
          name.includes('bridge') ||
          name.includes('passage')
        ) {
          this.footprints.streets.push(rect);
        } else if (
          name.includes('building') ||
          name.includes('block') ||
          name.includes('hotel') ||
          name.includes('imperial') ||
          name.includes('petra') ||
          name.includes('christ_church') ||
          name.includes('shop') ||
          name.includes('market') ||
          name.includes('mamilla') ||
          name.includes('info')
        ) {
          this.footprints.buildings.push(rect);
        }
      });
    }
  }

  _enuToCanvas(x, y) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const scale = w / (this.worldHalfSpan * 2);
    return {
      cx: w * 0.5 + x * scale,
      cy: h * 0.5 - y * scale
    };
  }

  _canvasToENU(cx, cy) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const scale = w / (this.worldHalfSpan * 2);
    return {
      x: (cx - w * 0.5) / scale,
      y: (h * 0.5 - cy) / scale
    };
  }

  _drawShapeList(ctx, list, fillStyle, strokeStyle, lineWidth = 1) {
    ctx.fillStyle = fillStyle;
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;

    for (const item of list) {
      if (item.type === 'poly' && item.points.length >= 3) {
        ctx.beginPath();
        item.points.forEach((pt, idx) => {
          const { cx, cy } = this._enuToCanvas(pt.x, pt.y);
          if (idx === 0) ctx.moveTo(cx, cy);
          else ctx.lineTo(cx, cy);
        });
        ctx.closePath();
        ctx.fill();
        if (strokeStyle) ctx.stroke();
      } else if (item.type === 'rect') {
        const pMin = this._enuToCanvas(item.minX, item.maxY);
        const pMax = this._enuToCanvas(item.maxX, item.minY);
        const rw = Math.max(2, pMax.cx - pMin.cx);
        const rh = Math.max(2, pMax.cy - pMin.cy);
        ctx.fillRect(pMin.cx, pMin.cy, rw, rh);
        if (strokeStyle) ctx.strokeRect(pMin.cx, pMin.cy, rw, rh);
      }
    }
  }

  render(playerPose, activePlaceId = null, fovDeg = 65) {
    if (!this.ctx || !this.visible) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Background
    ctx.fillStyle = '#131720';
    ctx.fillRect(0, 0, w, h);

    // Subtle metric grid every 25m
    ctx.strokeStyle = 'rgba(255, 238, 205, 0.06)';
    ctx.lineWidth = 1;
    for (let m = -100; m <= 100; m += 25) {
      const p1 = this._enuToCanvas(m, -100);
      const p2 = this._enuToCanvas(m, 100);
      ctx.beginPath();
      ctx.moveTo(p1.cx, 0);
      ctx.lineTo(p2.cx, h);
      ctx.stroke();

      const q1 = this._enuToCanvas(-100, m);
      const q2 = this._enuToCanvas(100, m);
      ctx.beginPath();
      ctx.moveTo(0, q1.cy);
      ctx.lineTo(w, q2.cy);
      ctx.stroke();
    }

    // Draw streets / plazas
    this._drawShapeList(
      ctx,
      this.footprints.streets,
      'rgba(198, 180, 148, 0.25)',
      'rgba(220, 202, 168, 0.32)',
      0.8
    );

    // Draw building footprints
    this._drawShapeList(
      ctx,
      this.footprints.buildings,
      'rgba(218, 192, 145, 0.42)',
      'rgba(245, 222, 178, 0.65)',
      1.0
    );

    // Draw Old City & Citadel fortification walls
    this._drawShapeList(
      ctx,
      this.footprints.walls,
      'rgba(235, 185, 105, 0.68)',
      'rgba(255, 224, 156, 0.9)',
      1.2
    );

    // Draw place markers (1 - 0)
    this.places.forEach((place, idx) => {
      const pxENU = Number(place.x ?? place.camera?.x ?? 0);
      const pyENU = Number(place.y ?? place.camera?.y ?? 0);
      const { cx, cy } = this._enuToCanvas(pxENU, pyENU);
      const isActive = place.id === activePlaceId;

      ctx.beginPath();
      ctx.arc(cx, cy, isActive ? 6.5 : 5, 0, Math.PI * 2);
      ctx.fillStyle = isActive ? '#ffe2a6' : 'rgba(30, 36, 48, 0.9)';
      ctx.fill();
      ctx.lineWidth = isActive ? 2 : 1.3;
      ctx.strokeStyle = isActive ? '#ffffff' : '#f3c878';
      ctx.stroke();

      const label = idx === 9 ? '0' : String(idx + 1);
      ctx.fillStyle = isActive ? '#141820' : '#f7f3eb';
      ctx.font = 'bold 8.5px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, cx, cy + 0.5);
    });

    // Draw Player Position + FOV Cone
    if (playerPose) {
      const { cx, cy } = this._enuToCanvas(playerPose.x, playerPose.y);
      const hdgRad = ((playerPose.heading - 90) * Math.PI) / 180; // Canvas angle: 0° North = -PI/2
      const halfFov = ((fovDeg * 0.5) * Math.PI) / 180;
      const coneLen = 34;

      const grad = ctx.createRadialGradient(cx, cy, 2, cx, cy, coneLen);
      grad.addColorStop(0, 'rgba(96, 215, 255, 0.55)');
      grad.addColorStop(1, 'rgba(96, 215, 255, 0.0)');

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, coneLen, hdgRad - halfFov, hdgRad + halfFov);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      // Player dot
      ctx.beginPath();
      ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#4ad4ff';
      ctx.fill();
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    }

    // North badge
    ctx.fillStyle = 'rgba(243, 200, 120, 0.85)';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText('N ▲', w - 6, 6);
  }
}
