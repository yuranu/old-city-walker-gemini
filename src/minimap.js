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
    this.worldHalfSpan = options.worldHalfSpan ?? 160; // metres around center (scroll to zoom 80m..650m)
    this.centerX = 35;
    this.centerY = 0;
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
      let closestDist = Math.max(12, this.worldHalfSpan * 0.08);
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

    this.canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        e.stopPropagation();
        const factor = e.deltaY > 0 ? 1.22 : 0.82;
        this.worldHalfSpan = THREE.MathUtils.clamp(this.worldHalfSpan * factor, 75, 650);
      },
      { passive: false }
    );
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

    if (sceneRoot) {
      const box = new THREE.Box3();
      sceneRoot.traverse((obj) => {
        if (!obj.isMesh || !obj.geometry) return;
        const name = `${obj.name || ''} ${obj.parent?.name || ''}`.toLowerCase();
        if (
          name.includes('ground_terrain') ||
          name.includes('ground_paving') ||
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
        if (sizeX < 1.5 && sizeZ < 1.5) return;
        if (sizeX > 550 && sizeZ > 550) return;

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
          name.includes('gate')
        ) {
          this.footprints.walls.push(rect);
        } else if (
          name.includes('street') ||
          name.includes('paving') ||
          name.includes('step') ||
          name.includes('plaza') ||
          name.includes('bridge') ||
          name.includes('passage') ||
          name.includes('esplanade')
        ) {
          this.footprints.streets.push(rect);
        } else {
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
      cx: w * 0.5 + (x - this.centerX) * scale,
      cy: h * 0.5 - (y - this.centerY) * scale
    };
  }

  _canvasToENU(cx, cy) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const scale = w / (this.worldHalfSpan * 2);
    return {
      x: this.centerX + (cx - w * 0.5) / scale,
      y: this.centerY + (h * 0.5 - cy) / scale
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

    if (playerPose) {
      if (this.worldHalfSpan <= 320) {
        this.centerX = playerPose.x;
        this.centerY = playerPose.y;
      } else {
        // When zoomed out to full Old City view, center on Old City heart (380, 80)
        this.centerX = 380;
        this.centerY = 80;
      }
    }

    ctx.fillStyle = '#131720';
    ctx.fillRect(0, 0, w, h);

    // Subtle metric grid every 50m or 100m
    const gridStep = this.worldHalfSpan > 250 ? 100 : 50;
    ctx.strokeStyle = 'rgba(255, 238, 205, 0.06)';
    ctx.lineWidth = 1;
    const startX = Math.floor((this.centerX - this.worldHalfSpan) / gridStep) * gridStep;
    const endX = Math.ceil((this.centerX + this.worldHalfSpan) / gridStep) * gridStep;
    const startY = Math.floor((this.centerY - this.worldHalfSpan) / gridStep) * gridStep;
    const endY = Math.ceil((this.centerY + this.worldHalfSpan) / gridStep) * gridStep;

    for (let gx = startX; gx <= endX; gx += gridStep) {
      const p = this._enuToCanvas(gx, this.centerY);
      ctx.beginPath();
      ctx.moveTo(p.cx, 0);
      ctx.lineTo(p.cx, h);
      ctx.stroke();
    }
    for (let gy = startY; gy <= endY; gy += gridStep) {
      const q = this._enuToCanvas(this.centerX, gy);
      ctx.beginPath();
      ctx.moveTo(0, q.cy);
      ctx.lineTo(w, q.cy);
      ctx.stroke();
    }

    this._drawShapeList(
      ctx,
      this.footprints.streets,
      'rgba(198, 180, 148, 0.25)',
      'rgba(220, 202, 168, 0.32)',
      0.8
    );

    this._drawShapeList(
      ctx,
      this.footprints.buildings,
      'rgba(218, 192, 145, 0.42)',
      'rgba(245, 222, 178, 0.65)',
      1.0
    );

    this._drawShapeList(
      ctx,
      this.footprints.walls,
      'rgba(235, 185, 105, 0.68)',
      'rgba(255, 224, 156, 0.9)',
      1.2
    );

    // Draw place markers
    this.places.forEach((place, idx) => {
      const pxENU = Number(place.x ?? place.camera?.x ?? 0);
      const pyENU = Number(place.y ?? place.camera?.y ?? 0);
      const { cx, cy } = this._enuToCanvas(pxENU, pyENU);
      if (cx < -10 || cx > w + 10 || cy < -10 || cy > h + 10) return;
      const isActive = place.id === activePlaceId;

      ctx.beginPath();
      ctx.arc(cx, cy, isActive ? 6.5 : 4.8, 0, Math.PI * 2);
      ctx.fillStyle = isActive ? '#ffe2a6' : 'rgba(30, 36, 48, 0.9)';
      ctx.fill();
      ctx.lineWidth = isActive ? 2 : 1.3;
      ctx.strokeStyle = isActive ? '#ffffff' : '#f3c878';
      ctx.stroke();

      const label = idx < 9 ? String(idx + 1) : idx === 9 ? '0' : '•';
      ctx.fillStyle = isActive ? '#141820' : '#f7f3eb';
      ctx.font = 'bold 8px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, cx, cy + 0.5);
    });

    // Draw Player Position + FOV Cone
    if (playerPose) {
      const { cx, cy } = this._enuToCanvas(playerPose.x, playerPose.y);
      const hdgRad = ((playerPose.heading - 90) * Math.PI) / 180;
      const halfFov = ((fovDeg * 0.5) * Math.PI) / 180;
      const coneLen = 32;

      const grad = ctx.createRadialGradient(cx, cy, 2, cx, cy, coneLen);
      grad.addColorStop(0, 'rgba(96, 215, 255, 0.55)');
      grad.addColorStop(1, 'rgba(96, 215, 255, 0.0)');

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, coneLen, hdgRad - halfFov, hdgRad + halfFov);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#4ad4ff';
      ctx.fill();
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    }

    // North badge & zoom span
    ctx.fillStyle = 'rgba(243, 200, 120, 0.85)';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText('N ▲', w - 6, 6);

    ctx.fillStyle = 'rgba(220, 210, 190, 0.65)';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`±${Math.round(this.worldHalfSpan)}m (scroll zoom)`, 6, h - 14);
  }
}
