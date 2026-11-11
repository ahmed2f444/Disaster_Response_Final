/* =========================================================================
   AEGIS TACTICAL LEAFLET v1.9.4 VECTOR & CANVAS COMMAND ENGINE
   ========================================================================= */

// Intelligent API Base Resolution (same-origin over HTTP/HTTPS; localhost fallback for file://)
function getApiBase() {
  if (window.location.protocol === 'file:') {
    return 'http://127.0.0.1:8000';
  }
  return '';
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Tactical Coordinates & Bounding Box (Aligned to seed.py xBD Houston-East Sector)
const SECTOR_BBOX = {
  minLon: -95.212,
  maxLon: -95.066,
  minLat: 29.760,
  maxLat: 29.866,
  centerLon: -95.138,
  centerLat: 29.814,
};

const SECTOR_BOUNDS = [
  [SECTOR_BBOX.minLat, SECTOR_BBOX.minLon],
  [SECTOR_BBOX.maxLat, SECTOR_BBOX.maxLon],
];

// Keep full sector & node labels clear of top .camera-dock and bottom .legend-dock overlays
const SECTOR_OVERVIEW_PADDING = {
  paddingTopLeft: [24, 64],
  paddingBottomRight: [24, 72],
};

const OPERATIONAL_GRID_BOUNDS = [
  [29.764, -95.205],
  [29.862, -95.072],
];

// Node Metadata (Aligned 1:1 with seed.py NODES coordinates)
const NODE_META = {
  0: { name: "N0: CMD & MED DEPOT", type: "DEPOT", lon: -95.198, lat: 29.830 },
  1: { name: "N1: NORTH HWY INTERCHANGE", type: "NODE", lon: -95.158, lat: 29.857 },
  2: { name: "N2: NE USAR DEPOT", type: "DEPOT", lon: -95.095, lat: 29.849 },
  3: { name: "N3: SW INDUSTRIAL PKWY", type: "NODE", lon: -95.190, lat: 29.798 },
  4: { name: "N4: BAYOU BRIDGEHEAD", type: "CHOKEPOINT", lon: -95.134, lat: 29.819 },
  5: { name: "N5: EAST BELTWAY OVERPASS", type: "NODE", lon: -95.078, lat: 29.810 },
  6: { name: "N6: SW SHELTER [TRN]", type: "TARGET", lon: -95.166, lat: 29.769 },
  7: { name: "N7: SOUTH BAYOU [RSC]", type: "TARGET", lon: -95.122, lat: 29.774 },
  8: { name: "N8: SE TRAUMA [MED]", type: "TARGET", lon: -95.084, lat: 29.786 },
  9: { name: "N9: NW MIDTOWN FORK", type: "JUNCTION", lon: -95.170, lat: 29.827 },
  10: { name: "N10: NE RIDGE JUNCTION", type: "JUNCTION", lon: -95.118, lat: 29.839 },
  11: { name: "N11: SOUTH-CENTRAL SPUR", type: "JUNCTION", lon: -95.152, lat: 29.792 },
  12: { name: "N12: SE BAYOU OVERLOOK", type: "JUNCTION", lon: -95.106, lat: 29.798 },
};

// Organic diagonally-flowing Buffalo Bayou river centerline across the valley
const BAYOU_CENTERLINE = [
  [-95.216, 29.794],
  [-95.198, 29.804],
  [-95.174, 29.802],
  [-95.152, 29.812],
  [-95.134, 29.819],
  [-95.114, 29.821],
  [-95.094, 29.813],
  [-95.076, 29.809],
  [-95.060, 29.804],
];

function smoothOpenPolyline(points, passes = 2) {
  let pts = points.map(([lon, lat]) => [lon, lat]);
  for (let p = 0; p < passes; p += 1) {
    if (pts.length < 3) break;
    const next = [pts[0]];
    for (let i = 0; i < pts.length - 1; i += 1) {
      const [x0, y0] = pts[i];
      const [x1, y1] = pts[i + 1];
      next.push([0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1]);
      next.push([0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1]);
    }
    next.push(pts[pts.length - 1]);
    pts = next;
  }
  return pts;
}

function computeBayouGeometry(halfLat) {
  const smoothCenter = smoothOpenPolyline(BAYOU_CENTERLINE, 2);
  const midIdx = (smoothCenter.length - 1) / 2;
  const northBank = [];
  const southBank = [];
  smoothCenter.forEach(([lon, lat], idx) => {
    const u = (idx - midIdx) / Math.max(midIdx * 0.35, 1);
    const taper = halfLat * (1 + 0.18 * Math.exp(-(u * u)));
    northBank.push([lat + taper, lon]);
    southBank.push([lat - taper, lon]);
  });
  return {
    northBank,
    southBank,
    polygon: [...southBank, ...northBank.slice().reverse()],
  };
}

const UNIT_CONFIG = {
  "MED-01": { hex: "#00e5ff", label: "MED-01 (Medical Ambulance)", speed: 0.045 },
  "TRN-01": { hex: "#c084fc", label: "TRN-01 (Evacuation Transport)", speed: 0.038 },
  "RSC-01": { hex: "#34d399", label: "RSC-01 (USAR Tactical Drone)", speed: 0.055 },
};

const DAMAGE_STYLE_MAP = {
  no_damage: { color: "#10b981", fillColor: "#059669", radius: 3.6 },
  minor:     { color: "#eab308", fillColor: "#ca8a04", radius: 4.0 },
  major:     { color: "#f97316", fillColor: "#ea580c", radius: 4.4 },
  destroyed: { color: "#ef4444", fillColor: "#dc2626", radius: 4.8 },
};

// State Variables
let tacticalState = null;
let whatIfPlan = null;
let selectedBlockedEdges = new Set();
let selectedOpenedEdges = new Set();
let currentFloodDelta = 0.0;
let isFloodSimRunning = false;
let floodSimDirection = 1;
let floodSimTimer = null;
let selectedEntity = null;
let cameraMode = 'iso';
let selectedPreTileName = null;
let selectedPostTileName = null;
let selectedPreTileDataUrl = null;
let selectedPostTileDataUrl = null;
let pendingTileReads = { pre: null, post: null };
let currentInspectedBuildingData = null;
let toastTimer = null;

// Toast Notification Helper
function showToast(msg, isError = false) {
  const toast = document.getElementById("toastHud");
  if (!toast) return;
  if (toastTimer) clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.className = "toast-hud " + (isError ? "error" : "success");
  toast.style.display = "block";
  toastTimer = setTimeout(() => {
    toast.style.display = "none";
  }, 4200);
}

function positionTooltipAtEvent(evt) {
  const tooltip = document.getElementById("mapTooltip");
  const shell = document.getElementById("mapShell");
  if (!tooltip || !shell) return;
  const orig = evt && (evt.originalEvent || evt);
  if (!orig || typeof orig.clientX !== "number") return;
  const rect = shell.getBoundingClientRect();
  tooltip.style.display = "block";
  tooltip.style.left = `${Math.min(rect.width - 300, Math.max(12, orig.clientX - rect.left + 14))}px`;
  tooltip.style.top = `${Math.min(rect.height - 90, Math.max(12, orig.clientY - rect.top + 14))}px`;
}

function hideMapTooltip() {
  const tooltip = document.getElementById("mapTooltip");
  if (tooltip) tooltip.style.display = "none";
}

function buildDarkGridTileLayer() {
  const DarkGrid = L.GridLayer.extend({
    createTile(coords) {
      const tile = document.createElement("canvas");
      const size = this.getTileSize();
      tile.width = size.x;
      tile.height = size.y;
      const ctx = tile.getContext("2d");

      ctx.fillStyle = "#02050b";
      ctx.fillRect(0, 0, size.x, size.y);

      // Subtle tactical grid subdivisions (zero network tile dependency)
      ctx.strokeStyle = "rgba(56, 189, 248, 0.07)";
      ctx.lineWidth = 1;
      for (let g = 0; g <= size.x; g += 64) {
        ctx.beginPath();
        ctx.moveTo(g + 0.5, 0);
        ctx.lineTo(g + 0.5, size.y);
        ctx.moveTo(0, g + 0.5);
        ctx.lineTo(size.x, g + 0.5);
        ctx.stroke();
      }
      ctx.strokeStyle = "rgba(0, 229, 255, 0.14)";
      ctx.strokeRect(0.5, 0.5, size.x - 1, size.y - 1);

      ctx.fillStyle = "rgba(100, 116, 139, 0.25)";
      ctx.font = "10px monospace";
      ctx.fillText(`${coords.z}:${coords.x},${coords.y}`, 6, 14);
      return tile;
    },
  });
  return new DarkGrid({ minZoom: 10, maxZoom: 18 });
}

// -------------------------------------------------------------------------
// Offline Leaflet v1.9.4 Map Engine (Canvas + Vector Layers)
// -------------------------------------------------------------------------
class TacticalLeafletMap {
  constructor(containerId) {
    if (typeof L === "undefined") {
      throw new Error("Offline Leaflet v1.9.4 library failed to load");
    }
    this.container = document.getElementById(containerId);
    this.mapShell = document.getElementById("mapShell");
    this.canvasRenderer = L.canvas({ padding: 0.5 });

    this.map = L.map(this.container, {
      center: [SECTOR_BBOX.centerLat, SECTOR_BBOX.centerLon],
      zoom: 12.5,
      zoomSnap: 0.25,
      zoomDelta: 0.5,
      minZoom: 11,
      maxZoom: 18,
      preferCanvas: true,
      zoomControl: false,
      attributionControl: false,
    });

    L.control.zoom({ position: "bottomright" }).addTo(this.map);
    buildDarkGridTileLayer().addTo(this.map);

    this.layers = {
      satellite: L.layerGroup().addTo(this.map),
      water: L.layerGroup().addTo(this.map),
      secondaryRoads: L.layerGroup().addTo(this.map),
      buildings: L.layerGroup().addTo(this.map),
      lowConf: L.layerGroup().addTo(this.map),
      roads: L.layerGroup().addTo(this.map),
      routes: L.layerGroup().addTo(this.map),
      labels: L.layerGroup().addTo(this.map),
      incidents: L.layerGroup().addTo(this.map),
      fleet: L.layerGroup().addTo(this.map),
    };

    this._activeSatelliteKey = null;
    this.floodPolygon = null;
    this.floodBankNorth = null;
    this.floodBankSouth = null;

    this.buildFloodChannel();
    this.buildSecondaryStreetGrid();
    this.map.fitBounds(SECTOR_BOUNDS, SECTOR_OVERVIEW_PADDING);
    this.map.on("dragstart zoomstart", () => {
      if (cameraMode === "drone") {
        cameraMode = "iso";
        setActiveCamBtn("btnCamIso");
      }
    });
  }

  buildFloodChannel() {
    this.layers.water.clearLayers();
    const geom = computeBayouGeometry(0.011);

    this.floodPolygon = L.polygon(geom.polygon, {
      renderer: this.canvasRenderer,
      color: "#38bdf8",
      weight: 1.5,
      opacity: 0.75,
      fillColor: "#0284c7",
      fillOpacity: 0.32,
      interactive: false,
    }).addTo(this.layers.water);

    this.floodBankNorth = L.polyline(geom.northBank, {
      renderer: this.canvasRenderer,
      color: "#38bdf8",
      weight: 1.5,
      dashArray: "6, 5",
      interactive: false,
    }).addTo(this.layers.water);

    this.floodBankSouth = L.polyline(geom.southBank, {
      renderer: this.canvasRenderer,
      color: "#38bdf8",
      weight: 1.5,
      dashArray: "6, 5",
      interactive: false,
    }).addTo(this.layers.water);
  }

  updateWaterLevel(floodDelta) {
    if (!this.floodPolygon) return;
    const halfLat = Math.max(0.007, 0.011 + floodDelta * 0.017);
    const geom = computeBayouGeometry(halfLat);
    this.floodPolygon.setLatLngs(geom.polygon);
    this.floodBankNorth.setLatLngs(geom.northBank);
    this.floodBankSouth.setLatLngs(geom.southBank);

    let fillColor = "#0284c7";
    let strokeColor = "#38bdf8";
    let fillOpacity = 0.30 + Math.max(0, floodDelta) * 0.25;
    if (floodDelta >= 0.45) {
      fillColor = "#b91c1c";
      strokeColor = "#f87171";
    } else if (floodDelta >= 0.20) {
      fillColor = "#b45309";
      strokeColor = "#fbbf24";
    }
    this.floodPolygon.setStyle({ fillColor, color: strokeColor, fillOpacity: Math.min(0.62, fillOpacity) });
    this.floodBankNorth.setStyle({ color: strokeColor });
    this.floodBankSouth.setStyle({ color: strokeColor });
  }

  buildSecondaryStreetGrid() {
    this.layers.secondaryRoads.clearLayers();
    const streetStyle = {
      renderer: this.canvasRenderer,
      color: "#1e293b",
      weight: 1.25,
      opacity: 0.65,
      interactive: false,
    };
    const localStreets = [
      // Winding NW-to-NE neighborhood collector north of the bayou
      [[29.824, -95.204], [29.839, -95.181], [29.847, -95.148], [29.839, -95.116], [29.834, -95.078]],
      // Gentle residential arc south of the bayou linking SW industrial to SE trauma campus
      [[29.784, -95.195], [29.779, -95.152], [29.787, -95.114], [29.797, -95.075]],
      // Diagonal NW feeder across residential blocks
      [[29.855, -95.174], [29.836, -95.166], [29.821, -95.160], [29.804, -95.172]],
      // Diagonal NE subdivision connector avoiding N4 orthogonal crossing
      [[29.852, -95.114], [29.837, -95.119], [29.826, -95.112], [29.814, -95.092]],
      // Southern bayou feeder looping beneath N4 without orthogonal T-junctions
      [[29.801, -95.155], [29.793, -95.137], [29.784, -95.128], [29.773, -95.138]],
      // Eastern beltway spur
      [[29.828, -95.086], [29.815, -95.091], [29.798, -95.096], [29.782, -95.102]],
    ];
    localStreets.forEach(pts => {
      L.polyline(smoothOpenPolyline(pts, 1), streetStyle).addTo(this.layers.secondaryRoads);
    });
  }

  setLayerVisibility(key, visible) {
    const group = this.layers[key];
    if (!group) return;
    if (visible && !this.map.hasLayer(group)) {
      group.addTo(this.map);
    } else if (!visible && this.map.hasLayer(group)) {
      this.map.removeLayer(group);
    }
  }

  setViewportPreset(mode, activeDronePos = null) {
    cameraMode = mode;
    if (mode === "iso") {
      this.map.fitBounds(SECTOR_BOUNDS, { ...SECTOR_OVERVIEW_PADDING, animate: true });
    } else if (mode === "top") {
      this.map.fitBounds(OPERATIONAL_GRID_BOUNDS, { ...SECTOR_OVERVIEW_PADDING, animate: true });
    } else if (mode === "bridge") {
      this.map.setView([29.819, -95.134], 15, { animate: true });
    } else if (mode === "drone") {
      const target = activeDronePos || [29.792, -95.100];
      this.map.setView(target, 14, { animate: true });
    }
  }

  attachBuilder(builder) {
    this.builder = builder;
  }

  _updatePerfChips(renderMs, geoBytes, stateData) {
    const geoKb = geoBytes / 1024;
    const renderOk = renderMs < 50.0;
    const sizeOk = geoBytes < 102400;
    const bFeatures = (stateData && stateData.buildings_geojson && stateData.buildings_geojson.features) || [];
    const rFeatures = (stateData && stateData.roads_geojson && stateData.roads_geojson.features) || [];
    const lowConfCount = bFeatures.filter(f => f.properties && f.properties.status_flag === "LOW_CONFIDENCE").length;

    window.__lastMapRenderMs = Number(renderMs.toFixed(2));
    window.__lastGeoJsonBytes = geoBytes;
    window.__lastRenderStats = {
      renderMs: window.__lastMapRenderMs,
      geoBytes,
      geoKb: Number(geoKb.toFixed(2)),
      buildingsCount: bFeatures.length,
      roadsCount: rFeatures.length,
      lowConfCount,
      leafletVersion: typeof L !== "undefined" ? L.version : null,
    };

    const msEl = document.getElementById("hudRenderMsVal");
    if (msEl) {
      msEl.textContent = `${renderMs.toFixed(1)} ms (<50ms)`;
      msEl.style.color = renderOk ? "var(--accent-emerald)" : "var(--status-blocked)";
    }
    const kbEl = document.getElementById("hudGeoJsonKbVal");
    if (kbEl) {
      kbEl.textContent = `${geoKb.toFixed(1)} KB (<100KB)`;
      kbEl.style.color = sizeOk ? "var(--accent-emerald)" : "var(--status-blocked)";
    }
  }

  async syncSatelliteOverlay(stateData) {
    const bFeats = (stateData && stateData.buildings_geojson && stateData.buildings_geojson.features) || [];
    const hasLiveSat = bFeats.some(f => f.properties && f.properties.data_source === "resnet18_unet_pretrained");
    if (!hasLiveSat) {
      this.layers.satellite.clearLayers();
      this._activeSatelliteKey = null;
      return;
    }

    const apiBase = getApiBase();
    try {
      const res = await fetch(`${apiBase}/active-upload/latest_meta.json`, { cache: "no-store" });
      if (!res.ok) return;
      const meta = await res.json();
      const rawBounds = Array.isArray(meta && meta.bounds) ? meta.bounds : null;
      if (!rawBounds || rawBounds.length !== 4) return;

      const [minLon, minLat, maxLon, maxLat] = rawBounds.map(Number);
      if (![minLon, minLat, maxLon, maxLat].every(Number.isFinite)) return;

      const nextKey = `${meta.pair_digest || "upload"}:${minLon},${minLat},${maxLon},${maxLat}`;
      if (this._activeSatelliteKey === nextKey && this.layers.satellite.getLayers().length > 0) {
        return;
      }

      const latLngBounds = [
        [minLat, minLon],
        [maxLat, maxLon],
      ];
      this.layers.satellite.clearLayers();
      L.imageOverlay(
        `${apiBase}/active-upload/latest_post.png?v=${encodeURIComponent(meta.pair_digest || Date.now())}`,
        latLngBounds,
        { opacity: 0.78, interactive: false }
      ).addTo(this.layers.satellite);
      this._activeSatelliteKey = nextKey;

      const outsideSector =
        Math.abs(minLon - (-95.202)) > 0.01 ||
        Math.abs(minLat - 29.768) > 0.01 ||
        Math.abs(maxLon - (-95.078)) > 0.01 ||
        Math.abs(maxLat - 29.858) > 0.01;
      if (outsideSector) {
        this.map.fitBounds(latLngBounds, SECTOR_OVERVIEW_PADDING);
      }
    } catch (err) {
      console.warn("Satellite overlay sync skipped:", err && err.message);
    }
  }

  renderState(stateData, overridePlan = null) {
    if (!this.builder || !stateData) return;
    const encoded = new TextEncoder().encode(
      JSON.stringify({
        buildings_geojson: stateData.buildings_geojson || {},
        roads_geojson: stateData.roads_geojson || {},
      })
    );
    const geoBytes = encoded.length;

    const t0 = performance.now();
    this.builder.clear();
    this.builder.buildBuildings(stateData.buildings_geojson);
    this.builder.buildRoads(stateData.roads_geojson, overridePlan ? buildWhatIfHazardMap(overridePlan) : null);
    this.builder.buildNodes();
    this.builder.buildIncidents(stateData.incidents);
    this.builder.buildFleetUnits();
    this.builder.buildTwoLineRoutes(overridePlan || stateData.active_plan || null);
    const renderMs = performance.now() - t0;

    this._updatePerfChips(renderMs, geoBytes, stateData);
    this.syncSatelliteOverlay(stateData);
  }

  renderWhatIfUpdate(stateData, plan) {
    if (!this.builder || !stateData || !plan) return;
    const t0 = performance.now();
    this.builder.buildRoads(stateData.roads_geojson, buildWhatIfHazardMap(plan));
    this.builder.buildTwoLineRoutes(plan);
    const renderMs = performance.now() - t0;

    const geoBytes = window.__lastGeoJsonBytes || new TextEncoder().encode(
      JSON.stringify({
        buildings_geojson: stateData.buildings_geojson || {},
        roads_geojson: stateData.roads_geojson || {},
      })
    ).length;
    this._updatePerfChips(renderMs, geoBytes, stateData);
  }
}

function buildWhatIfHazardMap(plan) {
  const hazards = Object.assign({}, (plan && plan.dynamic_hazards) || {});
  selectedOpenedEdges.forEach((eid) => {
    if (!hazards[eid]) hazards[eid] = { status: "OPEN" };
  });
  selectedBlockedEdges.forEach((eid) => {
    hazards[eid] = { status: "BLOCKED" };
  });
  const newlyBlocked = (plan && plan.newly_blocked_edges) || [];
  newlyBlocked.forEach((eid) => {
    hazards[eid] = { status: "BLOCKED" };
  });
  return hazards;
}

// -------------------------------------------------------------------------
// Leaflet Vector & Canvas Sector Entities Builder
// -------------------------------------------------------------------------
class SectorMapBuilder {
  constructor(mapController) {
    this.mc = mapController;
    this.fleetUnits = {};
  }

  clear() {
    const { layers } = this.mc;
    layers.buildings.clearLayers();
    layers.lowConf.clearLayers();
    layers.roads.clearLayers();
    layers.routes.clearLayers();
    layers.labels.clearLayers();
    layers.incidents.clearLayers();
  }

  buildBuildings(geojson) {
    if (!geojson || !geojson.features) return;
    const tooltip = document.getElementById("mapTooltip");
    const liveSatFeatures = geojson.features.filter(
      (f) => f.properties && f.properties.data_source === "resnet18_unet_pretrained"
    );
    const featuresToRender = liveSatFeatures.length > 0 ? liveSatFeatures : geojson.features;
    const isSatMode = liveSatFeatures.length > 0;

    featuresToRender.forEach((feat) => {
      const coords = feat.geometry && feat.geometry.coordinates && feat.geometry.coordinates[0];
      if (!coords || coords.length === 0) return;

      let latSum = 0;
      let lonSum = 0;
      const latLngRing = coords.map(pt => {
        lonSum += pt[0];
        latSum += pt[1];
        return [pt[1], pt[0]];
      });
      const cLat = latSum / coords.length;
      const cLon = lonSum / coords.length;
      const props = feat.properties || {};
      const dmgLabel = normalizeDamageLabel(props.damage_class);
      const dmgTier = getDamageTier(dmgLabel);
      const styleCfg = DAMAGE_STYLE_MAP[dmgLabel] || DAMAGE_STYLE_MAP.minor;
      const isLowConf = props.status_flag === "LOW_CONFIDENCE";

      // Subtle polygon footprint on shared Canvas renderer
      L.polygon(latLngRing, {
        renderer: this.mc.canvasRenderer,
        color: styleCfg.color,
        weight: isSatMode ? 1.8 : 1,
        opacity: isSatMode ? 0.88 : 0.55,
        fillColor: styleCfg.fillColor,
        fillOpacity: isSatMode ? 0.26 : 0.28,
        interactive: false,
      }).addTo(this.mc.layers.buildings);

      // Interactive SVG/Canvas circleMarker per Section 11
      const marker = L.circleMarker([cLat, cLon], {
        renderer: this.mc.canvasRenderer,
        radius: styleCfg.radius,
        color: isLowConf ? "#e879f9" : styleCfg.color,
        weight: isLowConf ? 2 : 1.5,
        fillColor: styleCfg.fillColor,
        fillOpacity: 0.9,
      }).addTo(this.mc.layers.buildings);

      marker.on("mouseover", (evt) => {
        marker.setStyle({ radius: styleCfg.radius + 3, color: "#00e5ff", weight: 2.5 });
        if (tooltip) {
          tooltip.innerHTML = `
            <div style="color:var(--accent-cyan); font-weight:700;">BUILDING: ${escapeHtml(props.building_id)}</div>
            <div>Damage: <strong style="color:${dmgTier === 3 ? 'var(--status-blocked)' : (dmgTier >= 1 ? 'var(--status-hazard)' : 'var(--status-open)')}">${escapeHtml(dmgLabel)} (tier ${dmgTier})</strong></div>
            <div>Confidence: ${((props.max_prob || props.confidence || 0.9) * 100).toFixed(1)}% ${isLowConf ? '<span style="color:var(--accent-purple); font-weight:700;">[PURPLE: LOW CONF]</span>' : ''}</div>
            <div style="color:var(--text-dim); font-size:9.5px; margin-top:2px;">Click circle marker to inspect Grad-CAM &amp; confidence</div>
          `;
        }
        positionTooltipAtEvent(evt);
      });

      marker.on("mousemove", (evt) => positionTooltipAtEvent(evt));

      marker.on("mouseout", () => {
        marker.setStyle({
          radius: styleCfg.radius,
          color: isLowConf ? "#e879f9" : styleCfg.color,
          weight: isLowConf ? 2 : 1.5,
        });
        hideMapTooltip();
      });

      marker.on("click", () => {
        selectedEntity = { type: "building", data: feat };
        renderInspectorBuilding(feat);
        switchTab("tab-inspect");
      });

      // Pulsing SVG Halo Ring on Low-Confidence (<0.65) detections
      if (isLowConf) {
        const ringHtml = `<svg width="20" height="20" viewBox="0 0 20 20"><circle cx="10" cy="10" r="7.5" fill="none" stroke="#e879f9" stroke-width="1.6" stroke-dasharray="3.5 2.5"/></svg>`;
        L.marker([cLat, cLon], {
          interactive: false,
          icon: L.divIcon({
            className: "low-conf-svg-ring",
            html: ringHtml,
            iconSize: [20, 20],
            iconAnchor: [10, 10],
          }),
        }).addTo(this.mc.layers.lowConf);
      }
    });
  }

  buildRoads(geojson, dynamicHazards) {
    this.mc.layers.roads.clearLayers();
    if (!geojson || !geojson.features) return;
    const tooltip = document.getElementById("mapTooltip");

    geojson.features.forEach((feat) => {
      const coords = feat.geometry && feat.geometry.coordinates;
      if (!coords || coords.length < 2) return;
      const props = feat.properties || {};
      const eid = props.edge_id;
      const override = dynamicHazards ? dynamicHazards[eid] : null;
      let effectiveStatus = override ? override.status : props.edge_status;
      if (selectedBlockedEdges.has(eid)) {
        effectiveStatus = "BLOCKED";
      } else if (selectedOpenedEdges.has(eid) && !override) {
        effectiveStatus = "OPEN";
      }

      const latLngs = coords.map(pt => [pt[1], pt[0]]);
      let roadColor = "#10b981";
      let dashArray = null;
      if (effectiveStatus === "HAZARDOUS") {
        roadColor = "#f59e0b";
      } else if (effectiveStatus === "BLOCKED") {
        roadColor = "#ef4444";
        dashArray = "7, 5";
      }

      const edgeNum = Number.parseInt(String(eid).replace(/\D/g, ""), 10) || 1;
      const isConnector = edgeNum > 12;
      const baseWeight = isConnector ? 3.8 : 5.2;
      const hoverWeight = isConnector ? 6.5 : 8.2;
      const baseOpacity = isConnector ? 0.78 : 0.90;

      const edgeLine = L.polyline(latLngs, {
        renderer: this.mc.canvasRenderer,
        color: roadColor,
        weight: baseWeight,
        opacity: baseOpacity,
        dashArray,
      }).addTo(this.mc.layers.roads);

      edgeLine.on("mouseover", (evt) => {
        edgeLine.setStyle({ weight: hoverWeight, opacity: 1.0 });
        if (tooltip) {
          tooltip.innerHTML = `
            <div style="color:var(--accent-cyan); font-weight:700;">${isConnector ? "CONNECTOR ARTERIAL" : "PRIMARY ARTERIAL"}: ${escapeHtml(eid)}</div>
            <div>Status: <strong style="color:${effectiveStatus === 'BLOCKED' ? 'var(--status-blocked)' : (effectiveStatus === 'HAZARDOUS' ? 'var(--status-hazard)' : 'var(--status-open)')}">${escapeHtml(effectiveStatus)}</strong></div>
            <div>Flood Depth: ${Number(props.flood_depth_m || 0).toFixed(2)}m | Debris: ${Number(props.debris_score || 0).toFixed(2)}</div>
            <div style="color:var(--accent-amber); font-size:9.5px; margin-top:2px;">Click segment to toggle blockade &amp; recompute What-If</div>
          `;
        }
        positionTooltipAtEvent(evt);
      });

      edgeLine.on("mousemove", (evt) => positionTooltipAtEvent(evt));

      edgeLine.on("mouseout", () => {
        edgeLine.setStyle({ weight: baseWeight, opacity: baseOpacity });
        hideMapTooltip();
      });

      edgeLine.on("click", () => {
        selectedEntity = { type: "road", data: feat };
        renderInspectorRoad(feat);
        switchTab("tab-inspect");
        window.toggleRoadClosure(eid);
      });
    });
  }

  buildNodes() {
    this.mc.layers.labels.clearLayers();
    Object.entries(NODE_META).forEach(([, meta]) => {
      const isJunction = meta.type === "JUNCTION";
      const color = meta.type === "DEPOT"
        ? "#00e5ff"
        : (meta.type === "TARGET" ? "#10b981" : (meta.type === "CHOKEPOINT" ? "#f59e0b" : (isJunction ? "#64748b" : "#94a3b8")));
      L.circleMarker([meta.lat, meta.lon], {
        renderer: this.mc.canvasRenderer,
        radius: meta.type === "DEPOT" ? 8 : (isJunction ? 4.5 : 6),
        color,
        weight: isJunction ? 1.5 : 2,
        fillColor: "#0b1322",
        fillOpacity: 0.95,
        interactive: false,
      }).addTo(this.mc.layers.labels);

      L.marker([meta.lat, meta.lon], {
        interactive: false,
        icon: L.divIcon({
          className: "",
          html: `<div class="tactical-node-pill${isJunction ? " junction" : ""}" style="border-color:${color};">${meta.name.split(":")[0]}</div>`,
          iconSize: isJunction ? [28, 15] : [34, 18],
          iconAnchor: isJunction ? [14, -6] : [17, -8],
        }),
      }).addTo(this.mc.layers.labels);
    });
  }

  buildIncidents(incidents) {
    this.mc.layers.incidents.clearLayers();
    if (!Array.isArray(incidents)) return;
    const tooltip = document.getElementById("mapTooltip");
    incidents.forEach((inc) => {
      if (!inc.location) return;
      const pos = [inc.location.lat, inc.location.lon];
      const incMarker = L.circleMarker(pos, {
        renderer: this.mc.canvasRenderer,
        radius: 8,
        color: "#ef4444",
        weight: 2.5,
        fillColor: "#7f1d1d",
        fillOpacity: 0.85,
        interactive: true,
      }).addTo(this.mc.layers.incidents);

      incMarker.on("mouseover", (evt) => {
        if (tooltip) {
          tooltip.innerHTML = `
            <div style="color:#f87171; font-weight:700;">INCIDENT: ${escapeHtml(inc.incident_id)}</div>
            <div>Severity: <strong>${escapeHtml(String(inc.severity || "high").toUpperCase())}</strong></div>
            <div>Capability: ${escapeHtml(String(inc.required_capability || "rescue").toUpperCase())} | Status: ${escapeHtml(String(inc.status || "open").toUpperCase())}</div>
          `;
        }
        positionTooltipAtEvent(evt);
      });
      incMarker.on("mousemove", (evt) => positionTooltipAtEvent(evt));
      incMarker.on("mouseout", () => hideMapTooltip());

      L.marker(pos, {
        interactive: false,
        icon: L.divIcon({
          className: "",
          html: `<div class="tactical-incident-badge">${escapeHtml(inc.incident_id)} (${escapeHtml(String(inc.severity || "high").toUpperCase())})</div>`,
          iconSize: [96, 18],
          iconAnchor: [48, 22],
        }),
      }).addTo(this.mc.layers.incidents);
    });
  }

  buildTwoLineRoutes(activePlan) {
    this.mc.layers.routes.clearLayers();
    if (!activePlan) return;

    const roadsMap = {};
    if (tacticalState && tacticalState.roads_geojson) {
      tacticalState.roads_geojson.features.forEach(f => {
        roadsMap[f.properties.edge_id] = f.geometry.coordinates;
      });
    }

    const unitList = (activePlan.routes && activePlan.routes.length > 0) ? activePlan.routes : [activePlan];

    unitList.forEach(unit => {
      // 1. Solid cyan line: CSP safety-filtered path (masked_path / edge_ids)
      let safeLatLngs = [];
      if (unit.masked_path && unit.masked_path.length > 1) {
        safeLatLngs = unit.masked_path.map(pt => [pt.lat, pt.lon]);
      } else if (unit.edge_ids) {
        safeLatLngs = buildContinuousEdgeLatLngs(unit.edge_ids, roadsMap);
      }

      if (safeLatLngs.length > 1) {
        L.polyline(safeLatLngs, {
          renderer: this.mc.canvasRenderer,
          color: "#00e5ff",
          weight: 4,
          opacity: 0.95,
          interactive: false,
        }).addTo(this.mc.layers.routes);
      }

      // 2. Dashed rose line: Raw pre-mask PPO proposal (ppo_proposed_path)
      const rawLatLngs = [];
      const ppoPath = (unit.ppo_proposed_path && unit.ppo_proposed_path.length > 1)
        ? unit.ppo_proposed_path
        : (activePlan.ppo_proposed_path || []);

      if (ppoPath.length > 1) {
        ppoPath.forEach(pt => rawLatLngs.push([pt.lat, pt.lon]));
      } else if (safeLatLngs.length > 1) {
        rawLatLngs.push(safeLatLngs[0], [29.819, -95.134], safeLatLngs[safeLatLngs.length - 1]);
      }

      if (rawLatLngs.length > 1) {
        L.polyline(rawLatLngs, {
          renderer: this.mc.canvasRenderer,
          color: "#f43f5e",
          weight: 3,
          opacity: 0.9,
          dashArray: "8, 6",
          interactive: false,
        }).addTo(this.mc.layers.routes);
      }
    });
  }

  buildFleetUnits() {
    this.mc.layers.fleet.clearLayers();
    this.fleetUnits = {};
    const defaults = {
      "MED-01": [29.830, -95.198],
      "TRN-01": [29.830, -95.198],
      "RSC-01": [29.849, -95.095],
    };

    Object.entries(UNIT_CONFIG).forEach(([unitId, cfg], idx) => {
      const pos = defaults[unitId] || [29.815, -95.125];
      const dot = L.circleMarker(pos, {
        renderer: this.mc.canvasRenderer,
        radius: 6.5,
        color: "#ffffff",
        weight: 2,
        fillColor: cfg.hex,
        fillOpacity: 1.0,
        interactive: false,
      }).addTo(this.mc.layers.fleet);

      const badge = L.marker(pos, {
        interactive: false,
        icon: L.divIcon({
          className: "",
          html: `<div class="tactical-unit-badge" style="border-color:${cfg.hex}; color:${cfg.hex};">${unitId}</div>`,
          iconSize: [54, 18],
          iconAnchor: [-8, 9],
        }),
      }).addTo(this.mc.layers.fleet);

      this.fleetUnits[unitId] = {
        dot,
        badge,
        config: cfg,
        progress: idx * 0.33,
        latLng: pos,
      };
    });
  }

  updateFleetMovement(dtSec, activePlan) {
    if (!activePlan || !activePlan.routes) return;
    const roadsMap = {};
    if (tacticalState && tacticalState.roads_geojson) {
      tacticalState.roads_geojson.features.forEach(f => {
        roadsMap[f.properties.edge_id] = f.geometry.coordinates;
      });
    }

    activePlan.routes.forEach(unit => {
      const unitObj = this.fleetUnits[unit.unit_id];
      if (!unitObj) return;

      const pts = buildContinuousEdgeLatLngs(unit.edge_ids || [], roadsMap);
      if (pts.length < 2) return;

      unitObj.progress = (unitObj.progress + unitObj.config.speed * dtSec) % 1.0;
      const segFloat = unitObj.progress * (pts.length - 1);
      const idx = Math.min(pts.length - 2, Math.floor(segFloat));
      const frac = segFloat - idx;
      const lat = pts[idx][0] + (pts[idx + 1][0] - pts[idx][0]) * frac;
      const lon = pts[idx][1] + (pts[idx + 1][1] - pts[idx][1]) * frac;
      unitObj.latLng = [lat, lon];
      unitObj.dot.setLatLng(unitObj.latLng);
      unitObj.badge.setLatLng(unitObj.latLng);

      const nowMs = performance.now();
      if (
        cameraMode === "drone" &&
        currentScreen === "screen-map" &&
        unit.unit_id === "RSC-01" &&
        (!this._lastDronePanMs || nowMs - this._lastDronePanMs > 150)
      ) {
        this._lastDronePanMs = nowMs;
        this.mc.map.panTo(unitObj.latLng, { animate: false });
      }
    });
  }
}

// -------------------------------------------------------------------------
// Tactical Application UI Coordinator & REST Client
// -------------------------------------------------------------------------
function buildContinuousEdgeLatLngs(edgeIds, roadsMap) {
  const pts = [];
  if (!Array.isArray(edgeIds) || edgeIds.length === 0) return pts;
  const sqDist = (a, b) => ((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2);

  edgeIds.forEach((eid, idx) => {
    const raw = roadsMap[eid];
    if (!raw || raw.length < 2) return;
    let seg = raw.map(pt => [pt[1], pt[0]]);
    if (idx === 0 && edgeIds.length > 1) {
      const nextRaw = roadsMap[edgeIds[1]];
      if (nextRaw && nextRaw.length >= 2) {
        const nextFirst = [nextRaw[0][1], nextRaw[0][0]];
        const nextLast = [nextRaw[nextRaw.length - 1][1], nextRaw[nextRaw.length - 1][0]];
        const distEnd = Math.min(sqDist(seg[seg.length - 1], nextFirst), sqDist(seg[seg.length - 1], nextLast));
        const distStart = Math.min(sqDist(seg[0], nextFirst), sqDist(seg[0], nextLast));
        if (distStart < distEnd) seg = seg.slice().reverse();
      }
    } else if (pts.length > 0) {
      const prevEnd = pts[pts.length - 1];
      if (sqDist(prevEnd, seg[seg.length - 1]) < sqDist(prevEnd, seg[0])) {
        seg = seg.slice().reverse();
      }
    }
    seg.forEach((p, pIdx) => {
      if (pts.length === 0 || pIdx > 0 || sqDist(pts[pts.length - 1], p) > 1e-10) {
        pts.push(p);
      }
    });
  });
  return pts;
}

let tacticalMap = null;
let sectorBuilder = null;

async function fetchState() {
  const apiBase = getApiBase();
  try {
    const res = await fetch(`${apiBase}/api/state`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    tacticalState = await res.json();

    tacticalMap.renderState(tacticalState, whatIfPlan);

    updateUI();
  } catch (err) {
    console.warn("API State fetch error, verifying connection:", err.message);
    showToast("Backend link: " + err.message + " (Targeting " + (apiBase || "origin") + ")", true);
  }
}

async function fetchAuditLog() {
  const apiBase = getApiBase();
  try {
    const res = await fetch(`${apiBase}/api/audit-log`);
    if (!res.ok) return;
    const data = await res.json();
    renderAuditTable(data.entries, data.valid, data.count);
  } catch (err) {
    console.warn("Audit log fetch error:", err.message);
  }
}

function rollbackEdgeOverrides(rollbackState) {
  if (!rollbackState) return;
  if (rollbackState instanceof Set) {
    selectedBlockedEdges = new Set(rollbackState);
  } else {
    if (rollbackState.blocked) selectedBlockedEdges = new Set(rollbackState.blocked);
    if (rollbackState.opened) selectedOpenedEdges = new Set(rollbackState.opened);
  }
  updateClosuresUI();
  if (selectedEntity && selectedEntity.type === "road" && selectedEntity.data && selectedEntity.data.properties) {
    renderInspectorRoad(selectedEntity.data);
  }
}

async function submitWhatIf(silent = false, rollbackState = null) {
  const apiBase = getApiBase();
  const payload = {
    blocked_edges: Array.from(selectedBlockedEdges),
    opened_edges: Array.from(selectedOpenedEdges),
    flood_delta: currentFloodDelta,
  };

  try {
    const res = await fetch(`${apiBase}/api/routes/what-if`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.status === 501) {
      const errBody = await res.json().catch(() => ({}));
      const msg = errBody.detail || "Pending Portion 2/3 integration (HTTP 501).";
      rollbackEdgeOverrides(rollbackState);
      if (typeof isFloodSimRunning !== "undefined" && isFloodSimRunning) {
        toggleFloodSimulation();
      }
      if (!silent) {
        showToast(msg, true);
      } else {
        console.info("What-If endpoint shell (501):", msg);
      }
      return;
    }
    if (!res.ok) throw new Error("What-If failed");
    whatIfPlan = await res.json();

    tacticalMap.renderWhatIfUpdate(tacticalState, whatIfPlan);
    updateUI();
  } catch (err) {
    rollbackEdgeOverrides(rollbackState);
    console.warn("What-if call error:", err.message);
    if (!silent) showToast("What-If failed: " + err.message, true);
  }
}

async function recomputeDispatch(options = {}) {
  const isArbitrary = Boolean(options && options.arbitraryBlockage);
  const btn = document.getElementById(isArbitrary ? "btnSimulateSatBlockage" : "btnRecomputeDispatch");
  const label = btn ? btn.querySelector("span") : null;
  const origLabel = isArbitrary
    ? "SIMULATE LIVE SATELLITE BLOCKAGE (ARBITRARY MARL REROUTE)"
    : "RECOMPUTE MARL DISPATCH (PORTION 2)";
  if (btn) btn.disabled = true;
  if (label) label.textContent = isArbitrary ? "CAPTURING & REROUTING..." : "RECOMPUTING MARL DISPATCH...";

  const apiBase = getApiBase();
  const prevBlockedIds = new Set(
    ((tacticalState && tacticalState.roads_geojson && tacticalState.roads_geojson.features) || [])
      .filter(f => f.properties && f.properties.edge_status === "BLOCKED")
      .map(f => f.properties.edge_id)
  );
  const payload = {
    water_level_delta_m: currentFloodDelta,
    blocked_edges: Array.from(selectedBlockedEdges),
    opened_edges: Array.from(selectedOpenedEdges),
    arbitrary_blockage: isArbitrary,
  };

  try {
    const res = await fetch(`${apiBase}/api/routes/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.status === 501) {
      const errBody = await res.json().catch(() => ({}));
      showToast(errBody.detail || "Routing module not implemented (HTTP 501).", true);
      return;
    }
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${res.status}`);
    }
    const plan = await res.json();
    whatIfPlan = null;
    selectedBlockedEdges.clear();
    selectedOpenedEdges.clear();
    currentFloodDelta = 0.0;
    const slider = document.getElementById("floodSlider");
    if (slider) slider.value = 0.0;
    updateSurgeGauge(0.0);
    updateClosuresUI();
    await fetchState();
    await fetchAuditLog();
    if (currentScreen === "screen-route") await populateScreenRouteData();
    const msText = typeof plan.compute_ms === "number" ? ` in ${plan.compute_ms.toFixed(1)}ms` : "";
    if (isArbitrary) {
      const newBlocks = (plan.blocked_edges || []).filter(eid => !prevBlockedIds.has(eid));
      const tag = newBlocks.length ? `Blocked ${newBlocks.join(", ")} -> ` : "";
      showToast(`Live satellite blockage simulated: ${tag}MARL ${plan.route_id}${msText}.`, false);
    } else {
      showToast(`MARL Dispatch ${plan.route_id} ready${msText}.`, false);
    }
  } catch (err) {
    showToast("Dispatch recompute failed: " + err.message, true);
  } finally {
    if (btn) btn.disabled = false;
    if (label) label.textContent = origLabel;
  }
}

function simulateLiveSatelliteBlockage() {
  return recomputeDispatch({ arbitraryBlockage: true });
}

// Robust Commander Approval Gate calling Endpoint 16 (POST /api/routes/{route_id}/approve)
async function approvePlan() {
  return commanderDecisionAction("approve");
}

async function resetBaseline() {
  const resetBtn = document.getElementById("resetBtn");
  const resetLabel = resetBtn ? resetBtn.querySelector("span") : null;
  if (resetBtn) resetBtn.disabled = true;
  if (resetLabel) resetLabel.textContent = "RESETTING...";
  if (isFloodSimRunning) toggleFloodSimulation();
  selectedBlockedEdges.clear();
  selectedOpenedEdges.clear();
  currentFloodDelta = 0.0;
  document.getElementById("floodSlider").value = 0.0;
  whatIfPlan = null;
  updateSurgeGauge(0.0);
  updateClosuresUI();

  const apiBase = getApiBase();
  try {
    const rstRes = await fetch(`${apiBase}/api/reset-sector`, { method: "POST" });
    if (!rstRes.ok) throw new Error(`HTTP ${rstRes.status}`);
    const rstData = await rstRes.json();
    if (rstData && rstData.state) {
      tacticalState = rstData.state;
      tacticalMap.renderState(tacticalState, null);
      updateUI();
    } else {
      await fetchState();
    }
    await fetchAuditLog();
    if (currentScreen === "screen-route") await populateScreenRouteData();
    if (currentScreen === "screen-building") await populateScreenBuildingData(currentInspectedBuildingId);
    showToast("Sector reset to clean baseline.", false);
  } catch (err) {
    showToast("Baseline reset error: " + err.message, true);
  } finally {
    if (resetBtn) resetBtn.disabled = false;
    if (resetLabel) resetLabel.textContent = "RESET";
  }
}

function updateDriftFooterUI(driftDistance, feedbackCount, activeAdapterVersion) {
  if (typeof driftDistance === "number") {
    const driftEl = document.getElementById("statusDriftVal");
    const driftBadge = document.getElementById("statusDriftBadge");
    if (driftEl) driftEl.textContent = driftDistance.toFixed(3);
    if (driftBadge) {
      if (driftDistance >= 0.15) {
        driftBadge.textContent = "DRIFT ALERT (>=0.15)";
        driftBadge.style.color = "#f87171";
      } else if (driftDistance >= 0.08) {
        driftBadge.textContent = "SOFT WARNING (>=0.08)";
        driftBadge.style.color = "#fbbf24";
      } else {
        driftBadge.textContent = "NORMAL (<0.08)";
        driftBadge.style.color = "#34d399";
      }
    }
  }
  if (typeof feedbackCount === "number") {
    const feedEl = document.getElementById("statusFeedbackCount");
    if (feedEl) feedEl.textContent = `${feedbackCount} QUEUED`;
  }
  if (typeof activeAdapterVersion === "string" && activeAdapterVersion.length > 0) {
    const label = activeAdapterVersion === "none" ? "BASE (none)" : activeAdapterVersion;
    const footerAdp = document.getElementById("statusAdapterVal");
    if (footerAdp) footerAdp.textContent = label;
    const s4Adp = document.getElementById("s4AdapterVersion");
    if (s4Adp) s4Adp.textContent = `Active Adapter: ${label}`;
  }
}

// UI Updates (Screen 2 & Screen 6 Status Footer)
function updateUI() {
  if (!tacticalState) return;

  if (tacticalState.audit_status) {
    const auditFooter = document.getElementById("statusAuditValidity");
    if (auditFooter) {
      auditFooter.textContent = tacticalState.audit_status.valid
        ? `SHA-256 CHAIN INTACT (${tacticalState.audit_status.entries} ENTRIES)`
        : "CHAIN COMPROMISED";
      auditFooter.style.color = tacticalState.audit_status.valid ? "var(--accent-emerald)" : "var(--status-blocked)";
    }
  }

  // Sync Screen 6 Status Footer with live drift_distance & feedback_count from /api/state
  updateDriftFooterUI(tacticalState.drift_distance, tacticalState.feedback_count);

  const allBldFeats = (tacticalState.buildings_geojson && tacticalState.buildings_geojson.features) || [];
  const satBldCount = allBldFeats.filter(f => f.properties && f.properties.data_source === "resnet18_unet_pretrained").length;
  const bldToggleLabel = document.getElementById("lblToggleBuildingsText");
  if (bldToggleLabel) {
    bldToggleLabel.textContent = satBldCount > 0 ? `Buildings (${satBldCount} SAT)` : `Buildings (${allBldFeats.length})`;
  }

  let openCnt = tacticalState.road_counts.OPEN;
  let hazCnt = tacticalState.road_counts.HAZARDOUS;
  let blkCnt = tacticalState.road_counts.BLOCKED;
  const activePlanObj = tacticalState.active_plan || (tacticalState.active_routes && tacticalState.active_routes[0]) || null;
  let totalEta = activePlanObj ? (activePlanObj.total_eta_sec || activePlanObj.eta_seconds || 0) : 0;
  const baseEta = totalEta;

  const routeStatusStr = String((activePlanObj && (activePlanObj.status || activePlanObj.approval_status)) || "pending_approval").toUpperCase();
  const tabStatusBadge = document.getElementById("tabRouteStatusBadge");
  const approveBtn = document.getElementById("approveBtn");
  const rejectBtnTab = document.getElementById("rejectBtnTab");
  if (tabStatusBadge) {
    tabStatusBadge.textContent = routeStatusStr;
    if (routeStatusStr === "APPROVED") {
      tabStatusBadge.style.background = "rgba(16,185,129,0.15)";
      tabStatusBadge.style.color = "#34d399";
    } else if (routeStatusStr === "REJECTED") {
      tabStatusBadge.style.background = "rgba(239,68,68,0.15)";
      tabStatusBadge.style.color = "#f87171";
    } else {
      tabStatusBadge.style.background = "rgba(245,158,11,0.15)";
      tabStatusBadge.style.color = "#fbbf24";
    }
  }
  if (approveBtn) {
    if (routeStatusStr === "APPROVED") {
      const shortId = activePlanObj && activePlanObj.route_id ? escapeHtml(activePlanObj.route_id.slice(0, 14)) : "PLAN";
      approveBtn.disabled = true;
      approveBtn.className = "btn btn-approve approved";
      approveBtn.innerHTML = `✔ ${shortId} APPROVED`;
    } else if (routeStatusStr === "REJECTED") {
      approveBtn.disabled = true;
      approveBtn.className = "btn btn-approve";
      approveBtn.style.opacity = "0.5";
      approveBtn.innerHTML = `<span>GATE LOCKED (REJECTED)</span>`;
    } else {
      approveBtn.disabled = false;
      approveBtn.style.opacity = "1";
      approveBtn.className = "btn btn-approve";
      approveBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg><span>APPROVE &amp; DISPATCH</span>`;
    }
  }
  if (rejectBtnTab) {
    if (routeStatusStr === "PENDING_APPROVAL") {
      rejectBtnTab.disabled = false;
      rejectBtnTab.style.opacity = "1";
      rejectBtnTab.textContent = "REJECT";
    } else {
      rejectBtnTab.disabled = true;
      rejectBtnTab.style.opacity = "0.5";
      rejectBtnTab.textContent = routeStatusStr === "REJECTED" ? "✖ REJECTED" : "LOCKED";
    }
  }

  if (whatIfPlan) {
    openCnt = 0; hazCnt = 0; blkCnt = 0;
    const dynamicHazards = buildWhatIfHazardMap(whatIfPlan);
    tacticalState.roads_geojson.features.forEach(f => {
      const eid = f.properties.edge_id;
      const st = dynamicHazards[eid] ? dynamicHazards[eid].status : f.properties.edge_status;
      if (st === "OPEN") openCnt++;
      else if (st === "HAZARDOUS") hazCnt++;
      else blkCnt++;
    });
    totalEta = (typeof whatIfPlan.whatif_eta_sec === "number")
      ? whatIfPlan.whatif_eta_sec
      : whatIfPlan.total_eta_sec;
  }

  document.getElementById("kpiOpen").textContent = openCnt;
  document.getElementById("kpiHazard").textContent = hazCnt;
  document.getElementById("kpiBlocked").textContent = blkCnt;
  document.getElementById("kpiEta").textContent = `${Math.round(totalEta)}s`;

  const deltaBadge = document.getElementById("etaDeltaBadge");
  const computeTag = (whatIfPlan && typeof whatIfPlan.compute_ms === "number")
    ? ` • ${Math.round(whatIfPlan.compute_ms)}ms`
    : "";
  if (whatIfPlan && Math.abs(totalEta - baseEta) > 1) {
    const diff = Math.round(totalEta - baseEta);
    const sign = diff >= 0 ? "+" : "";
    const pctText = baseEta > 0 ? ` (${((diff / baseEta) * 100).toFixed(1)}%)` : "";
    deltaBadge.className = "eta-delta-badge detour";
    deltaBadge.textContent = `WHAT-IF DETOUR: ${sign}${diff}s${pctText}${computeTag}`;
  } else if (whatIfPlan) {
    deltaBadge.className = "eta-delta-badge baseline";
    deltaBadge.textContent = `WHAT-IF ACTIVE: NO DETOUR${computeTag}`;
  } else {
    deltaBadge.className = "eta-delta-badge baseline";
    deltaBadge.textContent = "BASELINE PLAN (<500ms REROUTE)";
  }

  const activeNarrative = (whatIfPlan && whatIfPlan.narrative_text)
    ? whatIfPlan.narrative_text
    : (activePlanObj ? (activePlanObj.narrative_text || activePlanObj.explanation) : "No route plan generated.");
  document.getElementById("narrativeDeck").textContent = activeNarrative;

  const unitCardsWrap = document.getElementById("unitRouteCards");
  const activePlan = whatIfPlan || activePlanObj;
  if (activePlan && activePlan.routes) {
    unitCardsWrap.innerHTML = activePlan.routes.map(u => {
      const cfg = UNIT_CONFIG[u.unit_id] || { hex: "#ffffff", label: u.unit_id };
      const etaCell = u.reachable === false
        ? `<span style="color:#f87171; font-weight:700;">UNREACHABLE</span>`
        : `<span>ETA: <strong>${Math.round(u.eta_sec)}s</strong></span>`;
      return `
        <div class="unit-route-card">
          <span style="color:${cfg.hex}; font-weight:700;">${escapeHtml(cfg.label)}</span>
          <span>Target: <strong>N${escapeHtml(u.target_node)}</strong></span>
          ${etaCell}
          <span style="color:var(--text-dim);">${u.edge_ids.length} edges</span>
        </div>
      `;
    }).join("");
  }

  // Refresh Screen 4 Building Selector options from live buildings_geojson cleanly
  const bldSelect = document.getElementById("s4BuildingSelector");
  if (bldSelect && tacticalState.buildings_geojson) {
    const currentVal = bldSelect.value || currentInspectedBuildingId || "BLD-0042";
    bldSelect.innerHTML = tacticalState.buildings_geojson.features.map(f => {
      const p = f.properties;
      return `<option value="${escapeHtml(p.building_id)}" ${p.building_id === currentVal ? "selected" : ""}>${escapeHtml(p.building_id)} (${escapeHtml((p.class_label || "").toUpperCase())} - ${Math.round((p.confidence || 0.9) * 100)}%)</option>`;
    }).join("");
  }

  updateClosuresUI();
}

function updateClosuresUI() {
  const wrap = document.getElementById("closureTagsWrap");
  if (!wrap) return;
  if (selectedBlockedEdges.size === 0 && selectedOpenedEdges.size === 0) {
    wrap.innerHTML = `<span style="color:var(--text-dim); font-size:10.5px;">No manual overrides active. Click any road edge on the map to close an open corridor or reopen a blocked one.</span>`;
  } else {
    const blockedTags = Array.from(selectedBlockedEdges).map(eid => {
      const safeId = escapeHtml(eid);
      return `<span class="closure-tag" role="button" tabindex="0" data-kind="block" data-eid="${safeId}">✕ BLOCK ${safeId}</span>`;
    });
    const openedTags = Array.from(selectedOpenedEdges).map(eid => {
      const safeId = escapeHtml(eid);
      return `<span class="closure-tag" style="border-color:rgba(16,185,129,0.55); color:#34d399; background:rgba(16,185,129,0.12);" role="button" tabindex="0" data-kind="open" data-eid="${safeId}">✕ OPEN ${safeId}</span>`;
    });
    wrap.innerHTML = [...blockedTags, ...openedTags].join(" ");
    wrap.querySelectorAll(".closure-tag").forEach(tag => {
      tag.addEventListener("click", () => window.toggleRoadClosure(tag.getAttribute("data-eid")));
    });
  }
}

window.removeClosure = function (eid) {
  if (!selectedBlockedEdges.has(eid) && !selectedOpenedEdges.has(eid)) return;
  window.toggleRoadClosure(eid);
};

function updateSurgeGauge(val) {
  const pct = Math.max(0, Math.min(100, ((val + 0.10) / 0.85) * 100));
  const fill = document.getElementById("surgeFill");
  const text = document.getElementById("surgeGaugeText");
  const badge = document.getElementById("floodValBadge");
  const hudVal = document.getElementById("hudSurgeVal");

  fill.style.width = `${pct}%`;

  let stage = "NORMAL POOL";
  let color = "var(--accent-cyan)";
  if (val >= 0.45) {
    stage = "CRITICAL SURGE";
    color = "var(--status-blocked)";
  } else if (val >= 0.20) {
    stage = "ELEVATED STAGE";
    color = "var(--status-hazard)";
  }

  text.textContent = `BAYOU SURGE GAUGE: ${val >= 0 ? "+" : ""}${val.toFixed(2)}m (${stage})`;
  badge.textContent = `${val >= 0 ? "+" : ""}${val.toFixed(2)} m (${stage})`;
  badge.style.color = color;
  hudVal.textContent = `${val >= 0 ? "+" : ""}${val.toFixed(2)} m`;

  if (tacticalMap) {
    tacticalMap.updateWaterLevel(val);
  }
}

window.setSurgePreset = function (val) {
  currentFloodDelta = val;
  document.getElementById("floodSlider").value = val;
  updateSurgeGauge(val);
  if (Math.abs(val) < 0.01 && selectedBlockedEdges.size === 0 && selectedOpenedEdges.size === 0) {
    whatIfPlan = null;
    fetchState();
  } else {
    submitWhatIf();
  }
};

function renderAuditTable(entries, valid, count) {
  const body = document.getElementById("auditTableBody");
  document.getElementById("auditEntriesCount").textContent = `${count} ENTRIES`;
  if (!entries || entries.length === 0) {
    body.innerHTML = `<tr><td colspan="4" style="text-align:center;">No audit records</td></tr>`;
    return;
  }
  body.innerHTML = entries.slice(0, 12).map(e => {
    return `
      <tr>
        <td>${escapeHtml(e.seq_id)}</td>
        <td><strong>${escapeHtml(e.event_type)}</strong></td>
        <td>${escapeHtml(e.actor_role)}</td>
        <td class="hash-token" title="${escapeHtml(e.curr_hash)}">${escapeHtml(e.curr_hash)}</td>
      </tr>
    `;
  }).join("");
}

function renderGradCamRegionRows(regions) {
  if (!Array.isArray(regions) || regions.length === 0) {
    return '<div style="color:var(--text-dim);">No Grad-CAM spatial regions recorded.</div>';
  }
  return regions.map((regionItem, idx) => {
    const w = Number(regionItem.weight ?? 0);
    const pct = Math.min(100, Math.round(Math.abs(w) * 100));
    return `
      <div class="shap-meter-row">
        <div class="shap-meter-header">
          <span>Grad-CAM Region #${idx + 1} (Decoder Saliency)</span>
          <span style="color:var(--accent-cyan); font-weight:700;">${w.toFixed(2)} (${pct}%)</span>
        </div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill" style="width:${pct}%;"></div>
        </div>
      </div>
    `;
  }).join("");
}

async function renderInspectorBuilding(b) {
  document.getElementById("inspectorTypeTag").textContent = "BUILDING";
  const props = b.properties;
  const card = document.getElementById("inspectorCardBody");
  const isLowConf = props.status_flag === "LOW_CONFIDENCE";
  currentInspectedBuildingId = props.building_id;
  const conf = Number(props.confidence ?? props.max_prob ?? 0.9);
  const sysUncertainty = Math.max(0, 1.0 - conf);
  const safeBldId = escapeHtml(props.building_id);

  card.innerHTML = `
    <div style="font-size:13px; font-weight:800; color:var(--accent-cyan); margin-bottom:4px;">BUILDING: ${safeBldId}</div>
    <div class="prop-row">
      <span class="prop-key">Damage Assessment:</span>
      <span class="prop-val" style="color:${getDamageTier(props.damage_class) === 3 ? 'var(--status-blocked)' : (getDamageTier(props.damage_class) >= 1 ? 'var(--status-hazard)' : 'var(--status-open)')}">
        ${escapeHtml(normalizeDamageLabel(props.damage_class))} (tier ${getDamageTier(props.damage_class)})
      </span>
    </div>
    <div class="prop-row">
      <span class="prop-key">Calibrated Confidence:</span>
      <span class="prop-val">${(conf * 100).toFixed(1)}% ${isLowConf ? '<strong style="color:var(--accent-purple);">[LOW_CONF: PURPLE RING]</strong>' : '(' + escapeHtml(props.status_flag) + ')'}</span>
    </div>
    <div class="prop-row">
      <span class="prop-key">System Uncertainty:</span>
      <span class="prop-val" style="color:var(--accent-amber);">±${sysUncertainty.toFixed(3)}</span>
    </div>
    <div class="prop-row">
      <span class="prop-key">Embedding Ref:</span>
      <span class="prop-val" style="color:var(--accent-purple); font-size:10px;">${escapeHtml(props.embedding_ref || "pending")}</span>
    </div>
    <div class="prop-row">
      <span class="prop-key">Data Source:</span>
      <span class="prop-val" style="color:var(--text-muted);">${escapeHtml(props.data_source)}</span>
    </div>

    <div style="margin-top:8px; font-size:10.5px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">
      Grad-CAM Spatial Attributions (Portion 3):
    </div>
    <div class="shap-meter-group" id="inspectorGradCamGroup">
      <div style="color:var(--text-dim);">Loading Grad-CAM regions via /api/buildings/${safeBldId}...</div>
    </div>
    <button type="button" class="btn btn-sim" id="btnInspectorScreen4" style="margin-top:10px; width:100%; font-size:11px;">
      INSPECT GRAD-CAM &amp; LABEL (SCREEN 4)
    </button>
  `;
  document.getElementById("btnInspectorScreen4")?.addEventListener("click", () => {
    window.openBuildingInScreen4(props.building_id);
  });

  try {
    const apiBase = getApiBase();
    const res = await fetch(`${apiBase}/api/buildings/${encodeURIComponent(props.building_id)}`);
    if (!res.ok) return;
    const detail = await res.json();
    const regions = (detail.building && detail.building.gradcam_attribution) || [];
    const groupEl = document.getElementById("inspectorGradCamGroup");
    if (groupEl && currentInspectedBuildingId === props.building_id) {
      groupEl.innerHTML = renderGradCamRegionRows(regions);
    }
  } catch (err) {
    console.warn("Building Grad-CAM inspector fetch error:", err.message);
  }
}

function getPersistedRoadStatus(eid) {
  const feats = (tacticalState && tacticalState.roads_geojson && tacticalState.roads_geojson.features) || [];
  const feat = feats.find(f => f.properties && f.properties.edge_id === eid);
  return feat && feat.properties ? String(feat.properties.edge_status || "OPEN").toUpperCase() : "OPEN";
}

function renderInspectorRoad(r) {
  document.getElementById("inspectorTypeTag").textContent = "ROAD";
  const props = r.properties;
  const isOpenedOverride = selectedOpenedEdges.has(props.edge_id);
  const isBlocked = selectedBlockedEdges.has(props.edge_id) || (props.edge_status === "BLOCKED" && !isOpenedOverride);
  const card = document.getElementById("inspectorCardBody");
  const safeEid = escapeHtml(props.edge_id);
  const statusColor = isOpenedOverride
    ? "var(--status-open)"
    : (isBlocked ? "var(--status-blocked)" : (props.edge_status === "HAZARDOUS" ? "var(--status-hazard)" : "var(--status-open)"));
  const statusLabel = isOpenedOverride
    ? "OPEN (MANUAL REOPEN OVERRIDE)"
    : (isBlocked ? "BLOCKED (MANUAL DETOUR)" : escapeHtml(props.edge_status));

  card.innerHTML = `
    <div style="font-size:13px; font-weight:800; color:var(--accent-cyan); margin-bottom:4px;">PRIMARY ARTERIAL: ${safeEid}</div>
    <div class="prop-row">
      <span class="prop-key">Network Topology:</span>
      <span class="prop-val">Node N${escapeHtml(props.u_node)} ➔ Node N${escapeHtml(props.v_node)}</span>
    </div>
    <div class="prop-row">
      <span class="prop-key">Display State:</span>
      <span class="prop-val" style="color:${statusColor}">
        ${statusLabel}
      </span>
    </div>
    <div class="prop-row">
      <span class="prop-key">Sec. 4.2 Attributes:</span>
      <span class="prop-val" style="font-size:10px;">closure=${props.closure_state ? "1" : "0"} | cap=${Number(props.capacity ?? 1).toFixed(1)} | ${escapeHtml(props.vehicle_restriction || "none")}</span>
    </div>
    <div class="prop-row">
      <span class="prop-key">Flood Depth:</span>
      <span class="prop-val">${Number(props.flood_depth_m || 0).toFixed(2)} m</span>
    </div>
    <div class="prop-row">
      <span class="prop-key">Debris Hazard Score:</span>
      <span class="prop-val">${Number(props.debris_score || 0).toFixed(2)} / 1.0</span>
    </div>
    <div class="prop-row">
      <span class="prop-key">Road Length:</span>
      <span class="prop-val">${escapeHtml(props.length_m)} meters</span>
    </div>

    <button type="button" class="btn btn-secondary" id="btnInspectorToggleRoad" style="margin-top:10px; width:100%; font-size:11px;">
      ${isBlocked ? 'OPEN ROAD (ALLOW FLEET TRAVERSAL)' : 'BLOCK ROAD (FORCE WHAT-IF DETOUR)'}
    </button>
  `;
  document.getElementById("btnInspectorToggleRoad")?.addEventListener("click", () => {
    window.toggleRoadClosure(props.edge_id);
  });
}

window.toggleRoadClosure = function (eid) {
  const prevState = {
    blocked: new Set(selectedBlockedEdges),
    opened: new Set(selectedOpenedEdges),
  };
  const dbStatus = getPersistedRoadStatus(eid);
  if (dbStatus === "BLOCKED") {
    if (selectedOpenedEdges.has(eid)) {
      selectedOpenedEdges.delete(eid);
    } else {
      selectedOpenedEdges.add(eid);
      selectedBlockedEdges.delete(eid);
    }
  } else {
    if (selectedBlockedEdges.has(eid)) {
      selectedBlockedEdges.delete(eid);
    } else {
      selectedBlockedEdges.add(eid);
      selectedOpenedEdges.delete(eid);
    }
  }
  updateClosuresUI();
  if (selectedEntity && selectedEntity.type === "road" && selectedEntity.data && selectedEntity.data.properties && selectedEntity.data.properties.edge_id === eid) {
    renderInspectorRoad(selectedEntity.data);
  }
  if (selectedBlockedEdges.size === 0 && selectedOpenedEdges.size === 0 && Math.abs(currentFloodDelta) < 0.01) {
    whatIfPlan = null;
    fetchState();
  } else {
    submitWhatIf(false, prevState);
  }
};

function toggleFloodSimulation() {
  isFloodSimRunning = !isFloodSimRunning;
  const btn = document.getElementById("simToggleBtn");
  if (isFloodSimRunning) {
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>
      <span>PAUSE FLOOD SURGE SIMULATION</span>
    `;
    btn.classList.add("running");
    stepFloodSimulation();
  } else {
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
      <span>AUTO-SIMULATE RISING FLOOD SURGE</span>
    `;
    btn.classList.remove("running");
    if (floodSimTimer) clearTimeout(floodSimTimer);
  }
}

function stepFloodSimulation() {
  if (!isFloodSimRunning) return;
  let nextVal = Math.round((currentFloodDelta + (0.05 * floodSimDirection)) * 100) / 100;
  if (nextVal >= 0.70) {
    nextVal = 0.70;
    floodSimDirection = -1;
  } else if (nextVal <= 0.0) {
    nextVal = 0.0;
    floodSimDirection = 1;
  }
  currentFloodDelta = nextVal;
  document.getElementById("floodSlider").value = nextVal;
  updateSurgeGauge(nextVal);
  submitWhatIf(true);
  floodSimTimer = setTimeout(stepFloodSimulation, 480);
}

function switchTab(tabId) {
  document.querySelectorAll(".dock-tab").forEach(tab => {
    tab.classList.toggle("active", tab.getAttribute("data-tab") === tabId);
  });
  document.querySelectorAll(".tab-deck .shell-card").forEach(card => {
    card.style.display = (card.id === tabId) ? "flex" : "none";
  });
}

document.querySelectorAll(".dock-tab").forEach(tab => {
  tab.addEventListener("click", () => switchTab(tab.getAttribute("data-tab")));
});

document.getElementById("btnOpenGuide").addEventListener("click", () => {
  if (currentScreen !== "screen-map") {
    switchScreen("screen-map");
  }
  switchTab("tab-guide");
});

// Map Viewport Presets & Workspace Extend Toggle
let activeViewportPreset = "iso";
let isMapExtended = false;

function applyViewportPreset(mode, btnId) {
  activeViewportPreset = mode;
  if (tacticalMap) {
    const rsc = sectorBuilder && sectorBuilder.fleetUnits["RSC-01"];
    tacticalMap.setViewportPreset(mode, rsc ? rsc.latLng : null);
  }
  setActiveCamBtn(btnId);
}

function setMapExtended(extended) {
  isMapExtended = Boolean(extended);
  const workspace = document.getElementById("mapWorkspace");
  const btn = document.getElementById("btnExtendMap");
  const label = document.getElementById("extendMapLabel");
  const icon = document.getElementById("extendMapIcon");

  if (workspace) workspace.classList.toggle("map-extended", isMapExtended);
  if (btn) {
    btn.classList.toggle("is-extended", isMapExtended);
    btn.setAttribute("aria-pressed", isMapExtended ? "true" : "false");
    btn.title = isMapExtended ? "Restore Side Command Panel [Esc / F]" : "Extend Map Full Workspace [F]";
  }
  if (label) label.textContent = isMapExtended ? "COLLAPSE MAP" : "EXTEND MAP";
  if (icon) {
    icon.innerHTML = isMapExtended
      ? '<polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/>'
      : '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>';
  }

  if (tacticalMap && tacticalMap.map) {
    const syncMapViewport = () => {
      tacticalMap.map.invalidateSize({ animate: false });
      if (activeViewportPreset === "iso" || activeViewportPreset === "top") {
        tacticalMap.setViewportPreset(activeViewportPreset);
      }
    };
    requestAnimationFrame(syncMapViewport);
    setTimeout(syncMapViewport, 80);
  }
}

document.getElementById("btnCamIso").addEventListener("click", () => applyViewportPreset("iso", "btnCamIso"));
document.getElementById("btnCamTop").addEventListener("click", () => applyViewportPreset("top", "btnCamTop"));
document.getElementById("btnCamBridge").addEventListener("click", () => applyViewportPreset("bridge", "btnCamBridge"));
document.getElementById("btnCamDrone").addEventListener("click", () => applyViewportPreset("drone", "btnCamDrone"));
document.getElementById("btnCamReset").addEventListener("click", () => applyViewportPreset("iso", "btnCamIso"));
document.getElementById("btnExtendMap")?.addEventListener("click", () => setMapExtended(!isMapExtended));

function setActiveCamBtn(btnId) {
  document.querySelectorAll(".cam-btn:not(.cam-btn-extend)").forEach(b => b.classList.toggle("active", b.id === btnId));
}

// Layer Controls
[
  ["toggleSatellite", "lblToggleSatellite", "satellite"],
  ["toggleWater", "lblToggleWater", "water"],
  ["toggleBuildings", "lblToggleBuildings", "buildings"],
  ["toggleLowConf", "lblToggleLowConf", "lowConf"],
  ["toggleSecondaryRoads", "lblToggleSecondaryRoads", "secondaryRoads"],
  ["toggleRoutes", "lblToggleRoutes", "routes"],
  ["toggleLabels", "lblToggleLabels", "labels"],
  ["toggleIncidents", "lblToggleIncidents", "incidents"],
].forEach(([inputId, labelId, layerKey]) => {
  const input = document.getElementById(inputId);
  if (!input) return;
  input.addEventListener("change", (e) => {
    const checked = e.target.checked;
    if (tacticalMap) {
      tacticalMap.setLayerVisibility(layerKey, checked);
      if (layerKey === "routes") {
        tacticalMap.setLayerVisibility("fleet", checked);
      }
    }
    document.getElementById(labelId)?.classList.toggle("checked", checked);
  });
});

// Slider Event
document.getElementById("floodSlider").addEventListener("input", (e) => {
  currentFloodDelta = parseFloat(e.target.value);
  updateSurgeGauge(currentFloodDelta);
  if (Math.abs(currentFloodDelta) < 0.01 && selectedBlockedEdges.size === 0 && selectedOpenedEdges.size === 0) {
    whatIfPlan = null;
    fetchState();
  } else {
    submitWhatIf();
  }
});

document.getElementById("simToggleBtn").addEventListener("click", toggleFloodSimulation);
document.getElementById("resetBtn").addEventListener("click", resetBaseline);
document.getElementById("approveBtn").addEventListener("click", approvePlan);
document.getElementById("btnRecomputeDispatch").addEventListener("click", () => recomputeDispatch());
document.getElementById("btnSimulateSatBlockage").addEventListener("click", simulateLiveSatelliteBlockage);
document.getElementById("btnLoadDatasetSample")?.addEventListener("click", () => loadDatasetSampleTiles());

// Keyboard Shortcuts (scoped to active map view & unmodified keypresses)
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    const p4Modal = document.getElementById("portion4MonitorModal");
    if (p4Modal && p4Modal.classList.contains("active")) {
      closePortion4MonitorModal();
      return;
    }
    const tourModal = document.getElementById("projectTourModal");
    if (tourModal && tourModal.classList.contains("active")) {
      closeProjectTour();
      return;
    }
    const modal = document.getElementById("screenApprovalModal");
    if (modal && modal.classList.contains("active")) {
      closeApprovalModal();
      return;
    }
    if (isMapExtended) {
      setMapExtended(false);
    }
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const tag = e.target && e.target.tagName ? e.target.tagName.toUpperCase() : "";
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON", "A"].includes(tag) || (e.target && e.target.isContentEditable)) {
    return;
  }
  const modalOpen =
    document.getElementById("screenApprovalModal")?.classList.contains("active") ||
    document.getElementById("projectTourModal")?.classList.contains("active") ||
    document.getElementById("portion4MonitorModal")?.classList.contains("active");
  if (currentScreen !== "screen-map" || modalOpen) return;

  if (e.code === "Space") {
    e.preventDefault();
    toggleFloodSimulation();
  } else if (e.key === "1") {
    applyViewportPreset("iso", "btnCamIso");
  } else if (e.key === "2") {
    applyViewportPreset("top", "btnCamTop");
  } else if (e.key === "3") {
    applyViewportPreset("bridge", "btnCamBridge");
  } else if (e.key === "4") {
    applyViewportPreset("drone", "btnCamDrone");
  } else if (e.key.toLowerCase() === "f") {
    setMapExtended(!isMapExtended);
  } else if (e.key.toLowerCase() === "r") {
    resetBaseline();
  } else if (e.key.toLowerCase() === "a") {
    approvePlan();
  }
});

function updateClock() {
  const now = new Date();
  document.getElementById("clockDisplay").textContent = `UTC: ${now.toISOString().slice(11, 19)}`;
}
setInterval(updateClock, 1000);
updateClock();

let lastFrameTs = null;
function animLoop(timestamp) {
  const dtSec = lastFrameTs ? Math.min(0.05, (timestamp - lastFrameTs) / 1000) : 0.016;
  lastFrameTs = timestamp;

  if (sectorBuilder && tacticalState) {
    const activePlan = whatIfPlan || tacticalState.active_plan;
    sectorBuilder.updateFleetMovement(dtSec, activePlan);
  }

  requestAnimationFrame(animLoop);
}


// =========================================================================
// 6-SCREEN WORKFLOW CONTROLLER (workflow/ahmed.pdf — Live Endpoint Wiring)
// =========================================================================
let currentScreen = "screen-map";
let currentInspectedBuildingId = "BLD-0042";

window.openBuildingInScreen4 = function (buildingId) {
  currentInspectedBuildingId = buildingId;
  switchScreen("screen-building");
};

function switchScreen(screenId) {
  currentScreen = screenId;
  document.querySelectorAll(".screen-btn").forEach(b => b.classList.remove("active"));

  const uploadView = document.getElementById("screenUploadView");
  const routeView = document.getElementById("screenRouteView");
  const buildingView = document.getElementById("screenBuildingView");
  const mapWorkspace = document.getElementById("mapWorkspace");
  const mapSubBar = document.getElementById("mapSubBar");
  const mapShell = document.getElementById("mapShell");
  const commandDock = document.querySelector(".command-dock");

  // Hide all full-screen overlays first
  if (uploadView) uploadView.classList.remove("active");
  if (routeView) routeView.classList.remove("active");
  if (buildingView) buildingView.classList.remove("active");

  if (screenId === "screen-upload") {
    document.getElementById("navScreen1")?.classList.add("active");
    if (mapWorkspace) mapWorkspace.style.display = "none";
    if (mapSubBar) mapSubBar.style.display = "none";
    if (mapShell) mapShell.style.display = "none";
    if (commandDock) commandDock.style.display = "none";
    if (uploadView) uploadView.classList.add("active");
  } else if (screenId === "screen-map") {
    document.getElementById("navScreen2")?.classList.add("active");
    if (mapWorkspace) mapWorkspace.style.display = "flex";
    if (mapSubBar) mapSubBar.style.display = "flex";
    if (mapShell) mapShell.style.display = "flex";
    if (commandDock) commandDock.style.display = "flex";
    if (tacticalMap && tacticalMap.map) {
      setTimeout(() => {
        tacticalMap.map.invalidateSize(false);
        if (tacticalState) {
          tacticalMap.renderState(tacticalState, whatIfPlan);
        }
      }, 30);
    }
  } else if (screenId === "screen-route") {
    document.getElementById("navScreen3")?.classList.add("active");
    if (mapWorkspace) mapWorkspace.style.display = "none";
    if (mapSubBar) mapSubBar.style.display = "none";
    if (mapShell) mapShell.style.display = "none";
    if (commandDock) commandDock.style.display = "none";
    if (routeView) routeView.classList.add("active");
    populateScreenRouteData();
  } else if (screenId === "screen-building") {
    document.getElementById("navScreen4")?.classList.add("active");
    if (mapWorkspace) mapWorkspace.style.display = "none";
    if (mapSubBar) mapSubBar.style.display = "none";
    if (mapShell) mapShell.style.display = "none";
    if (commandDock) commandDock.style.display = "none";
    if (buildingView) buildingView.classList.add("active");
    populateScreenBuildingData(currentInspectedBuildingId || "BLD-0042");
  }
}

const MAX_UPLOAD_TILE_BYTES = 25 * 1024 * 1024;

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

function applySelectedTileFile(side, file) {
  if (!file) return;
  if (file.size > MAX_UPLOAD_TILE_BYTES) {
    showToast(`Tile ${file.name} exceeds 25 MB upload limit.`, true);
    return;
  }
  if (side === "pre") selectedPreTileName = file.name;
  else selectedPostTileName = file.name;

  const preview = document.getElementById(side === "pre" ? "preTilePreview" : "postTilePreview");
  const sizeKb = Math.max(1, Math.round(file.size / 1024));
  if (preview) {
    preview.innerHTML = `<span style="color:var(--accent-cyan);">✓ Loaded ${side === "pre" ? "T0 Pre" : "T1 Post"} Tile: <strong>${escapeHtml(file.name)}</strong> (${sizeKb} KB)</span>`;
  }

  const readPromise = readFileAsDataUrl(file)
    .then((dataUrl) => {
      if (side === "pre") selectedPreTileDataUrl = dataUrl;
      else selectedPostTileDataUrl = dataUrl;
    })
    .catch((err) => {
      if (side === "pre") selectedPreTileDataUrl = null;
      else selectedPostTileDataUrl = null;
      showToast(err.message, true);
    });
  pendingTileReads[side] = readPromise;
}

function handleTileUpload(side) {
  const input = document.getElementById(side === "pre" ? "preImgInput" : "postImgInput");
  const file = input && input.files && input.files[0];
  applySelectedTileFile(side, file);
}

async function loadDatasetSampleTiles() {
  const btn = document.getElementById("btnLoadDatasetSample");
  const label = btn ? btn.querySelector("span") : null;
  if (btn) btn.disabled = true;
  if (label) label.textContent = "LOADING DATASET PAIR...";
  const apiBase = getApiBase();
  const preName = "hurricane-matthew_00000030_pre_disaster.png";
  const postName = "hurricane-matthew_00000030_post_disaster.png";
  try {
    const [preRes, postRes] = await Promise.all([
      fetch(`${apiBase}/static/samples/${preName}`),
      fetch(`${apiBase}/static/samples/${postName}`),
    ]);
    if (!preRes.ok || !postRes.ok) {
      throw new Error("Dataset sample tiles unavailable under /static/samples/");
    }
    const [preBlob, postBlob] = await Promise.all([preRes.blob(), postRes.blob()]);
    applySelectedTileFile("pre", new File([preBlob], preName, { type: "image/png" }));
    applySelectedTileFile("post", new File([postBlob], postName, { type: "image/png" }));
    await Promise.all([pendingTileReads.pre, pendingTileReads.post].filter(Boolean));
    showToast("Loaded real-world xBD sample pair (hurricane-matthew_00000030 T0 + T1). Ready to trigger assessment.", false);
  } catch (err) {
    showToast("Dataset sample load failed: " + err.message, true);
  } finally {
    if (btn) btn.disabled = false;
    if (label) label.textContent = "LOAD DATASET SAMPLE (T0 + T1)";
  }
}

function wireTileDropZones() {
  [
    ["preDropZone", "pre"],
    ["postDropZone", "post"],
  ].forEach(([zoneId, side]) => {
    const zone = document.getElementById(zoneId);
    if (!zone) return;
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.style.borderColor = "var(--accent-cyan)";
    });
    zone.addEventListener("dragleave", () => {
      zone.style.borderColor = "";
    });
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.style.borderColor = "";
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      applySelectedTileFile(side, file);
    });
  });
}

async function triggerDamageAssessment() {
  const btn = document.getElementById("btnTriggerAssessment");
  const progressWrap = document.getElementById("assessmentProgressWrap");
  const bar = document.getElementById("assessmentProgressBar");
  const stepText = document.getElementById("assessmentStepText");
  const pctText = document.getElementById("assessmentPctText");
  const resultsCard = document.getElementById("assessmentResultsCard");

  await Promise.all([pendingTileReads.pre, pendingTileReads.post].filter(Boolean));

  if ((selectedPreTileDataUrl && !selectedPostTileDataUrl) || (!selectedPreTileDataUrl && selectedPostTileDataUrl)) {
    showToast("Please select both Before (T0) and After (T1) satellite images before triggering inference.", true);
    return;
  }

  btn.disabled = true;
  progressWrap.style.display = "block";
  resultsCard.style.display = "none";
  bar.style.width = "25%";
  pctText.textContent = "RUNNING";
  stepText.textContent = selectedPreTileDataUrl
    ? "Uploading T0/T1 pair & running ResNet-18 U-Net + Grad-CAM..."
    : "Querying POST /api/assessment/upload & polling Portion 1 status...";

  const apiBase = getApiBase();
  try {
    // Call Endpoint 3: POST /api/assessment/upload (Portion 1 Ammar)
    const res = await fetch(`${apiBase}/api/assessment/upload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pre_disaster_image: selectedPreTileDataUrl,
        post_disaster_image: selectedPostTileDataUrl,
        pre_tile_ref: selectedPreTileName || "xbd_houston_east_pre.tif",
        post_tile_ref: selectedPostTileName || "xbd_houston_east_post.tif",
        sector_id: "houston-east",
      }),
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) detail = String(errJson.detail);
      } catch (_) {}
      throw new Error(detail);
    }
    const uploadData = await res.json();
    const taskId = uploadData.task_id;

    // Call Endpoint 4: GET /api/assessment/status/{task_id} (real backend progress & status)
    const statusRes = await fetch(`${apiBase}/api/assessment/status/${encodeURIComponent(taskId)}`);
    if (!statusRes.ok) {
      throw new Error(`Assessment status failed (HTTP ${statusRes.status})`);
    }
    const statusData = await statusRes.json();
    const realStatus = String(statusData.status || "pending").toLowerCase();
    const realPct = Math.max(0, Math.min(100, Number(statusData.progress_pct ?? 0)));

    bar.style.width = `${realPct}%`;
    pctText.textContent = `${realStatus.toUpperCase()} (${realPct.toFixed(0)}%)`;
    stepText.textContent = uploadData.message || `Task ${taskId} status: ${realStatus}`;

    const titleEl = document.getElementById("assessmentResultsTitle");
    const summaryEl = document.getElementById("assessmentResultsSummary");
    if (titleEl) {
      const isDone = realStatus === "completed";
      titleEl.textContent = isDone
        ? `✓ DAMAGE ASSESSMENT COMPLETED (${statusData.task_id})`
        : `ASSESSMENT QUEUED / ${realStatus.toUpperCase()} (${statusData.task_id})`;
      titleEl.style.color = isDone ? "var(--accent-emerald)" : "var(--accent-amber)";
      resultsCard.style.borderColor = isDone ? "var(--accent-emerald)" : "var(--accent-amber)";
    }
    if (summaryEl) {
      const baseSummary = `${statusData.buildings_detected} Detected/Indexed Buildings • ${statusData.low_confidence_count} Low-Confidence (< 0.65) • Backend task status: ${realStatus} (${realPct.toFixed(0)}%)`;
      summaryEl.textContent = uploadData.message ? `${baseSummary} — ${uploadData.message}` : baseSummary;
    }
    resultsCard.style.display = "block";
    if (tacticalMap) tacticalMap._activeSatelliteKey = null;
    await fetchState();
    await fetchAuditLog();
    showToast(
      realStatus === "completed"
        ? (uploadData.message || `Assessment ${statusData.task_id} completed (${statusData.buildings_detected} buildings).`)
        : `Assessment ${statusData.task_id} queued (${realStatus}): upload T0+T1 images to run inference.`
    );
  } catch (err) {
    bar.style.width = "0%";
    pctText.textContent = "ERROR";
    showToast("Assessment upload failed: " + err.message, true);
  } finally {
    btn.disabled = false;
  }
}

async function populateScreenRouteData() {
  const apiBase = getApiBase();
  const persistedPlan = tacticalState
    ? (tacticalState.active_plan || (tacticalState.active_routes && tacticalState.active_routes[0]))
    : null;
  const routeId = persistedPlan ? (persistedPlan.route_id || persistedPlan.plan_id || "PLAN-BASELINE-01") : "PLAN-BASELINE-01";

  document.getElementById("s3RouteHeader").textContent = "ROUTE PLAN: " + routeId;
  const approvalBadge = document.getElementById("s3ApprovalStatusBadge");
  const st = String((persistedPlan && (persistedPlan.status || persistedPlan.approval_status)) || "pending_approval").toUpperCase();
  if (approvalBadge) {
    approvalBadge.textContent = st;
    approvalBadge.style.color = st === "APPROVED" ? "#34d399" : (st === "REJECTED" ? "#f87171" : "#fbbf24");
  }
  const s3Approve = document.getElementById("s3ApproveBtn");
  const s3Reject = document.getElementById("s3RejectBtn");
  const isLocked = st === "APPROVED" || st === "REJECTED";
  if (s3Approve) {
    s3Approve.disabled = isLocked;
    s3Approve.style.opacity = isLocked ? "0.55" : "1";
  }
  if (s3Reject) {
    s3Reject.disabled = isLocked;
    s3Reject.style.opacity = isLocked ? "0.55" : "1";
  }

  try {
    // Call Endpoint 14: GET /api/explanations/route/{route_id} (Portion 3 Yousef + Portion 5 Validation Gate)
    const res = await fetch(`${apiBase}/api/explanations/route/${encodeURIComponent(routeId)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const expData = await res.json();

    document.getElementById("s3NarrativeText").textContent = expData.natural_language;
    const gateBadge = document.getElementById("s3ValidationBadge");
    if (gateBadge) {
      const passed = expData.validation_passed !== false;
      gateBadge.textContent = passed
        ? `GTF GATE: PASSED (${expData.llm_used ? "LLM" : "TEMPLATE"})`
        : "GTF GATE: FALLBACK TRIGGERED";
      gateBadge.style.color = passed ? "#34d399" : "#fbbf24";
    }
    const uncBadge = document.getElementById("s3UncertaintyBadge");
    if (uncBadge && typeof expData.system_uncertainty === "number") {
      uncBadge.textContent = `±${expData.system_uncertainty.toFixed(2)} ${expData.system_uncertainty < 0.15 ? "LOW UNCERTAINTY" : "ELEVATED UNCERTAINTY"}`;
    }
    const maskEl = document.getElementById("s3MaskAlteredText");
    if (maskEl) {
      maskEl.textContent = expData.mask_altered
        ? `YES (${expData.mask_reason})`
        : "NO (Primary Proposal Clear)";
    }

const SHAP_DESCRIPTIONS = {
  hazard_distance: "How strongly avoiding flooded or blocked roads shaped the route choice.",
  travel_time: "Importance of total corridor travel duration to reach the target quickly.",
  road_capacity: "Available lane width and throughput for emergency vehicles.",
  closure_state: "Hard safety exclusion for closed or impassable road edges.",
  vehicle_restriction: "Clearance limits (e.g., high-clearance only vs. standard ambulance).",
  incident_severity: "Urgency priority of the medical, rescue, or evacuation incident.",
  unit_capability: "Match between responder unit role (MED, TRN, RSC) and target need.",
};

    const shapGroup = document.getElementById("s3ShapMetersGroup");
    const shapList = expData.shap_values || [];
    shapGroup.innerHTML = shapList.map(s => {
      const val = Number(s.value ?? s.impact ?? s.attribution ?? 0);
      const pct = Math.min(100, Math.round(Math.abs(val) * 100));
      const key = String(s.feature || "").toLowerCase();
      const hint = SHAP_DESCRIPTIONS[key] || "Routing feature contribution weight.";
      return `
        <div style="display:flex; flex-direction:column; gap:4px;">
          <div style="display:flex; justify-content:space-between; font-size:11px; font-family:var(--font-mono);">
            <span style="color:var(--text-bright);">${escapeHtml(s.feature.toUpperCase())}</span>
            <span style="color:var(--accent-cyan); font-weight:700;">+${val.toFixed(2)} (${pct}%)</span>
          </div>
          <div style="height:6px; background:#0f172a; border-radius:3px; overflow:hidden;">
            <div style="height:100%; width:${pct}%; background:linear-gradient(90deg, #00e5ff, #38bdf8);"></div>
          </div>
          <div class="shap-feature-hint">${escapeHtml(hint)}</div>
        </div>
      `;
    }).join("");

    const cfWrap = document.getElementById("s3CounterfactualsList");
    const cfList = (persistedPlan && persistedPlan.counterfactuals) || [];
    if (cfWrap) {
      cfWrap.innerHTML = cfList.length > 0
        ? cfList.map(c => `<div style="display:flex; justify-content:space-between; padding:5px 8px; background:rgba(255,255,255,0.02); border-radius:4px;"><span>${escapeHtml(c.alternative)}</span><span style="color:${Number(c.delta_eta) < 0 ? '#f87171' : '#34d399'};">${Number(c.delta_eta) >= 0 ? "+" : ""}${Number(c.delta_eta).toFixed(1)}s</span></div>`).join("")
        : '<div style="color:var(--text-dim);">No counterfactual alternatives logged.</div>';
    }
  } catch (err) {
    console.warn("Route explanation fetch error:", err.message);
  }
}

async function populateScreenBuildingData(buildingId) {
  currentInspectedBuildingId = buildingId || "BLD-0042";
  const selector = document.getElementById("s4BuildingSelector");
  if (selector && selector.value !== currentInspectedBuildingId) {
    selector.value = currentInspectedBuildingId;
  }

  const apiBase = getApiBase();
  try {
    // Call Endpoint 6 (Portion 1 Ammar) and Endpoint 15 (Portion 3 Yousef)
    const [bldRes, expRes] = await Promise.all([
      fetch(`${apiBase}/api/buildings/${encodeURIComponent(currentInspectedBuildingId)}`),
      fetch(`${apiBase}/api/explanations/building/${encodeURIComponent(currentInspectedBuildingId)}`)
    ]);
    const bldDetail = bldRes.ok ? await bldRes.json() : null;
    const expDetail = expRes.ok ? await expRes.json() : null;
    const bld = bldDetail ? bldDetail.building : null;
    currentInspectedBuildingData = bld;

    if (bld) {
      document.getElementById("s4BuildingTitle").textContent = "BUILDING: " + bld.building_id;
      const lbl = normalizeDamageLabel(bld.damage_class || bld.class_label || "major");
      const badge = document.getElementById("s4DamageBadge");
      const DAMAGE_BADGE_TEXT = {
        no_damage: "NO DAMAGE",
        minor: "MINOR DAMAGE",
        major: "MAJOR DAMAGE",
        destroyed: "DESTROYED",
      };
      badge.textContent = DAMAGE_BADGE_TEXT[lbl] || (lbl.replace(/_/g, " ").toUpperCase() + " DAMAGE");
      const dmgStyle = DAMAGE_STYLE_MAP[lbl] || DAMAGE_STYLE_MAP.minor;
      badge.style.color = dmgStyle.color;
      const conf = Number(bld.confidence ?? 0.89);
      const confPct = Math.round(conf * 1000) / 10;
      document.getElementById("s4ConfidenceVal").textContent = `${confPct}% (${bld.status_flag || "CALIBRATED"})`;
      const confBar = document.getElementById("s4ConfidenceBar");
      confBar.style.width = confPct + "%";
      confBar.style.background = conf < 0.65 ? "var(--accent-purple)" : "var(--accent-cyan)";

      const uncBadge = document.getElementById("s4UncertaintyBadge");
      if (uncBadge) {
        const unc = expDetail && typeof expDetail.system_uncertainty === "number"
          ? expDetail.system_uncertainty
          : Math.max(0, 1.0 - conf);
        uncBadge.textContent = `±${unc.toFixed(3)} UNCERTAINTY`;
      }
      const embEl = document.getElementById("s4EmbeddingRef");
      if (embEl) {
        embEl.textContent = `Embedding Ref: ${bld.embedding_ref || "none"}`;
      }
      const adpEl = document.getElementById("s4AdapterVersion");
      if (adpEl && document.getElementById("statusAdapterVal")) {
        adpEl.textContent = `Active Adapter: ${document.getElementById("statusAdapterVal").textContent || "BASE (none)"}`;
      }

      const corrSelect = document.getElementById("s4CorrectionSelect");
      if (corrSelect) corrSelect.value = lbl;
      syncScreen4CorrectionHint();
    }
    if (expDetail && expDetail.explanation_text) {
      const expTextEl = document.getElementById("s4ExplanationText");
      if (expTextEl) expTextEl.textContent = expDetail.explanation_text;
    }

    const regions = (expDetail && expDetail.gradcam_regions) || (bld && bld.gradcam_attribution) || [];
    const overlayUrl = (expDetail && expDetail.gradcam_image_url) ? `${apiBase}${expDetail.gradcam_image_url}` : null;
    drawGradCamCanvas(
      "gradcamCanvas",
      bld ? normalizeDamageLabel(bld.damage_class) : "major",
      regions,
      bld ? bld.geometry : null,
      overlayUrl,
      currentInspectedBuildingId
    );
  } catch (err) {
    console.warn("Building detail fetch error:", err.message);
  }
}

const DAMAGE_LABEL_MAP = {
  0: "no_damage",
  1: "minor",
  2: "major",
  3: "destroyed",
  no_damage: "no_damage",
  "no-damage": "no_damage",
  minor: "minor",
  "minor-damage": "minor",
  major: "major",
  "major-damage": "major",
  destroyed: "destroyed",
};

const DAMAGE_TIER_MAP = {
  no_damage: 0,
  minor: 1,
  major: 2,
  destroyed: 3,
};

function normalizeDamageLabel(val) {
  if (val === null || val === undefined) return "minor";
  const key = String(val).trim().toLowerCase();
  return DAMAGE_LABEL_MAP[key] || "minor";
}

function getDamageTier(val) {
  return DAMAGE_TIER_MAP[normalizeDamageLabel(val)] ?? 1;
}

function drawGradCamCanvas(canvasId, damageClass, gradcamRegions = [], geometry = null, overlayUrl = null, expectedBuildingId = null) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const targetBldId = expectedBuildingId || currentInspectedBuildingId || "";
  canvas.dataset.buildingId = targetBldId;

  const drawVectorHeatmap = () => {
    ctx.fillStyle = "#0c1424";
    ctx.fillRect(0, 0, w, h);

    const ring = geometry && geometry.coordinates && geometry.coordinates[0];
    let projectFn = (lon, lat) => [w * 0.5, h * 0.5];
    if (Array.isArray(ring) && ring.length >= 3) {
      const lons = ring.map(p => p[0]);
      const lats = ring.map(p => p[1]);
      const minLon = Math.min(...lons), maxLon = Math.max(...lons);
      const minLat = Math.min(...lats), maxLat = Math.max(...lats);
      const spanLon = Math.max(1e-6, maxLon - minLon);
      const spanLat = Math.max(1e-6, maxLat - minLat);
      projectFn = (lon, lat) => [
        w * 0.22 + ((lon - minLon) / spanLon) * (w * 0.56),
        h * 0.78 - ((lat - minLat) / spanLat) * (h * 0.56),
      ];

      ctx.beginPath();
      ring.forEach((pt, idx) => {
        const [px, py] = projectFn(pt[0], pt[1]);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = "rgba(30, 41, 59, 0.7)";
      ctx.fill();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
      ctx.lineWidth = 2;
      ctx.strokeRect(w * 0.25, h * 0.2, w * 0.5, h * 0.6);
    }

    const hasGradCam = Array.isArray(gradcamRegions) && gradcamRegions.length > 0;
    if (hasGradCam) {
      gradcamRegions.forEach((reg) => {
        const weight = Math.max(0.1, Math.min(1.0, Number(reg.weight ?? 0.5)));
        let cx = w * 0.5;
        let cy = h * 0.5;
        const regRing = reg.region && reg.region.coordinates && reg.region.coordinates[0];
        if (Array.isArray(regRing) && regRing.length > 0 && ring) {
          const avgLon = regRing.reduce((s, p) => s + p[0], 0) / regRing.length;
          const avgLat = regRing.reduce((s, p) => s + p[1], 0) / regRing.length;
          [cx, cy] = projectFn(avgLon, avgLat);
        }
        const radius = w * (0.18 + weight * 0.16);
        const grad = ctx.createRadialGradient(cx, cy, 8, cx, cy, radius);
        grad.addColorStop(0, `rgba(239, 68, 68, ${Math.min(0.95, weight)})`);
        grad.addColorStop(0.35, "rgba(245, 158, 11, 0.68)");
        grad.addColorStop(0.7, "rgba(0, 229, 255, 0.35)");
        grad.addColorStop(1, "rgba(12, 20, 36, 0.0)");
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);
      });
    } else {
      ctx.fillStyle = "rgba(148, 163, 184, 0.78)";
      ctx.font = "11px monospace";
      ctx.textAlign = "center";
      ctx.fillText("No Grad-CAM spatial regions recorded", w * 0.5, h * 0.9);
    }

    ctx.strokeStyle = "rgba(0, 229, 255, 0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w * 0.5, 0); ctx.lineTo(w * 0.5, h);
    ctx.moveTo(0, h * 0.5); ctx.lineTo(w, h * 0.5);
    ctx.stroke();
  };

  drawVectorHeatmap();

  if (overlayUrl && String(overlayUrl).includes("/api/explanations/building/")) {
    const overlayImg = new Image();
    overlayImg.onload = () => {
      if (canvas.dataset.buildingId !== targetBldId) return;
      ctx.fillStyle = "#0c1424";
      ctx.fillRect(0, 0, w, h);
      const scale = Math.min(w / Math.max(1, overlayImg.width), h / Math.max(1, overlayImg.height));
      const drawW = overlayImg.width * scale;
      const drawH = overlayImg.height * scale;
      const dx = (w - drawW) * 0.5;
      const dy = (h - drawH) * 0.5;
      ctx.drawImage(overlayImg, dx, dy, drawW, drawH);
      ctx.strokeStyle = "rgba(0, 229, 255, 0.35)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(dx, dy, drawW, drawH);
    };
    overlayImg.onerror = () => {
      if (canvas.dataset.buildingId === targetBldId) drawVectorHeatmap();
    };
    overlayImg.src = overlayUrl;
  }
}

async function submitBuildingFeedback(action) {
  const targetBldId = currentInspectedBuildingId || "BLD-0042";
  const apiBase = getApiBase();
  const prediction = currentInspectedBuildingData
    ? normalizeDamageLabel(currentInspectedBuildingData.damage_class || currentInspectedBuildingData.class_label)
    : (document.getElementById("s4CorrectionSelect")?.value || "minor");
  const confVal = currentInspectedBuildingData
    ? Number(currentInspectedBuildingData.confidence ?? 0.5)
    : 0.5;
  const uncertainty = Number(Math.max(0, Math.min(1, 1.0 - confVal)).toFixed(4));

  try {
    // Call Endpoint 18: POST /api/feedback (Portion 4 Active Learning)
    const res = await fetch(`${apiBase}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_id: targetBldId,
        prediction,
        uncertainty,
        responder_action: action,
        corrected_label: action === "confirm" ? prediction : "",
        responder_id: "FIELD_RESPONDER_ALPHA",
        time_taken_sec: 1.4,
      }),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    updateDriftFooterUI(data.drift_distance, data.feedback_count);
    const secBanner = document.getElementById("s4SecondOpinionBanner");
    if (secBanner) {
      secBanner.style.display = data.requires_second_opinion ? "block" : "none";
    }
    const extraWarn = data.rate_limited_warning ? " | Responder hourly cap warning" : "";
    showToast(`1-tap ${action.toUpperCase()} queued for ${targetBldId}. Drift: ${data.drift_distance} | Queue: ${data.feedback_count}${extraWarn}`);
    await fetchState();
    await pollDriftAndFeedback();
  } catch (err) {
    showToast("Feedback submission error: " + err.message, true);
  }
}

async function submitFieldCorrection() {
  const select = document.getElementById("s4CorrectionSelect");
  const newClass = select.value;
  const targetBldId = currentInspectedBuildingId || "BLD-0042";
  const apiBase = getApiBase();

  try {
    // Call Endpoint 7: POST /api/buildings/{id}/correct (updates label + queues feedback + appends audit)
    const res = await fetch(`${apiBase}/api/buildings/${encodeURIComponent(targetBldId)}/correct`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        corrected_damage_class: newClass,
        responder_id: "FIELD_RESPONDER_ALPHA",
      }),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();

    // Update Screen 6 Status Footer immediately with live drift & feedback count returned by API
    updateDriftFooterUI(data.drift_distance, data.feedback_count);
    const secBanner = document.getElementById("s4SecondOpinionBanner");
    if (secBanner) {
      if (data.requires_second_opinion) {
        secBanner.style.display = "block";
        secBanner.textContent = `Second opinion flagged (${data.severity_jump}-tier jump on ${targetBldId}). Weighted at ${data.reliability_weight}.`;
      } else {
        secBanner.style.display = "none";
      }
    }

    showToast(`Correction recorded: ${targetBldId} updated to ${newClass.toUpperCase()}. Drift: ${data.drift_distance} | Queue: ${data.feedback_count}`);
    await fetchState();
    await fetchAuditLog();
    await populateScreenBuildingData(targetBldId);
    await pollDriftAndFeedback();
  } catch (err) {
    showToast("Correction submission error: " + err.message, true);
  }
}

async function inspectNextActiveLearningCandidate() {
  const apiBase = getApiBase();
  try {
    const res = await fetch(`${apiBase}/api/adaptation/queue?budget=5&responder_id=FIELD_RESPONDER_ALPHA`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const q = await res.json();
    const candidates = q.candidates || [];
    if (candidates.length === 0) {
      showToast("All buildings in current sector already reviewed.");
      return;
    }
    const nextBld = candidates[0];
    await populateScreenBuildingData(nextBld.building_id);
    const secBanner = document.getElementById("s4SecondOpinionBanner");
    if (secBanner) {
      if (nextBld.requires_second_opinion) {
        secBanner.style.display = "block";
        secBanner.textContent = `Second opinion requested for ${nextBld.building_id} (>=2 tier correction).`;
      } else {
        secBanner.style.display = "none";
      }
    }
    showToast(`Active Learning selected ${nextBld.building_id} (uncertainty ${(nextBld.uncertainty * 100).toFixed(1)}%, score ${nextBld.acquisition_score}).`);
  } catch (err) {
    showToast("Active learning queue unavailable: " + err.message, true);
  }
}

async function commanderDecisionAction(action) {
  const apiBase = getApiBase();
  const persistedPlan = tacticalState
    ? (tacticalState.active_plan || (tacticalState.active_routes && tacticalState.active_routes[0]))
    : null;
  const planId = persistedPlan ? (persistedPlan.route_id || persistedPlan.plan_id || "PLAN-BASELINE-01") : "PLAN-BASELINE-01";
  const endpoint = action === "approve" ? `/api/routes/${encodeURIComponent(planId)}/approve` : `/api/routes/${encodeURIComponent(planId)}/reject`;

  const approveBtn = document.getElementById("approveBtn");
  const rejectBtnTab = document.getElementById("rejectBtnTab");
  if (approveBtn) approveBtn.disabled = true;
  if (rejectBtnTab) rejectBtnTab.disabled = true;

  try {
    const res = await fetch(`${apiBase}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ commander_id: "CMD_LOCAL_01", notes: `Commander decision: ${action.toUpperCase()}` }),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Decision failed (HTTP ${res.status})`);
    }
    const decisionData = await res.json();
    whatIfPlan = null;
    openApprovalModal(decisionData);
    showToast(`✔ Route ${decisionData.route_id} ${String(decisionData.action || action).toLowerCase()}. SHA-256 chain updated.`, false);
    await fetchState();
    await fetchAuditLog();
    if (currentScreen === "screen-route") {
      await populateScreenRouteData();
    }
  } catch (err) {
    showToast("Decision dispatch error: " + err.message, true);
    updateUI();
  }
}

function openApprovalModal(decisionData) {
  const modal = document.getElementById("screenApprovalModal");
  if (!modal) return;
  document.querySelectorAll(".screen-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("navScreen5")?.classList.add("active");

  const titleEl = document.getElementById("modalDecisionTitle");
  const actionStr = String(decisionData.action || "").toUpperCase();
  titleEl.textContent = `COMMANDER GATE: ROUTE ${actionStr}`;
  const isRejected = actionStr.includes("REJECT");
  const isApproved = actionStr.includes("APPROV");
  const dotEl = titleEl.previousElementSibling;
  const accentColor = isRejected ? "var(--status-blocked)" : (isApproved ? "var(--accent-emerald)" : "var(--accent-cyan)");
  titleEl.style.color = isRejected ? "#f87171" : "var(--text-bright)";
  if (dotEl) {
    dotEl.style.background = accentColor;
    dotEl.style.boxShadow = `0 0 10px ${accentColor}`;
  }

  document.getElementById("modalDecisionDesc").textContent = decisionData.message;
  document.getElementById("modalAuditHashVal").textContent = decisionData.audit_hash;
  document.getElementById("modalPrevHashVal").textContent = decisionData.prev_hash;
  modal.classList.add("active");
}

async function openApprovalModalManual() {
  const apiBase = getApiBase();
  try {
    const res = await fetch(`${apiBase}/api/audit-log`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const latest = data.entries && data.entries.length > 0 ? data.entries[0] : null;
    if (latest) {
      openApprovalModal({
        action: `${latest.event_type} (SEQ #${latest.seq_id})`,
        message: `Verified live SHA-256 Merkle chain (${data.count} entries intact). Actor: ${latest.actor_role} | Timestamp: ${latest.utc_timestamp}`,
        audit_hash: latest.curr_hash,
        prev_hash: latest.prev_hash
      });
    } else {
      openApprovalModal({
        action: "AUDIT CHAIN READY (0 ENTRIES)",
        message: "No audit records written yet.",
        audit_hash: "-- NO ENTRIES --",
        prev_hash: "--",
      });
    }
  } catch (err) {
    showToast("Audit modal fetch failed: " + err.message, true);
  }
}

function closeApprovalModal() {
  document.getElementById("screenApprovalModal")?.classList.remove("active");
  const screenNavMap = {
    "screen-upload": "navScreen1",
    "screen-map": "navScreen2",
    "screen-route": "navScreen3",
    "screen-building": "navScreen4",
  };
  document.querySelectorAll(".screen-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(screenNavMap[currentScreen] || "navScreen2")?.classList.add("active");
}

let latestPortion4Drift = null;
let latestPortion4Status = null;

function syncScreen4CorrectionHint() {
  const hintEl = document.getElementById("s4CorrectHint");
  const sel = document.getElementById("s4CorrectionSelect");
  if (!hintEl || !sel) return;
  const curRaw = currentInspectedBuildingData
    ? (currentInspectedBuildingData.damage_class || currentInspectedBuildingData.class_label || "minor")
    : "minor";
  const curLbl = normalizeDamageLabel(curRaw);
  const nextLbl = normalizeDamageLabel(sel.value || curLbl);
  if (nextLbl === curLbl) {
    hintEl.className = "p4-inline-hint";
    hintEl.textContent = `Matches current AI label (${curLbl.toUpperCase()}) — pick another tier to correct, or click CONFIRM.`;
    return;
  }
  const jump = Math.abs(getDamageTier(nextLbl) - getDamageTier(curLbl));
  if (jump >= 2) {
    hintEl.className = "p4-inline-hint warn";
    hintEl.textContent = `${jump}-tier shift (${curLbl.toUpperCase()} -> ${nextLbl.toUpperCase()}) — CORRECT will flag a 2nd-opinion review.`;
  } else {
    hintEl.className = "p4-inline-hint";
    hintEl.textContent = `1-tier adjustment (${curLbl.toUpperCase()} -> ${nextLbl.toUpperCase()}) ready for Portion 4 queue.`;
  }
}

function _renderP4TextChips(container, labels, tone) {
  if (!container) return;
  container.textContent = "";
  for (const label of labels) {
    const chip = document.createElement("span");
    chip.className = tone ? `p4-chip ${tone}` : "p4-chip";
    chip.textContent = label;
    container.appendChild(chip);
  }
}

function updatePortion4VisualIndicators(d, st) {
  if (d) latestPortion4Drift = d;
  if (st) latestPortion4Status = st;
  const driftObj = d || latestPortion4Drift;
  const statusObj = st || latestPortion4Status;

  const driftDist = Number((driftObj && driftObj.drift_distance) ?? (statusObj && statusObj.drift && statusObj.drift.drift_distance) ?? 0);
  const softThresh = Number((driftObj && driftObj.soft_threshold) ?? 0.08);
  const hardThresh = Number((driftObj && driftObj.threshold) ?? 0.15);
  const isHard = driftDist >= hardThresh;
  const isSoft = !isHard && driftDist >= softThresh;
  const tierWord = isHard ? "DRIFT ALERT" : (isSoft ? "SOFT WARN" : "NORMAL");

  const replaySum = (statusObj && statusObj.replay_summary) || {};
  const ltCount = Number(replaySum.long_term_count ?? 0);
  const recCount = Number(replaySum.recent_count ?? 0);
  const fbTotal = Number((driftObj && driftObj.total_feedback_samples) ?? (statusObj && statusObj.feedback_stats && statusObj.feedback_stats.total_count) ?? 0);
  const secondOpCount = Number((statusObj && statusObj.second_opinion_count) ?? 0);
  const embAvail = Boolean(driftObj && driftObj.embedding_signal_available);
  const embReason = String((driftObj && driftObj.embedding_reason) || (statusObj && statusObj.drift && statusObj.drift.embedding_reason) || "");
  const activeModel = String((statusObj && statusObj.active_model_version) || "ResNet18-UNet-xBD-v1.0");
  const activeAdapter = String((statusObj && statusObj.active_adapter_version) || (driftObj && driftObj.active_adapter_version) || "none");

  const hdrBtn = document.getElementById("btnPortion4Status");
  const hdrText = document.getElementById("p4HeaderBadgeText");
  if (hdrBtn && hdrText) {
    hdrBtn.className = isHard ? "btn-p4-indicator alert" : (isSoft ? "btn-p4-indicator warn" : "btn-p4-indicator");
    hdrText.textContent = `P4 ACTIVE • DRIFT ${driftDist.toFixed(3)} (${tierWord}) • ${fbTotal} FB • ${ltCount} LT`;
  }

  const s4Pill = document.getElementById("s4P4SignalPill");
  if (s4Pill) {
    s4Pill.className = isHard ? "tour-status-pill stub" : (isSoft ? "tour-status-pill stub" : "tour-status-pill live");
    s4Pill.textContent = embAvail ? "MMD2 + OPERATIONAL LIVE" : (isHard || isSoft ? `${tierWord} • OPERATIONAL` : "OPERATIONAL ACTIVE");
  }
  const s4Drift = document.getElementById("s4P4DriftVal");
  if (s4Drift) s4Drift.textContent = `${driftDist.toFixed(4)} (${tierWord})`;
  const s4Replay = document.getElementById("s4P4ReplayVal");
  if (s4Replay) s4Replay.textContent = `${ltCount} LT / ${recCount} REC`;
  const s4Sec = document.getElementById("s4P4SecondOpinionVal");
  if (s4Sec) s4Sec.textContent = `${secondOpCount} FLAGGED`;
  const s4Scope = document.getElementById("s4P4ScopeVal");
  if (s4Scope) {
    const coupled = Boolean(statusObj && statusObj.routing_policy_coupled);
    s4Scope.textContent = coupled ? "COUPLED" : "SEG-ONLY (MARL SAFE)";
  }

  // Modal fields
  const setModal = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  const modalPill = document.getElementById("p4ModalDriftPill");
  if (modalPill) {
    modalPill.className = (isHard || isSoft) ? "tour-status-pill stub" : "tour-status-pill live";
    modalPill.textContent = tierWord;
  }
  const op = (driftObj && driftObj.operational_metrics) || (statusObj && statusObj.drift && statusObj.drift.operational_metrics) || {};
  setModal("p4ModalDriftDist", driftDist.toFixed(4));
  setModal("p4ModalDriftThresholds", `${softThresh.toFixed(2)} / ${hardThresh.toFixed(2)}`);
  setModal("p4ModalClassJsd", Number(op.class_jsd ?? 0).toFixed(4));
  setModal("p4ModalMeanUnc", Number(op.mean_uncertainty ?? 0).toFixed(4));
  setModal("p4ModalCorrRate", Number(op.shrunk_correction_rate ?? 0).toFixed(4));
  setModal("p4ModalEmbAvail", embAvail ? "ACTIVE (MMD2)" : "FALLBACK (NO REF)");
  setModal("p4ModalEmbReason", embAvail ? "Reference embeddings verified; RBF MMD2 + Sliced Wasserstein active." : (embReason || "Operational signals active; awaiting data/exports/reference/reference.npy for embedding MMD2."));

  setModal("p4ModalLtCount", `${ltCount} samples`);
  setModal("p4ModalRecCount", `${recCount} samples`);
  setModal("p4ModalTotalFb", String(fbTotal));
  setModal("p4ModalSecondOp", String(secondOpCount));
  setModal("p4ModalModelVer", activeModel);
  setModal("p4ModalAdapterVer", activeAdapter === "none" ? "BASE (none)" : activeAdapter);

  const disasters = Array.isArray(replaySum.disasters_retained) ? replaySum.disasters_retained : [];
  _renderP4TextChips(
    document.getElementById("p4ModalDisastersWrap"),
    disasters.length > 0 ? disasters.map(name => `STRATUM: ${name}`) : ["NO REPLAY STRATA YET"],
    disasters.length > 0 ? "pass" : ""
  );

  const recentRuns = (statusObj && Array.isArray(statusObj.recent_runs)) ? statusObj.recent_runs : [];
  const lastRun = recentRuns[0] || null;
  const gateWrap = document.getElementById("p4ModalGateChecksWrap");
  if (lastRun) {
    const verdict = lastRun.promoted ? "PROMOTED" : (lastRun.gate_passed ? "SHADOW STAGED" : "REJECTED (FAIL-CLOSED)");
    setModal("p4ModalLastRunSummary", `Latest Run ${lastRun.run_id || "--"} (${lastRun.tier || "tier_a"}): ${verdict} • Adapter: ${lastRun.adapter_version || "--"}`);
    const ts = lastRun.gate_report && lastRun.gate_report.training_summary;
    if (ts) {
      setModal(
        "p4ModalTrainMetrics",
        `Tier-A PyTorch (${ts.imagery_source || "imagery"}): ${ts.verified_buildings_used ?? 0} verified (${ts.train_buildings ?? 0} train / ${ts.holdout_buildings ?? 0} holdout) • Loss ${Number(ts.final_loss ?? 0).toFixed(4)} • Macro F1 ${Number(ts.baseline_macro_f1 ?? 0).toFixed(3)} -> ${Number(ts.candidate_macro_f1 ?? 0).toFixed(3)} • Holdout F1 ${Number(ts.baseline_holdout_f1 ?? 0).toFixed(3)} -> ${Number(ts.candidate_holdout_f1 ?? 0).toFixed(3)} • ECE ${Number(ts.baseline_ece ?? 0).toFixed(3)} -> ${Number(ts.candidate_ece ?? 0).toFixed(3)} • ${Number(ts.duration_sec ?? 0).toFixed(2)}s`
      );
    } else {
      setModal("p4ModalTrainMetrics", "Gate evaluation recorded without live PyTorch Tier-A training summary.");
    }
    if (gateWrap) {
      gateWrap.textContent = "";
      const checks = (lastRun.gate_report && lastRun.gate_report.checks) || {};
      for (const [k, v] of Object.entries(checks)) {
        const chip = document.createElement("span");
        const passed = Boolean(v && v.passed);
        chip.className = passed ? "p4-chip pass" : "p4-chip fail";
        chip.textContent = `${k}: ${passed ? "PASS" : "FAIL"}`;
        gateWrap.appendChild(chip);
      }
    }
  } else {
    setModal("p4ModalLastRunSummary", "No adaptation runs recorded yet. All 5 Safety Gate checks fail closed on run trigger.");
    setModal("p4ModalTrainMetrics", "Click TRAIN & ADAPT MODEL (TIER A) to fine-tune the frozen-encoder ResNet-18 adapter head on responder-verified labels and run all 5 safety checks.");
    _renderP4TextChips(gateWrap, ["per_class_f1", "forgetting_check", "geographic_holdout", "calibration_preserved", "edge_latency"], "");
  }

  const rbBtn = document.getElementById("btnP4Rollback");
  if (rbBtn) {
    rbBtn.disabled = activeAdapter === "none";
    rbBtn.textContent = activeAdapter === "none" ? "ALREADY AT BASE WEIGHTS" : `ROLLBACK ${activeAdapter} -> BASE`;
  }
}

function openPortion4MonitorModal() {
  document.getElementById("portion4MonitorModal")?.classList.add("active");
  pollDriftAndFeedback();
}

function closePortion4MonitorModal() {
  document.getElementById("portion4MonitorModal")?.classList.remove("active");
}

async function triggerPortion4Adaptation() {
  const modalBtn = document.getElementById("btnP4RunAdaptation");
  const cardBtn = document.getElementById("btnS4AdaptNow");
  if (modalBtn) {
    modalBtn.disabled = true;
    modalBtn.textContent = "TRAINING TIER-A ADAPTER...";
  }
  if (cardBtn) {
    cardBtn.disabled = true;
    cardBtn.textContent = "ADAPTING...";
  }
  const modalEl = document.getElementById("portion4MonitorModal");
  if (modalEl && !modalEl.classList.contains("active")) {
    modalEl.classList.add("active");
  }

  try {
    const res = await fetch(`${getApiBase()}/api/adaptation/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        disaster_id: "HOUSTON_HARVEY_2017",
        tier: "tier_a",
        promote: true,
        epochs: 8,
        learning_rate: 0.02,
        train_from_feedback: true,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    if (data.promoted) {
      showToast(`Tier-A adapter ${data.adapter_version} trained & promoted via 5-check safety gate.`);
    } else if (data.gate_passed) {
      showToast(`Tier-A adapter ${data.adapter_version} passed gate and is shadow-staged.`);
    } else {
      const checks = (data.gate_report && data.gate_report.checks) || {};
      const failedNames = Object.entries(checks).filter(([, v]) => !v || !v.passed).map(([k]) => k).join(", ");
      showToast(`Tier-A run ${data.adapter_version} blocked by safety gate${failedNames ? ` (${failedNames})` : " (fail-closed)"}.`, true);
    }

    await pollDriftAndFeedback();
    await fetchAuditLog();
    if (currentScreen === "screen-building" && currentInspectedBuildingId) {
      await populateScreenBuildingData(currentInspectedBuildingId);
    }
  } catch (err) {
    showToast("Tier-A adaptation error: " + err.message, true);
  } finally {
    if (modalBtn) {
      modalBtn.disabled = false;
      modalBtn.textContent = "⚡ TRAIN & ADAPT MODEL (TIER A)";
    }
    if (cardBtn) {
      cardBtn.disabled = false;
      cardBtn.textContent = "ADAPT ⚡";
    }
  }
}

async function triggerPortion4Rollback() {
  const rbBtn = document.getElementById("btnP4Rollback");
  if (rbBtn) rbBtn.disabled = true;
  try {
    const res = await fetch(`${getApiBase()}/api/adaptation/rollback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ commander_id: "CMD_LOCAL_01", reason: "Operator rollback from Portion 4 monitor" }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    showToast(data.message || "Adapter rollback completed.");
    await pollDriftAndFeedback();
    await fetchAuditLog();
  } catch (err) {
    showToast("Rollback error: " + err.message, true);
  }
}

async function pollDriftAndFeedback() {
  const apiBase = getApiBase();
  try {
    const [driftRes, statusRes] = await Promise.all([
      fetch(`${apiBase}/api/adaptation/drift`),
      fetch(`${apiBase}/api/adaptation/status`),
    ]);
    const d = driftRes.ok ? await driftRes.json() : null;
    const st = statusRes.ok ? await statusRes.json() : null;
    if (d) {
      updateDriftFooterUI(
        d.drift_distance,
        d.total_feedback_samples,
        (st && st.active_adapter_version) || d.active_adapter_version || "none"
      );
    }
    if (d || st) {
      updatePortion4VisualIndicators(d, st);
    }
  } catch (e) {
    const hdrBtn = document.getElementById("btnPortion4Status");
    const hdrText = document.getElementById("p4HeaderBadgeText");
    if (hdrBtn && hdrText) {
      hdrBtn.className = "btn-p4-indicator alert";
      hdrText.textContent = "P4 OFFLINE";
    }
  }
}
setInterval(pollDriftAndFeedback, 5000);

function openProjectTour() {
  document.getElementById("projectTourModal")?.classList.add("active");
}

function closeProjectTour() {
  document.getElementById("projectTourModal")?.classList.remove("active");
}

function inspectSampleBuilding() {
  const feats = (tacticalState && tacticalState.buildings_geojson && tacticalState.buildings_geojson.features) || [];
  if (feats.length === 0) {
    showToast("Building data is still syncing.", true);
    return;
  }
  const satFeats = feats.filter(f => f.properties && f.properties.data_source === "resnet18_unet_pretrained");
  const pool = satFeats.length > 0 ? satFeats : feats;
  const feat = pool.find(f => f.properties && f.properties.status_flag === "LOW_CONFIDENCE") || pool[0];
  selectedEntity = { type: "building", data: feat };
  renderInspectorBuilding(feat);
  switchTab("tab-inspect");
  showToast(`Inspecting ${feat.properties.building_id} (${feat.properties.status_flag}).`, false);
}

function inspectSampleRoad(preferredEdgeId = "E-03") {
  const feats = (tacticalState && tacticalState.roads_geojson && tacticalState.roads_geojson.features) || [];
  if (feats.length === 0) {
    showToast("Road network is still syncing.", true);
    return;
  }
  const feat =
    feats.find(f => f.properties && f.properties.edge_id === preferredEdgeId) ||
    feats.find(f => f.properties && f.properties.edge_status === "BLOCKED") ||
    feats[0];
  selectedEntity = { type: "road", data: feat };
  renderInspectorRoad(feat);
  switchTab("tab-inspect");
  showToast(`Inspecting road ${feat.properties.edge_id} (${feat.properties.edge_status}).`, false);
}

function runWorkflowJump(action) {
  closeProjectTour();
  if (action === "screen-upload" || action === "screen-route" || action === "screen-building") {
    switchScreen(action);
  } else if (action === "tab-sim") {
    switchScreen("screen-map");
    switchTab("tab-sim");
  } else if (action === "inspect-purple") {
    switchScreen("screen-map");
    inspectSampleBuilding();
  } else if (action === "inspect-bridge") {
    switchScreen("screen-map");
    inspectSampleRoad("E-03");
  } else if (action === "modal-audit") {
    openApprovalModalManual();
  }
}

function wireProjectGuidanceControls() {
  document.getElementById("btnOpenTour")?.addEventListener("click", openProjectTour);
  document.getElementById("btnOpenTourFromGuide")?.addEventListener("click", openProjectTour);
  document.getElementById("btnCloseTourModal")?.addEventListener("click", closeProjectTour);
  document.getElementById("btnCloseTourFooter")?.addEventListener("click", closeProjectTour);

  const tourOverlay = document.getElementById("projectTourModal");
  tourOverlay?.addEventListener("click", (evt) => {
    if (evt.target === tourOverlay) closeProjectTour();
  });

  document.querySelectorAll(".pipeline-step-btn[data-jump]").forEach(btn => {
    btn.addEventListener("click", () => runWorkflowJump(btn.getAttribute("data-jump")));
  });

  document.querySelectorAll("[data-tour-action]").forEach(btn => {
    btn.addEventListener("click", () => runWorkflowJump(btn.getAttribute("data-tour-action")));
  });

  document.getElementById("btnQuickInspectPurple")?.addEventListener("click", inspectSampleBuilding);
  document.getElementById("btnQuickInspectBridge")?.addEventListener("click", () => inspectSampleRoad("E-03"));

  document.getElementById("btnPortion4Status")?.addEventListener("click", openPortion4MonitorModal);
  document.getElementById("btnS4OpenP4Modal")?.addEventListener("click", openPortion4MonitorModal);
  document.getElementById("btnS4AdaptNow")?.addEventListener("click", triggerPortion4Adaptation);
  document.getElementById("btnScreen6P4Inspect")?.addEventListener("click", openPortion4MonitorModal);
  document.getElementById("btnCloseP4Modal")?.addEventListener("click", closePortion4MonitorModal);
  document.getElementById("btnCloseP4Footer")?.addEventListener("click", closePortion4MonitorModal);
  document.getElementById("btnP4RunAdaptation")?.addEventListener("click", triggerPortion4Adaptation);
  document.getElementById("btnP4Refresh")?.addEventListener("click", pollDriftAndFeedback);
  document.getElementById("btnP4Rollback")?.addEventListener("click", triggerPortion4Rollback);
  document.getElementById("s4CorrectionSelect")?.addEventListener("change", syncScreen4CorrectionHint);

  const p4Overlay = document.getElementById("portion4MonitorModal");
  p4Overlay?.addEventListener("click", (evt) => {
    if (evt.target === p4Overlay) closePortion4MonitorModal();
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  try {
    wireTileDropZones();
    wireProjectGuidanceControls();
    tacticalMap = new TacticalLeafletMap("leafletMap");
    sectorBuilder = new SectorMapBuilder(tacticalMap);
    tacticalMap.attachBuilder(sectorBuilder);
    await fetchState();
    await fetchAuditLog();
    await pollDriftAndFeedback();
    requestAnimationFrame(animLoop);
  } catch (err) {
    console.error("Leaflet Map Initialization Error:", err);
    const banner = document.getElementById("mapFallbackBanner");
    if (banner) banner.style.display = "block";
  }
});
