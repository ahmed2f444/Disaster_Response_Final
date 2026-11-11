"""Extract road network graphs directly from pre/post satellite imagery into SQLite `road_edges`."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree

import db
from damage.dataset import decode_image_input
from .graph import haversine_m

DEFAULT_GRID_SIZE = 256
MIN_REAL_ROAD_COVERAGE = 0.020


def _load_rgb_array(src: Path | str | Image.Image | np.ndarray, size: int = DEFAULT_GRID_SIZE) -> np.ndarray:
    """Load an RGB image array (path, data URI, PIL, or ndarray) resized to `(size, size)` in `[0, 1]`."""
    if isinstance(src, np.ndarray):
        arr = np.asarray(src, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[:2] == (size, size) and float(arr.max(initial=0.0)) <= 1.0:
            return np.clip(arr, 0.0, 1.0)
        arr_u8 = src if src.dtype == np.uint8 else np.clip(src * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr_u8).convert("RGB")
    else:
        pil_img, _ = decode_image_input(src)
    resized = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _resize_mask_prob(building_prob: np.ndarray | None, size: int = DEFAULT_GRID_SIZE) -> np.ndarray:
    """Normalize and resize optional 2D building probability map to `(size, size)`."""
    if building_prob is None:
        return np.zeros((size, size), dtype=np.float32)
    arr = np.asarray(building_prob, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.clip(arr, 0.0, 1.0)
    if arr.shape == (size, size):
        return arr
    pil_m = Image.fromarray((arr * 255.0).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(pil_m, dtype=np.float32) / 255.0


def _rdp_simplify(pts: list[tuple[float, float]], eps: float, max_points: int = 18) -> list[tuple[float, float]]:
    """Reduce vertices along a road polyline using Ramer-Douglas-Peucker simplification."""
    if len(pts) <= 2:
        return list(pts)
    a, b = pts[0], pts[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = math.hypot(dx, dy)
    best_i, best_d = -1, -1.0
    for idx in range(1, len(pts) - 1):
        p = pts[idx]
        if denom < 1e-6:
            d = math.hypot(p[0] - a[0], p[1] - a[1])
        else:
            d = abs(dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]) / denom
        if d > best_d:
            best_d = d
            best_i = idx
    if best_d > eps and best_i > 0:
        left = _rdp_simplify(pts[: best_i + 1], eps, max_points=max_points)
        right = _rdp_simplify(pts[best_i:], eps, max_points=max_points)
        merged = left[:-1] + right
    else:
        merged = [a, b]
    if len(merged) <= max_points:
        return merged
    idxs = [int(round(i * (len(merged) - 1) / (max_points - 1))) for i in range(max_points)]
    return [merged[i] for i in dict.fromkeys(idxs)]


def _compute_road_likelihood(arr: np.ndarray, bprob: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute per-pixel tubular road probability and textured non-building road fraction."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    greenness = np.maximum(0.0, np.minimum(g - r, g - b))
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    non_green = np.clip(1.0 - np.maximum(0.0, greenness - 0.008) / 0.045, 0.04, 1.0)
    neutral = np.clip(1.0 - np.maximum(0.0, sat - 0.14) / 0.28, 0.12, 1.0)
    non_dark_shadow = np.clip((lum - 0.10) / 0.22, 0.0, 1.0)

    bldg_core = ndimage.grey_dilation(bprob, size=(3, 3))
    non_building = np.clip(1.0 - bldg_core * 0.80, 0.10, 1.0)
    surface_prior = non_green * neutral * non_dark_shadow * non_building

    tube_max = np.zeros_like(lum)
    for sigma in (1.3, 2.2, 3.3):
        gxx = ndimage.gaussian_filter(lum, sigma=sigma, order=(0, 2)) * (sigma ** 2)
        gyy = ndimage.gaussian_filter(lum, sigma=sigma, order=(2, 0)) * (sigma ** 2)
        gxy = ndimage.gaussian_filter(lum, sigma=sigma, order=(1, 1)) * (sigma ** 2)
        trace = gxx + gyy
        diff = gxx - gyy
        rad = np.sqrt(diff * diff + 4.0 * gxy * gxy)
        l_plus = 0.5 * (trace + rad)
        l_minus = 0.5 * (trace - rad)
        swap = np.abs(l_minus) > np.abs(l_plus)
        lam2 = np.where(swap, l_minus, l_plus)
        lam1 = np.where(swap, l_plus, l_minus)
        ratio_sq = (np.abs(lam1) / (np.abs(lam2) + 1e-6)) ** 2
        elongation = np.exp(-ratio_sq / 0.35)
        bright_ridge = np.maximum(0.0, -lam2) * elongation
        dark_asphalt = np.maximum(0.0, lam2) * elongation * (sat < 0.14) * (lum < 0.45) * 0.85
        resp = (bright_ridge + dark_asphalt) * surface_prior
        p99 = float(np.percentile(resp, 99.5))
        if p99 > 1e-4:
            tube_max = np.maximum(tube_max, np.clip(resp / p99, 0.0, 1.0))

    smoothed = ndimage.gaussian_filter(lum * surface_prior, sigma=0.7)
    opened = ndimage.grey_opening(smoothed, size=(9, 9))
    diff_th = np.maximum(0.0, smoothed - opened)
    p99_th = float(np.percentile(diff_th, 99.0))
    tophat = np.clip(diff_th / p99_th, 0.0, 1.0) if p99_th > 0.035 else np.zeros_like(lum)

    road_prob = np.clip(0.62 * tube_max + 0.38 * tophat, 0.0, 1.0)
    road_prob = ndimage.gaussian_filter(road_prob, sigma=1.0) * non_building

    local_mean = ndimage.uniform_filter(lum, size=5)
    local_sq = ndimage.uniform_filter(lum * lum, size=5)
    local_std = np.sqrt(np.maximum(0.0, local_sq - local_mean * local_mean))
    # Exclude uniform solid monochrome blocks (synthetic fixtures) & their step edges
    solid_flat_halo = ndimage.binary_dilation((local_std < 0.004) & (lum > 0.45) & (greenness < 0.015), iterations=5)
    textured_road = (road_prob > 0.22) & (greenness < 0.015) & (local_std > 0.020) & (bprob < 0.25) & (~solid_flat_halo)
    return road_prob, float(np.mean(textured_road))


def _select_intersection_nodes(
    road_prob: np.ndarray,
    bprob: np.ndarray,
    n_nodes: int = 13,
) -> list[tuple[int, int]]:
    """Select well-distributed road hubs grounded on visible streets rather than roofs or open fields."""
    h, w = road_prob.shape
    local_road = ndimage.uniform_filter(road_prob, size=7)
    score_map = road_prob.copy()
    margin = 8
    score_map[:margin, :] = 0.0
    score_map[-margin:, :] = 0.0
    score_map[:, :margin] = 0.0
    score_map[:, -margin:] = 0.0
    score_map *= ((bprob < 0.25) & (local_road > 0.16)).astype(np.float32)

    local_max = (score_map == ndimage.maximum_filter(score_map, size=13)) & (score_map > 0.30)
    ys, xs = np.where(local_max)
    pts = sorted(zip(ys.tolist(), xs.tolist()), key=lambda p: float(score_map[p[0], p[1]]), reverse=True)
    if len(pts) < 4:
        local_max = (score_map == ndimage.maximum_filter(score_map, size=11)) & (score_map > 0.18)
        ys, xs = np.where(local_max)
        pts = sorted(zip(ys.tolist(), xs.tolist()), key=lambda p: float(score_map[p[0], p[1]]), reverse=True)
    if not pts:
        return []

    selected: list[tuple[int, int]] = [pts[0]]
    min_sep = h * 0.13
    for _ in range(n_nodes - 1):
        best: tuple[int, int] | None = None
        best_val = -1.0
        for cand in pts[:140]:
            dmin = min(math.hypot(cand[0] - s[0], cand[1] - s[1]) for s in selected)
            if dmin < min_sep:
                continue
            val = (dmin ** 0.42) * (float(score_map[cand[0], cand[1]]) ** 1.8)
            if val > best_val:
                best_val = val
                best = cand
        if best is None:
            break
        selected.append(best)
    return selected


def _trace_corridor_edges(
    road_prob: np.ndarray,
    bprob: np.ndarray,
    nodes_yx: list[tuple[int, int]],
    pre_arr: np.ndarray,
    post_arr: np.ndarray,
) -> list[tuple[int, int, list[tuple[float, float]], float]]:
    """Trace least-cost curved street polylines between intersection hubs via sparse pixel Dijkstra."""
    h, w = road_prob.shape
    cost = np.exp(6.2 * (1.0 - road_prob)) * (1.0 + 18.0 * bprob)
    flat_cost = cost.ravel()
    idx_grid = np.arange(h * w).reshape(h, w)

    rows_idx, cols_idx, weights = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        u = idx_grid[y0:y1, x0:x1].ravel()
        v = idx_grid[y0 + dy : y1 + dy, x0 + dx : x1 + dx].ravel()
        w_step = 0.5 * (flat_cost[u] + flat_cost[v]) * math.hypot(dy, dx)
        rows_idx.extend((u, v))
        cols_idx.extend((v, u))
        weights.extend((w_step, w_step))

    graph_mat = csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows_idx), np.concatenate(cols_idx))),
        shape=(h * w, h * w),
    )
    sources = [y * w + x for y, x in nodes_yx]
    dist_mat, pred = dijkstra(graph_mat, directed=False, indices=sources, return_predecessors=True)

    k = len(nodes_yx)
    pair_costs = np.zeros((k, k), dtype=np.float64)
    paths: dict[tuple[int, int], list[tuple[int, int]]] = {}
    mean_road: dict[tuple[int, int], float] = {}
    for i in range(k):
        for j in range(i + 1, k):
            tgt = sources[j]
            pair_costs[i, j] = pair_costs[j, i] = float(dist_mat[i, tgt])
            cur = tgt
            pth: list[tuple[int, int]] = []
            while cur != -9999 and len(pth) < 2000:
                pth.append((cur // w, cur % w))
                if cur == sources[i]:
                    break
                cur = int(pred[i, cur])
            pth.reverse()
            if not pth or pth[0] != nodes_yx[i]:
                pth = [nodes_yx[i], nodes_yx[j]]
            paths[(i, j)] = pth
            mean_road[(i, j)] = float(np.mean([road_prob[py, px] for py, px in pth]))

    mst = minimum_spanning_tree(pair_costs).toarray()
    chosen: set[tuple[int, int]] = set()
    deg = [0] * k
    used_pixels: set[tuple[int, int]] = set()
    for i in range(k):
        for j in range(i + 1, k):
            if mst[i, j] > 0 or mst[j, i] > 0:
                chosen.add((i, j))
                deg[i] += 1
                deg[j] += 1
                used_pixels.update(paths[(i, j)])

    cands: list[tuple[float, int, int]] = []
    for i in range(k):
        for j in range(i + 1, k):
            if (i, j) in chosen:
                continue
            euclid = math.hypot(nodes_yx[i][0] - nodes_yx[j][0], nodes_yx[i][1] - nodes_yx[j][1])
            if euclid > h * 0.45:
                continue
            cands.append((pair_costs[i, j] / max(12.0, euclid), i, j))
    cands.sort()
    for _, i, j in cands:
        if deg[i] >= 4 or deg[j] >= 4:
            continue
        if mean_road[(i, j)] < 0.18:
            continue
        pth = paths[(i, j)]
        interior = pth[3:-3] if len(pth) > 6 else pth
        if interior and (sum(1 for pt in interior if pt in used_pixels) / len(interior)) > 0.35:
            continue
        chosen.add((i, j))
        deg[i] += 1
        deg[j] += 1
        used_pixels.update(pth)

    delta_rgb = np.mean(np.abs(post_arr - pre_arr), axis=-1)
    edges_out: list[tuple[int, int, list[tuple[float, float]], float]] = []
    for i, j in sorted(chosen):
        pth = paths[(i, j)]
        delta_p75 = float(np.percentile([delta_rgb[py, px] for py, px in pth], 75))
        pts_xy = [(float(px), float(py)) for py, px in pth]
        simp = _rdp_simplify(pts_xy, eps=1.8, max_points=18)
        edges_out.append((i, j, simp, delta_p75))
    return edges_out


def extract_road_graph_from_imagery(
    pre_image: Path | str | Image.Image | np.ndarray,
    post_image: Path | str | Image.Image | np.ndarray,
    bounds: tuple[float, float, float, float],
    building_prob: np.ndarray | None = None,
    n_nodes: int = 13,
) -> dict[str, Any] | None:
    """Extract a curved road network embedded in `bounds` from real pre/post satellite imagery.

    Returns `None` when the pre-disaster image contains too little road structure (e.g. uniform
    test fixtures) so caller preserves baseline Houston-East roads cleanly.
    """
    pre_arr = _load_rgb_array(pre_image, size=DEFAULT_GRID_SIZE)
    post_arr = _load_rgb_array(post_image, size=DEFAULT_GRID_SIZE)
    bprob = _resize_mask_prob(building_prob, size=DEFAULT_GRID_SIZE)

    road_prob, road_cov = _compute_road_likelihood(pre_arr, bprob)
    if road_cov < MIN_REAL_ROAD_COVERAGE:
        return None

    nodes_yx = _select_intersection_nodes(road_prob, bprob, n_nodes=n_nodes)
    if len(nodes_yx) < 4:
        return None

    edges = _trace_corridor_edges(road_prob, bprob, nodes_yx, pre_arr, post_arr)
    if len(edges) < 4:
        return None

    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bounds)
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    scale_denom = float(DEFAULT_GRID_SIZE)

    def px_to_wgs84(px: float, py: float) -> list[float]:
        u = min(1.0, max(0.0, px / scale_denom))
        v = min(1.0, max(0.0, py / scale_denom))
        return [round(min_lon + u * lon_span, 6), round(max_lat - v * lat_span, 6)]

    edge_payloads: list[dict[str, Any]] = []
    for i, j, pts_xy, delta_p75 in edges:
        coords = [px_to_wgs84(px, py) for px, py in pts_xy]
        coords[0] = px_to_wgs84(float(nodes_yx[i][1]), float(nodes_yx[i][0]))
        coords[-1] = px_to_wgs84(float(nodes_yx[j][1]), float(nodes_yx[j][0]))
        length_m = sum(
            haversine_m(coords[idx][1], coords[idx][0], coords[idx + 1][1], coords[idx + 1][0])
            for idx in range(len(coords) - 1)
        )
        length_m = max(60.0, round(length_m, 2))
        travel_sec = round(max(6.0, length_m / (36.0 / 3.6)), 2)

        corridor_disruption = round(min(0.65, max(0.0, (delta_p75 - 0.13) * 3.0)), 3)
        if corridor_disruption >= 0.45:
            closure, cap, status = True, 0.0, "BLOCKED"
        elif corridor_disruption >= 0.15:
            closure, cap, status = False, 0.5, "HAZARDOUS"
        else:
            closure, cap, status = False, 1.0, "OPEN"

        edge_payloads.append(
            {
                "u_node": int(i),
                "v_node": int(j),
                "geometry": {"type": "LineString", "coordinates": coords},
                "length_m": length_m,
                "base_travel_sec": travel_sec,
                "flood_depth_m": corridor_disruption,
                "debris_score": 0.0,
                "closure_state": closure,
                "hazard_level": corridor_disruption,
                "capacity": cap,
                "vehicle_restriction": "closed" if closure else ("high_clearance_only" if status == "HAZARDOUS" else "none"),
                "edge_status": status,
            }
        )

    return {"road_coverage": road_cov, "node_count": len(nodes_yx), "edges": edge_payloads}


def _edges_match_digest_and_bounds(
    existing_rows: list[dict[str, Any]],
    prefix: str,
    bounds: tuple[float, float, float, float],
) -> bool:
    if not existing_rows or not all(str(r.get("edge_id", "")).startswith(prefix) for r in existing_rows):
        return False
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bounds)
    for row in existing_rows:
        for pt in (row.get("geometry") or {}).get("coordinates") or []:
            lon, lat = float(pt[0]), float(pt[1])
            if lon < min_lon - 1e-3 or lon > max_lon + 1e-3 or lat < min_lat - 1e-3 or lat > max_lat + 1e-3:
                return False
    return True


def sync_imagery_road_graph_to_db(
    pre_image: Path | str | Image.Image | np.ndarray,
    post_image: Path | str | Image.Image | np.ndarray,
    bounds: tuple[float, float, float, float],
    pair_digest: str,
    building_prob: np.ndarray | None = None,
    db_path: str = db.DEFAULT_DB_PATH,
    n_nodes: int = 13,
) -> bool:
    """Extract and persist imagery road edges (`IMG-<digest>-NN`) idempotently."""
    digest_tag = (pair_digest or "AUTO")[:8].upper()
    prefix = f"IMG-{digest_tag}-"
    existing_rows = db.get_road_edges(db_path=db_path)
    if _edges_match_digest_and_bounds(existing_rows, prefix, bounds):
        return False

    extracted = extract_road_graph_from_imagery(
        pre_image=pre_image,
        post_image=post_image,
        bounds=bounds,
        building_prob=building_prob,
        n_nodes=n_nodes,
    )
    if extracted is None:
        return False

    rows = [
        {**edge, "edge_id": f"{prefix}{idx:02d}"}
        for idx, edge in enumerate(extracted["edges"], start=1)
    ]
    db.replace_road_edges(rows, db_path=db_path)
    return True

