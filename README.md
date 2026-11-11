# Disaster Response Platform — Portion 5 Backbone (`disaster_platform`-Aligned)

Central integration spine, Pydantic V2 contracts (`Building.damage_class` enum `no_damage|minor|major|destroyed`), SQLite WAL persistence, SHA-256 Merkle audit ledger, Explanation Validation & ETA Gate, and 6-in-1 offline Leaflet v1.9.4 + SVG/Canvas circle-marker Tactical Web Console (`< 50 ms` map render, `< 100 KB` GeoJSON) for **Portion 5 (Ahmed)**, structured identically to `disaster_platform`.

---

## 1. Quick Start

Double-click `run_demo.bat` (or run `.\run_demo.bat` in terminal; pass `--reset` to re-seed `data\poc_mission.db`), or run manually:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe seed.py
.\.venv\Scripts\python.exe main.py
```

- **Web Console:** `http://127.0.0.1:8000`
- **Swagger / OpenAPI Docs (18 Contract Endpoints):** `http://127.0.0.1:8000/docs`
- **API Specification:** [`docs/api.md`](docs/api.md)

### Run Automated Tests
```bash
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 2. Ownership & Directory Rules (`disaster_platform` Parity)

| Path | Owner | Responsibility |
| :--- | :--- | :--- |
| `main.py`, `models.py`, `db.py`, `seed.py`, `static/`, `docs/api.md`, `tests/` | **Ahmed (Portion 5)** | FastAPI 18-endpoint contract, Validation & ETA Gate, SQLite WAL + 24 zero-SQL helpers, SHA-256 audit chain, 6-in-1 UI |
| `damage/` (`__init__.py`, `assess.py`, `dataset.py`, `exports.py`, `model.py`, `train.py`) | **Ammar (Portion 1)** | **Integrated** — Pretrained ResNet-18 U-Net (`data/models/unet_resnet18_best.pth`) + Grad-CAM + embeddings |
| `routing/` (`__init__.py`, `env.py`, `features.py`, `generate.py`, `graph.py`, `mask.py`, `train.py`) | **Negm (Portion 2)** | **Integrated** — Houston-East 13-node graph, PPO MARL policy, deterministic safety mask & What-If recompute |
| `explain/` (`__init__.py`, `generate.py`, `gradcam.py`, `llm.py`, `shap_explainer.py`, `validate.py`) | **Yousef (Portion 3)** | **Integrated** — Grad-CAM spatial saliency & PNG overlays, 7-feature KernelSHAP (`ppo_proposed_path`), Section 7 GTF validator & local Mistral-7B (`data/models/mistral-7b.gguf`) |
| `feedback/` (`__init__.py`, `config.py`, `feedback.py`, `drift.py`, `calibrate.py`, `selection.py`, `replay.py`, `adapt.py`, `gate.py`, `status.py`) | **Abdulrahman (Portion 4)** | **Integrated** — Multi-signal drift monitor, uncertainty+diversity active learning, dual-buffer replay, Tier A adapter & 5-check safety gate |

**Rules:**
- Only edit your own directory.
- If you need a shared model change, coordinate with Ahmed (`models.py`, `db.py`).
- Never write raw SQL outside `db.py`.
- Do not commit `data/` datasets or model weights (`*.pt`, `*.gguf`, `*.npy`).

---

## 3. Portion 5 Behavior & Active Module Integration

- **All 18 contract endpoints** (health, tactical state `< 100 KB` GeoJSON, building/incident/route queries, building label correction, responder feedback queue, explanations with P5 Validation Gate, MARL multi-unit dispatch, dry-run What-If counterfactuals, and commander `/approve` + `/reject` SHA-256 gates) operate live against `data/poc_mission.db`.
- **`POST /api/assessment/upload`** invokes `damage/assess.py` using pretrained `data/models/unet_resnet18_best.pth` when T0/T1 imagery is uploaded (and returns `status: "pending"` on metadata-only requests without fabricating detections).
- **`POST /api/routes/dispatch`** and **`POST /api/routes/what-if`** execute live **Portion 2 (`routing/generate.py`)** PPO + deterministic safety-mask routing (`< 500 ms` warmed reroute), while **`GET /api/explanations/*`** and async dispatch narration execute live **Portion 3 (`explain/`)** KernelSHAP, Grad-CAM, and Section 7 GTF-validated Mistral-7B translation (enforcing `HTTP 501` guards if a plugin is absent).
- **`POST /api/feedback`** and **`GET /api/adaptation/*`** execute live **Portion 4 (`feedback/`)** validation, multi-signal drift detection, active learning label selection, non-FIFO dual-buffer replay, Tier A frozen-backbone adapter training, and fail-closed 5-check safety gating.

---

## 4. Repository Structure

```text
├── main.py                  # P5: FastAPI 18-endpoint spine, Validation Gate, 501 shells & server entrypoint
├── models.py                # P5: Pydantic V2 schemas (4 shared objects, Sec. 4.2 road attrs, 18 HTTP contracts)
├── db.py                    # P5: SQLite WAL (data/poc_mission.db), 7 tables, SHA-256 Merkle chain & 24 helpers
├── seed.py                  # P5: Standalone Houston-East sector seeder (220 buildings, 26 edges, baseline route)
├── static/
│   ├── index.html           # P5: 6-in-1 Tactical Web Command Console markup
│   ├── style.css            # P5: Tactical command design system, Leaflet dark UI & SVG pulse styles
│   ├── app.js               # P5: Interactive Leaflet v1.9.4 Canvas console, <50ms timer & HTTP 501 handling
│   ├── leaflet.js           # P5: Offline Leaflet v1.9.4 library bundle
│   ├── leaflet.css          # P5: Offline Leaflet v1.9.4 stylesheet
│   └── images/              # P5: Offline Leaflet marker/layer image assets
├── damage/                  # P1 (Ammar): Integrated pretrained ResNet-18 U-Net + Grad-CAM module
├── routing/                 # P2 (Negm): graph, safety mask, PPO env/train, features & dispatch/what-if engine
├── explain/                 # P3 (Yousef): Grad-CAM overlays, 7-feature KernelSHAP, GTF validator & Mistral-7B translator
├── feedback/                # P4 (Abdulrahman): Drift monitor, active learning, dual replay, Tier A adapter & safety gate
├── data/
│   ├── poc_mission.db       # Pre-seeded Day-1 SQLite WAL database
│   ├── exports/.gitkeep
│   ├── models/.gitkeep
│   ├── osm/.gitkeep
│   └── xbd/.gitkeep
├── docs/
│   └── api.md               # Contract API specification & teammate plug-in hooks
├── scripts/
│   ├── download_osm.py      # 0 B stub
│   └── download_xbd.py      # 0 B stub
├── tests/
│   ├── test_poc_vertical_slice.py  # P5 QA, schema, SHA-256, Validation Gate & UI integration suite
│   └── test_unit_phases.py         # Granular unit tests across Phases 1–6
├── requirements.txt
└── .gitignore
```
