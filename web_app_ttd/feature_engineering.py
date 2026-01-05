from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


POSTER_BASE_COLS = [
    "poster_brightness",
    "poster_saturation",
    "poster_dom_r",
    "poster_dom_g",
    "poster_dom_b",
]
POSTER_DERIVED_COLS = [
    "poster_warmth",
    "poster_red_ratio",
    "poster_green_ratio",
    "poster_blue_ratio",
    "poster_vividness",
]


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for v in value:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    if not s:
        return []
    # input data in csv uses comma-separated lists
    return [x.strip() for x in s.split(",") if x.strip()]


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and not x.strip():
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def extract_poster_features_from_bytes(
    image_bytes: bytes,
    *,
    resize_max_side: int = 256,
    k: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Extract poster color features.

    - brightness/saturation: mean of HSV V/S channels, scaled 0..255
    - dominant RGB: k-means cluster center with largest support

    Returns keys matching POSTER_BASE_COLS.
    """

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, float(resize_max_side) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))

        rgb = np.asarray(img, dtype=np.float32)
        flat = rgb.reshape(-1, 3)

        # HSV for brightness/saturation
        hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
        # HSV channels are 0..255 in PIL
        sat = float(np.mean(hsv[..., 1]))
        bri = float(np.mean(hsv[..., 2]))

        # k-means dominant color
        sample_n = min(20_000, flat.shape[0])
        if flat.shape[0] > sample_n:
            idx = np.random.RandomState(random_state).choice(flat.shape[0], size=sample_n, replace=False)
            sample = flat[idx]
        else:
            sample = flat

        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=2048, n_init="auto")
        labels = km.fit_predict(sample)
        counts = np.bincount(labels, minlength=k)
        dom_cluster = int(np.argmax(counts))
        dom_rgb = km.cluster_centers_[dom_cluster]

        return {
            "poster_brightness": bri,
            "poster_saturation": sat,
            "poster_dom_r": float(dom_rgb[0]),
            "poster_dom_g": float(dom_rgb[1]),
            "poster_dom_b": float(dom_rgb[2]),
        }


def add_derived_poster_features(row: Dict[str, Any]) -> None:
    r = _safe_float(row.get("poster_dom_r"))
    g = _safe_float(row.get("poster_dom_g"))
    b = _safe_float(row.get("poster_dom_b"))
    sat = _safe_float(row.get("poster_saturation"))
    bri = _safe_float(row.get("poster_brightness"))

    if r is None or g is None or b is None or sat is None or bri is None:
        return

    row["poster_warmth"] = float(r - b)

    total_intensity = float(r + g + b)
    if total_intensity == 0:
        total_intensity = 1.0

    row["poster_red_ratio"] = float(r / total_intensity)
    row["poster_green_ratio"] = float(g / total_intensity)
    row["poster_blue_ratio"] = float(b / total_intensity)
    row["poster_vividness"] = float((sat / 255.0) * (bri / 255.0))


@dataclass
class EncodingStats:
    stats: Dict[str, Tuple[float, int]]  # item -> (sum_target, count)
    global_mean: float
    alpha: float


def build_item_stats(df: pd.DataFrame, list_col: str, target_col: str) -> EncodingStats:
    table: Dict[str, List[float]] = {}

    target = pd.to_numeric(df[target_col], errors="coerce")
    for raw_items, y in zip(df[list_col].fillna(""), target):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        items = _as_list(raw_items)
        if not items:
            continue
        for item in items:
            table.setdefault(item, []).append(float(y))

    global_mean = float(target.dropna().mean())
    stats: Dict[str, Tuple[float, int]] = {}
    for k, vals in table.items():
        if not vals:
            continue
        stats[k] = (float(np.sum(vals)), int(len(vals)))

    return EncodingStats(stats=stats, global_mean=global_mean, alpha=10.0)


def score_items(items: Iterable[str], enc: EncodingStats) -> float:
    items_list = [str(x).strip() for x in items if str(x).strip()]
    if not items_list:
        return float(enc.global_mean)

    means: List[float] = []
    for item in items_list:
        entry = enc.stats.get(item)
        if entry is None:
            means.append(float(enc.global_mean))
            continue
        s, c = entry
        # smoothing
        v = (float(s) + enc.alpha * enc.global_mean) / (int(c) + enc.alpha)
        means.append(float(v))

    # Mirror notebook aggregation: 70% max + 30% mean
    return 0.7 * float(np.max(means)) + 0.3 * float(np.mean(means))


def compute_date_features(release_date: Any) -> Dict[str, Any]:
    if release_date is None or (isinstance(release_date, str) and not release_date.strip()):
        return {
            "release_year": np.nan,
            "release_month": np.nan,
            "release_dayofweek": np.nan,
            "release_quarter": np.nan,
            "is_weekend": 0,
        }

    dt = pd.to_datetime(release_date, errors="coerce")
    if pd.isna(dt):
        return {
            "release_year": np.nan,
            "release_month": np.nan,
            "release_dayofweek": np.nan,
            "release_quarter": np.nan,
            "is_weekend": 0,
        }

    year = int(dt.year)
    month = int(dt.month)
    dayofweek = int(dt.dayofweek)  # 0=Mon
    quarter = int((month - 1) / 3) + 1
    is_weekend = 1 if dayofweek >= 5 else 0

    return {
        "release_year": year,
        "release_month": month,
        "release_dayofweek": dayofweek,
        "release_quarter": quarter,
        "is_weekend": is_weekend,
    }


def is_blockbuster_season(month: Any) -> int:
    m = _safe_float(month)
    if m is None:
        return 0
    return 1 if int(m) in [5, 6, 7, 11, 12] else 0


def build_genre_columns(df: pd.DataFrame, genres_col: str = "genres") -> List[str]:
    genre_set: set[str] = set()
    for raw in df[genres_col].fillna(""):
        for g in _as_list(raw):
            genre_set.add(g)
    return [f"genre_{g.replace(' ', '_')}" for g in sorted(genre_set)]


def build_feature_row_for_movie(
    movie: Mapping[str, Any],
    *,
    feature_columns: List[str],
    encoders: Mapping[str, EncodingStats],
    poster_medians: Mapping[str, float],
    feature_medians: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Convert a movie dict into a single-row DataFrame aligned with training columns."""

    if feature_medians is None:
        row: Dict[str, Any] = {c: 0.0 for c in feature_columns}
    else:
        row = {c: float(feature_medians.get(c, 0.0)) for c in feature_columns}

    # numeric
    budget_raw = _safe_float(movie.get("budget"))
    if budget_raw is None:
        row["budget"] = np.nan
    else:
        row["budget"] = float(np.log1p(max(0.0, float(budget_raw))))

    runtime = _safe_float(movie.get("runtime"))
    if runtime is None:
        row["runtime"] = np.nan
    else:
        row["runtime"] = float(runtime)

    # time
    date_feats = compute_date_features(movie.get("release_date"))
    for k, v in date_feats.items():
        if k in row:
            row[k] = v
    if "is_blockbuster_season" in row:
        row["is_blockbuster_season"] = is_blockbuster_season(row.get("release_month"))

    # franchise
    collection = str(movie.get("collection") or "").strip()
    if "is_franchise" in row:
        row["is_franchise"] = 1 if collection else 0

    # poster base
    for c in POSTER_BASE_COLS:
        if c in row:
            val = _safe_float(movie.get(c))
            if val is None:
                row[c] = float(poster_medians.get(c, 0.0))
            else:
                row[c] = float(val)

    # poster derived
    add_derived_poster_features(row)

    # list inputs
    def _apply_score(key: str, out_col: str) -> None:
        if out_col not in row:
            return
        enc = encoders.get(key)
        if enc is None:
            return
        row[out_col] = float(score_items(_as_list(movie.get(key)), enc))

    _apply_score("cast", "cast_score")
    _apply_score("director", "director_score")
    _apply_score("keywords", "keyword_score")
    _apply_score("genres", "genre_score")
    _apply_score("production_companies", "production_company_score")
    _apply_score("production_countries", "country_score")

    # collection list can be empty; treat as single item
    if "collection_score" in row and "collection_list" in encoders:
        enc = encoders["collection_list"]
        items = [collection] if collection else []
        row["collection_score"] = float(score_items(items, enc))

    # genre multi-hot
    for g in _as_list(movie.get("genres")):
        col = f"genre_{g.replace(' ', '_')}"
        if col in row:
            row[col] = 1.0

    X_row = pd.DataFrame([row], columns=feature_columns)
    return X_row


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
