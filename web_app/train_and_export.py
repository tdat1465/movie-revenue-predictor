from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer

from feature_engineering import POSTER_BASE_COLS, load_json, save_json


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "movies_dataset_enriched.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _parse_list_safe(x: Any) -> List[str]:
    if pd.isna(x) or str(x).strip() == "":
        return []
    if isinstance(x, str):
        return [i.strip() for i in x.split(",") if i.strip()]
    return []


def _parse_collection_to_list(x: Any) -> List[str]:
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [str(x).strip()]


def time_based_target_encoding(
    df_sorted: pd.DataFrame,
    list_col_name: str,
    target_col: str,
    *,
    alpha: float = 10,
) -> List[float]:
    """Notebook-faithful: sequential history encoding with 0.7*max + 0.3*mean aggregation."""

    global_mean = float(df_sorted[target_col].mean())
    history: Dict[str, Dict[str, float]] = {}
    feature_values: List[float] = []

    for _, row in df_sorted.iterrows():
        current_items = row[list_col_name]
        target_val = float(row[target_col])

        stats: List[float] = []
        for item in current_items:
            if item in history:
                rec = history[item]
                mean_val = (rec["sum"] + alpha * global_mean) / (rec["count"] + alpha)
                stats.append(float(mean_val))
            else:
                stats.append(float(global_mean))

        if stats:
            score = 0.7 * float(np.max(stats)) + 0.3 * float(np.mean(stats))
        else:
            score = float(global_mean)

        feature_values.append(float(score))

        if target_val > 0:
            for item in current_items:
                if item not in history:
                    history[item] = {"sum": 0.0, "count": 0.0}
                history[item]["sum"] += float(target_val)
                history[item]["count"] += 1.0

    return feature_values


def prepare_features(df_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = df_input.copy()

    df["budget_raw"] = pd.to_numeric(df.get("budget"), errors="coerce")
    df["revenue_raw"] = pd.to_numeric(df.get("revenue"), errors="coerce")

    # target: log1p(revenue)
    df["revenue"] = np.log1p(df["revenue_raw"].clip(lower=0))

    # release date features
    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_dayofweek"] = df["release_date"].dt.dayofweek
    df["release_quarter"] = df["release_date"].dt.quarter
    df["is_weekend"] = df["release_dayofweek"].apply(lambda x: 1 if pd.notna(x) and x >= 5 else 0)
    df["is_blockbuster_season"] = df["release_month"].apply(
        lambda x: 1 if pd.notna(x) and x in [5, 6, 7, 11, 12] else 0
    )

    # list-like columns
    list_cols = [
        "genres",
        "cast",
        "production_companies",
        "production_countries",
        "director",
        "keywords",
    ]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list_safe)
        else:
            df[col] = [[] for _ in range(len(df))]

    # franchise
    if "collection" in df.columns:
        df["is_franchise"] = df["collection"].notna().astype(int)
    else:
        df["is_franchise"] = 0

    # runtime
    if "runtime" in df.columns:
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
        df["runtime"] = df["runtime"].replace(0, np.nan)
        df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    else:
        df["runtime"] = np.nan

    # budget
    df["budget_raw"] = df["budget_raw"].replace(0, np.nan)
    df["temp_genre"] = df["genres"].apply(lambda x: x[0] if len(x) > 0 else "Unknown")
    budget_medians = df.groupby(["release_year", "temp_genre"])["budget_raw"].transform("median")
    df["budget_raw"] = df["budget_raw"].fillna(budget_medians).fillna(df["budget_raw"].median())
    df["budget"] = np.log1p(df["budget_raw"].clip(lower=0))

    # poster features (if exist)
    poster_cols = POSTER_BASE_COLS
    if all(c in df.columns for c in poster_cols):
        for c in poster_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
        df["poster_warmth"] = df["poster_dom_r"] - df["poster_dom_b"]
        total_intensity = (df["poster_dom_r"] + df["poster_dom_g"] + df["poster_dom_b"]).replace(0, 1)
        df["poster_red_ratio"] = df["poster_dom_r"] / total_intensity
        df["poster_green_ratio"] = df["poster_dom_g"] / total_intensity
        df["poster_blue_ratio"] = df["poster_dom_b"] / total_intensity
        df["poster_vividness"] = (df["poster_saturation"] / 255.0) * (df["poster_brightness"] / 255.0)

    # sort by time and compute time-based encodings
    df = df.sort_values("release_date").reset_index(drop=True)

    df["cast_score"] = time_based_target_encoding(df, "cast", "revenue", alpha=10)
    df["director_score"] = time_based_target_encoding(df, "director", "revenue", alpha=5)
    df["keyword_score"] = time_based_target_encoding(df, "keywords", "revenue", alpha=20)
    df["genre_score"] = time_based_target_encoding(df, "genres", "revenue", alpha=50)
    df["production_company_score"] = time_based_target_encoding(df, "production_companies", "revenue", alpha=10)
    df["country_score"] = time_based_target_encoding(df, "production_countries", "revenue", alpha=20)

    df["collection_list"] = df.get("collection", pd.Series([np.nan] * len(df))).apply(_parse_collection_to_list)
    df["collection_score"] = time_based_target_encoding(df, "collection_list", "revenue", alpha=1)

    # multi-hot for genres
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df["genres"])
    genres_df = pd.DataFrame(
        genres_encoded,
        columns=[f"genre_{c.replace(' ', '_')}" for c in mlb.classes_],
        index=df.index,
    )
    df = df.join(genres_df)

    # drop text/ids
    cols_to_drop = [
        "id",
        "title",
        "release_date",
        "genres",
        "cast",
        "production_companies",
        "production_countries",
        "keywords",
        "director",
        "original_language",
        "rating",
        "vote_count",
        "popularity",
        "collection_list",
        "collection",
        "temp_genre",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    df_model = df.drop(columns=cols_to_drop)
    y = df_model["revenue"]
    X = df_model.drop(columns=["revenue"])
    leakage_cols = [c for c in ["revenue_raw", "budget_raw"] if c in X.columns]
    if leakage_cols:
        X = X.drop(columns=leakage_cols)

    return df, X, y


def _build_item_stats(
    df_hist: pd.DataFrame, list_col: str, *, target_col: str = "revenue"
) -> tuple[Dict[str, List[float]], float]:
    """Build {item: [sum_target, count]} over full historical dataframe."""

    stats: Dict[str, List[float]] = {}
    for items, target_val in zip(df_hist[list_col], df_hist[target_col]):
        if items is None:
            continue
        for item in items:
            if not item:
                continue
            if item not in stats:
                stats[item] = [0.0, 0]
            stats[item][0] += float(target_val)
            stats[item][1] += 1

    return stats, float(df_hist[target_col].mean())


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(DATA_PATH)
    df_full, X, y = prepare_features(df_raw)

    train_size = int(len(df_full) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    print(f"RMSE (Log Scale): {rmse}")
    print(f"R2: {r2}")

    # Export prediction artifacts (mirror notebook prediction helpers)
    encoders: Dict[str, Dict[str, Any]] = {}
    for col, a in [
        ("cast", 10),
        ("director", 5),
        ("keywords", 20),
        ("genres", 50),
        ("production_companies", 10),
        ("production_countries", 20),
        ("collection_list", 1),
    ]:
        if col not in df_full.columns:
            continue
        stats, gmean = _build_item_stats(df_full, col, target_col="revenue")
        encoders[col] = {"stats": stats, "global_mean": float(gmean), "alpha": float(a)}

    poster_medians: Dict[str, float] = {}
    for c in POSTER_BASE_COLS:
        if c in df_raw.columns:
            poster_medians[c] = float(pd.to_numeric(df_raw[c], errors="coerce").median())

    feature_columns = list(X.columns)
    feature_medians = {k: float(v) for k, v in X.median(numeric_only=True).to_dict().items()}

    # Dropdown options for the app
    genre_options = sorted(
        {
            c[len("genre_"):].replace("_", " ")
            for c in feature_columns
            if c.startswith("genre_")
        }
    )
    # Prefer encoder-derived names (matches model encoding space)
    collection_options = []
    if "collection_list" in encoders:
        collection_options = sorted([k for k in encoders["collection_list"]["stats"].keys() if str(k).strip()])

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    save_json(ARTIFACTS_DIR / "feature_columns.json", feature_columns)
    save_json(ARTIFACTS_DIR / "feature_medians.json", feature_medians)
    save_json(ARTIFACTS_DIR / "encoders.json", encoders)
    save_json(ARTIFACTS_DIR / "poster_medians.json", poster_medians)
    save_json(ARTIFACTS_DIR / "genre_options.json", genre_options)
    save_json(ARTIFACTS_DIR / "collection_options.json", collection_options)

    print("Saved:")
    print(f"- {ARTIFACTS_DIR / 'model.joblib'}")
    print(f"- {ARTIFACTS_DIR / 'feature_columns.json'}")
    print(f"- {ARTIFACTS_DIR / 'feature_medians.json'}")
    print(f"- {ARTIFACTS_DIR / 'encoders.json'}")
    print(f"- {ARTIFACTS_DIR / 'poster_medians.json'}")
    print(f"- {ARTIFACTS_DIR / 'genre_options.json'}")
    print(f"- {ARTIFACTS_DIR / 'collection_options.json'}")


if __name__ == "__main__":
    main()
