from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_engineering import (
    EncodingStats,
    POSTER_BASE_COLS,
    extract_poster_features_from_bytes,
    load_json,
    build_feature_row_for_movie,
)


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass
class AppArtifacts:
    model: Any
    feature_columns: list[str]
    encoders: dict[str, EncodingStats]
    poster_medians: dict[str, float]
    feature_medians: dict[str, float]
    genre_options: list[str]
    collection_options: list[str]


@st.cache_resource
def load_artifacts() -> AppArtifacts:
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    feature_columns = load_json(ARTIFACTS_DIR / "feature_columns.json")
    feature_medians = load_json(ARTIFACTS_DIR / "feature_medians.json")
    enc_raw = load_json(ARTIFACTS_DIR / "encoders.json")
    poster_medians = load_json(ARTIFACTS_DIR / "poster_medians.json")

    # Optional UI dropdown options
    genre_options_path = ARTIFACTS_DIR / "genre_options.json"
    collection_options_path = ARTIFACTS_DIR / "collection_options.json"
    if genre_options_path.exists():
        genre_options = list(load_json(genre_options_path))
    else:
        genre_options = sorted(
            {
                c[len("genre_"):].replace("_", " ")
                for c in feature_columns
                if str(c).startswith("genre_")
            }
        )
    if collection_options_path.exists():
        collection_options = list(load_json(collection_options_path))
    else:
        # Derive from encoder statistics if options file wasn't exported.
        # This makes the UI usable even when only core artifacts exist.
        collection_stats = enc_raw.get("collection_list", {}).get("stats", {})
        collection_options = sorted([str(k) for k in collection_stats.keys() if str(k).strip()])

    encoders: dict[str, EncodingStats] = {}
    for k, v in enc_raw.items():
        encoders[k] = EncodingStats(
            stats={ik: (float(iv[0]), int(iv[1])) for ik, iv in v["stats"].items()},
            global_mean=float(v["global_mean"]),
            alpha=float(v["alpha"]),
        )

    return AppArtifacts(
        model=model,
        feature_columns=list(feature_columns),
        encoders=encoders,
        poster_medians={k: float(v) for k, v in poster_medians.items()},
        feature_medians={k: float(v) for k, v in feature_medians.items()},
        genre_options=genre_options,
        collection_options=collection_options,
    )


def main() -> None:
    st.set_page_config(page_title="Movie Revenue Prediction", layout="centered")

    st.title("Movie Revenue Prediction")
    st.write("Nhập thông tin phim và (tuỳ chọn) upload poster để dự đoán doanh thu.")

    if not (ARTIFACTS_DIR / "model.joblib").exists():
        st.error("Chưa có model. Hãy chạy `python web_app/train_and_export.py` trước.")
        st.stop()

    artifacts = load_artifacts()

    with st.form("movie_form"):
        title = st.text_input("Tiêu đề (optional)")
        release_date = st.text_input("Ngày phát hành (YYYY-MM-DD)", value="2026-07-15")

        budget = st.number_input("Ngân sách (USD)", min_value=0.0, value=80_000_000.0, step=1_000_000.0)
        runtime = st.number_input("Thời lượng (phút)", min_value=0.0, value=120.0, step=1.0)
        if artifacts.genre_options:
            default_genres = ["Action"] if "Action" in artifacts.genre_options else []
            genres_selected = st.multiselect(
                "Thể loại (Genre) (chọn nhiều)",
                options=artifacts.genre_options,
                default=default_genres,
            )
        else:
            genres_raw = st.text_input("Thể loại (comma-separated)", value="Action")
            genres_selected = [g.strip() for g in genres_raw.split(",") if g.strip()]
        cast = st.text_area("Diễn viên (comma-separated)", value="Tom Hanks, Scarlett Johansson")
        director = st.text_input("Đạo diễn", value="Christopher Nolan")
        production_companies = st.text_input("Công ty sản xuất (comma-separated)", value="Warner Bros. Pictures")
        production_countries = st.text_input("Quốc gia sản xuất (comma-separated)", value="United States of America")
        keywords = st.text_area("Từ khóa (comma-separated)", value="superhero, space, war")
        collection = st.selectbox(
            "Bộ sưu tập / Thương hiệu (tuỳ chọn)",
            options=[""] + artifacts.collection_options,
            index=0,
        )

        st.markdown("### Poster (tuỳ chọn)")
        poster_file = st.file_uploader("Tải lên ảnh poster", type=["png", "jpg", "jpeg", "webp"])

        submitted = st.form_submit_button("Dự đoán")

    if not submitted:
        return

    movie: Dict[str, Any] = {
        "title": title,
        "release_date": release_date,
        "budget": float(budget),
        "runtime": float(runtime),
        "genres": ", ".join(genres_selected),
        "cast": cast,
        "director": director,
        "production_companies": production_companies,
        "production_countries": production_countries,
        "keywords": keywords,
        "collection": collection,
    }

    if poster_file is not None:
        poster_bytes = poster_file.getvalue()
        feats = extract_poster_features_from_bytes(poster_bytes)
        movie.update(feats)

    X_row = build_feature_row_for_movie(
        movie,
        feature_columns=artifacts.feature_columns,
        encoders=artifacts.encoders,
        poster_medians=artifacts.poster_medians,
        feature_medians=artifacts.feature_medians,
    )

    # Fill any NaNs (should be rare) with 0/median to avoid model errors
    for c in X_row.columns:
        if X_row[c].dtype.kind in "biufc":
            X_row[c] = pd.to_numeric(X_row[c], errors="coerce")
            if X_row[c].isna().any():
                X_row[c] = X_row[c].fillna(float(0.0))

    pred_log = float(artifacts.model.predict(X_row)[0])
    pred_rev = float(np.expm1(pred_log))

    st.subheader("Prediction")
    st.metric("Predicted revenue (USD)", f"${pred_rev:,.0f}")

    if budget is not None:
        profit = pred_rev - float(budget)
        st.metric("Predicted profit (USD)", f"${profit:,.0f}")
        st.write(f"Predicted is profit? {'Yes' if profit > 0 else 'No'}")

    with st.expander("Debug: extracted poster features"):
        if poster_file is None:
            st.write("No poster uploaded")
        else:
            st.json({k: float(movie[k]) for k in POSTER_BASE_COLS if k in movie})


if __name__ == "__main__":
    main()
