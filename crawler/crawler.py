import os
import requests
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.getenv("API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
}

# ==========================
# Helper: safe request (retry)
# ==========================
def safe_get(url, params=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS)
            if r.status_code == 429:  # rate limit
                print("Rate limit hit → sleeping...")
                time.sleep(1.5)
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            print(f"[Error] {e} | retrying {attempt+1}/{max_retries}")
            time.sleep(1)

    return {}   # tránh None gây crash


# ==========================
# 1. Lấy danh sách 10.000 movie_id
# ==========================
def fetch_movie_list():
    movie_ids = []

    for page in range(1, 501):  # 500 pages ≈ 10k phim
        print(f"Fetching page {page} ...")
        
        r = safe_get(
            "https://api.themoviedb.org/3/discover/movie",
            params={
                "page": page,
                "primary_release_date.gte": "2000-01-01",
                "sort_by": "popularity.desc"
            }
        )

        results = r.get("results", [])
        for item in results:
            movie_ids.append(item["id"])

        time.sleep(0.20)  # tránh bị limit

    return movie_ids


# ==========================
# 2. Lấy dữ liệu đầy đủ 1 phim
# ==========================
def fetch_movie_data(movie_id):
    try:
        # --- Thông tin chính ---
        detail = safe_get(f"https://api.themoviedb.org/3/movie/{movie_id}")
        
        if not detail or "status_code" in detail:
            return None

        # --- Credits ---
        credits = safe_get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits")
        directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
        cast = [c["name"] for c in credits.get("cast", [])[:10]]

        # --- Keywords ---
        keywords_json = safe_get(f"https://api.themoviedb.org/3/movie/{movie_id}/keywords")
        keywords = ", ".join([k["name"] for k in keywords_json.get("keywords", [])])

        # Build object
        return {
            "id": movie_id,
            "title": detail.get("title"),
            "rating": detail.get("vote_average"),
            "vote_count": detail.get("vote_count"),
            "genres": ", ".join([g["name"] for g in detail.get("genres", [])]),

            "directors": ", ".join(directors),
            "actors": ", ".join(cast),

            "budget": detail.get("budget"),
            "revenue": detail.get("revenue"),
            "popularity": detail.get("popularity"),
            "runtime": detail.get("runtime"),

            "production_companies": ", ".join([c["name"] for c in detail.get("production_companies", [])]),
            "production_countries": ", ".join([c["name"] for c in detail.get("production_countries", [])]),
            "spoken_languages": ", ".join([c["english_name"] for c in detail.get("spoken_languages", [])]),

            "franchise": detail.get("belongs_to_collection", {}).get("name")
                         if detail.get("belongs_to_collection") else None,

            "keywords": keywords,
            "release_date": detail.get("release_date")
        }

    except Exception as e:
        print(f"Movie {movie_id} failed: {e}")
        return None


# ==========================
# 3. Đa luồng
# ==========================
def fetch_all_movies_multithread(movie_ids, max_workers=20, batch_save=500):
    results = []
    total = len(movie_ids)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_movie_data, mid): mid for mid in movie_ids}

        for idx, fut in enumerate(as_completed(futures), start=1):
            data = fut.result()
            if data:
                results.append(data)
                print(f"[{idx}/{total}] Done:", data["id"], "-", data["title"])

            # auto save mỗi 500 phim
            if idx % batch_save == 0:
                save_to_csv(results, "movies_partial.csv")
                print("Partial save (batch)")

    return results


# ==========================
# 4. Save CSV
# ==========================
def save_to_csv(data, filename="movies.csv"):
    if not data:
        return

    keys = data[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print("STEP 1: Fetching movie list...")
    ids = fetch_movie_list()

    print("STEP 2: Fetching movie data (multithread)...")
    movies = fetch_all_movies_multithread(ids, max_workers=20)

    print("STEP 3: Saving CSV...")
    save_to_csv(movies)

    print("DONE. Collected:", len(movies), "movies.")
