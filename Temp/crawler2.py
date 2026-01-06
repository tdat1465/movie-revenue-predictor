import os
import time
import csv
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ==============================================================================
# CẤU HÌNH (CONFIG)
# ==============================================================================
# Thay bằng API Key của bạn hoặc đặt trong biến môi trường
API_KEY = os.getenv("TMDB_API_KEY", "YOUR_API_KEY_HERE")

OUTPUT_FILE = "movies_dataset_revenue.csv"
START_YEAR = 2000
END_YEAR = 2024
PAGES_PER_YEAR = 25   # 25 trang * 20 phim = 500 phim doanh thu cao nhất mỗi năm
MAX_WORKERS = 10      # Số luồng xử lý song song

# Setup Log để dễ theo dõi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# HÀM GỌI API AN TOÀN (ROBUST REQUEST)
# ==============================================================================
def safe_get(url, params=None, max_retries=5):
    """
    Gửi request với cơ chế thử lại (retry) nếu gặp lỗi mạng hoặc Rate Limit (429).
    """
    if params is None:
        params = {}
    params["api_key"] = API_KEY

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 429:  # Rate Limit
                retry_after = int(response.headers.get("Retry-After", 1))
                logging.warning(f"Rate limit hit. Sleeping {retry_after}s...")
                time.sleep(retry_after + 0.5)
                continue
            
            else:
                # Các lỗi 4xx, 5xx khác
                logging.error(f"Request failed: {response.status_code} - {url}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {e}. Retrying {attempt+1}/{max_retries}...")
            time.sleep(1)
    
    return None

# ==============================================================================
# 1. LẤY DANH SÁCH ID THEO NĂM (STRATEGY: YEARLY REVENUE)
# ==============================================================================
def fetch_movie_ids_by_year(year):
    """
    Lấy danh sách ID phim trong 1 năm, sắp xếp theo doanh thu giảm dần.
    Mục đích: Chỉ lấy những phim có dữ liệu doanh thu để train model.
    """
    movie_ids = []
    base_url = "https://api.themoviedb.org/3/discover/movie"
    
    for page in range(1, PAGES_PER_YEAR + 1):
        params = {
            "primary_release_year": year,
            "sort_by": "revenue.desc",  # QUAN TRỌNG: Ưu tiên phim có doanh thu
            "page": page,
            "vote_count.gte": 10        # Lọc bớt phim quá rác
        }
        data = safe_get(base_url, params)
        
        if not data or "results" not in data:
            break
            
        for item in data["results"]:
            movie_ids.append(item["id"])
            
        # Delay nhẹ giữa các trang discover
        time.sleep(0.2)
        
    logging.info(f"Năm {year}: Tìm thấy {len(movie_ids)} phim tiềm năng.")
    return movie_ids

# ==============================================================================
# 2. LẤY CHI TIẾT (FEATURE ENGINEERING CHO REVENUE PREDICTION)
# ==============================================================================
def fetch_movie_details(movie_id):
    """
    Lấy tất cả đặc trưng cần thiết chỉ trong 1 request nhờ 'append_to_response'.
    """
    # Kỹ thuật lấy gộp: credits (diễn viên), keywords (từ khóa), release_dates (ngày phát hành)
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"append_to_response": "credits,keywords,release_dates"}
    
    data = safe_get(url, params)
    
    if not data:
        return None

    # --- Lọc dữ liệu rác ---
    # Nếu doanh thu và ngân sách đều bằng 0 hoặc không có -> bỏ qua (hoặc giữ lại tùy chiến lược)
    # Ở đây ta giữ lại để xử lý sau, nhưng ưu tiên data sạch.
    revenue = data.get("revenue", 0)
    budget = data.get("budget", 0)
    
    # --- Trích xuất đặc trưng (Features Extraction) ---
    
    # 1. Đạo diễn & Diễn viên (Top 5)
    credits = data.get("credits", {})
    directors = [m["name"] for m in credits.get("crew", []) if m["job"] == "Director"]
    cast = [m["name"] for m in credits.get("cast", [])[:5]] # Lấy top 5 star power
    
    # 2. Keywords (Cực quan trọng cho Content-based)
    keywords = [k["name"] for k in data.get("keywords", {}).get("keywords", [])]
    
    # 3. Thông tin sản xuất
    production_companies = [c["name"] for c in data.get("production_companies", [])]
    production_countries = [c["name"] for c in data.get("production_countries", [])]
    genres = [g["name"] for g in data.get("genres", [])]
    
    # 4. Ngày phát hành chuẩn (Tại Mỹ - US release thường quan trọng nhất cho doanh thu)
    release_date = data.get("release_date", "")
    
    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "release_date": release_date,
        # "year": release_date[:4] if release_date else "",
        # "month": release_date[5:7] if len(release_date) >= 7 else "", # Feature mùa vụ
        
        # Target Variables
        "budget": budget,
        "revenue": revenue,
        
        # Numeric Features
        "runtime": data.get("runtime"),
        "rating": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "popularity": data.get("popularity"),
        
        # Categorical / Text Features
        "genres": ", ".join(genres),
        "production_companies": ", ".join(production_companies),
        "production_countries": ", ".join(production_countries),
        "director": ", ".join(directors),
        "cast": ", ".join(cast),
        "keywords": ", ".join(keywords),
        "original_language": data.get("original_language"),
        
        # Series phim (Harry Potter, Marvel...) ảnh hưởng lớn doanh thu
        "collection": data.get("belongs_to_collection", {}).get("name") if data.get("belongs_to_collection") else None
    }

# ==============================================================================
# 3. HÀM LƯU CSV
# ==============================================================================
def save_to_csv(data_list, filename, mode='a'):
    if not data_list:
        return
    
    file_exists = os.path.isfile(filename)
    keys = data_list[0].keys()
    
    with open(filename, mode, newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists or mode == 'w':
            writer.writeheader()
        writer.writerows(data_list)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f"=== BẮT ĐẦU CRAWL DỮ LIỆU ({START_YEAR}-{END_YEAR}) ===")
    
    # Xóa file cũ nếu muốn chạy lại từ đầu
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    total_collected = 0

    for year in range(START_YEAR, END_YEAR + 1):
        logging.info(f"--> Đang xử lý năm: {year}")
        
        # B1: Lấy list ID
        movie_ids = fetch_movie_ids_by_year(year)
        if not movie_ids:
            continue
            
        year_data = []
        
        # B2: Lấy chi tiết đa luồng
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_id = {executor.submit(fetch_movie_details, mid): mid for mid in movie_ids}
            
            for future in as_completed(future_to_id):
                try:
                    result = future.result()
                    # Chỉ lấy phim có Doanh thu > 0 (Để train tốt hơn)
                    if result and result['revenue'] > 0:
                        year_data.append(result)
                except Exception as e:
                    logging.error(f"Error fetching movie: {e}")

        # B3: Lưu ngay sau khi xong 1 năm (Checkpoint)
        if year_data:
            save_to_csv(year_data, OUTPUT_FILE, mode='a')
            count = len(year_data)
            total_collected += count
            logging.info(f"    Đã lưu {count} phim của năm {year}. Tổng cộng: {total_collected}")
        
        # Nghỉ một chút giữa các năm để API thở
        time.sleep(1)

    print(f"\n=== HOÀN TẤT. TỔNG SỐ PHIM: {total_collected} ===")
    print(f"File dữ liệu: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()