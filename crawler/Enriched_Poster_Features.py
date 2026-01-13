import pandas as pd
import requests
import numpy as np
import os
import time
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
INPUT_FILE = "data/raw/movies_dataset_revenue.csv"       # File CSV bạn đã cào
OUTPUT_FILE = "movies_dataset_enriched.csv"     # File kết quả
API_KEY = os.getenv("TMDB_API_KEY", "5bae744934d0a79c18c935e723ea8ac2")
IMG_BASE_URL = "https://image.tmdb.org/t/p/w185" # Dùng ảnh nhỏ (w185) để xử lý cho nhanh
MAX_WORKERS = 8  # Số luồng tải ảnh song song

# ==============================================================================
# HÀM XỬ LÝ ẢNH (CORE LOGIC)
# ==============================================================================
def extract_poster_features(img_url):
    """
    Tải ảnh từ URL và trích xuất đặc trưng màu sắc/ánh sáng.
    Trả về dictionary các features.
    """
    try:
        response = requests.get(img_url, timeout=5)
        if response.status_code != 200:
            return None
        
        # Mở ảnh từ bytes trong RAM (không cần lưu ra ổ cứng)
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGB") # Đảm bảo là hệ màu RGB
        
        # Resize nhỏ lại để tính toán nhanh hơn (ví dụ 50x75 pixels)
        # Việc này không làm sai lệch nhiều về màu sắc chủ đạo nhưng tăng tốc độ 100 lần
        img_small = img.resize((50, 75)) 
        img_array = np.array(img_small)
        
        # 1. Tính độ sáng (Brightness) và Độ bão hòa (Saturation)
        # Chuyển từ RGB sang HSV để tính toán chuẩn hơn
        img_hsv = img_small.convert("HSV")
        hsv_array = np.array(img_hsv)
        
        # Kênh V (Value/Brightness) là chỉ số thứ 2, S (Saturation) là chỉ số 1
        saturation = hsv_array[:, :, 1].mean()
        brightness = hsv_array[:, :, 2].mean()

        # 2. Tìm màu chủ đạo (Dominant Color) bằng K-Means Clustering
        # Reshape thành danh sách các điểm ảnh (pixel)
        pixels = img_array.reshape(-1, 3)
        
        # Dùng KMeans để tìm 1 cụm màu lớn nhất (k=1)
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[0] # [R, G, B]
        
        return {
            "poster_brightness": round(brightness, 2),
            "poster_saturation": round(saturation, 2),
            "poster_dom_r": int(dominant_color[0]),
            "poster_dom_g": int(dominant_color[1]),
            "poster_dom_b": int(dominant_color[2])
        }

    except Exception as e:
        # Nếu lỗi (ảnh hỏng, không tải được...), trả về None
        return None

# ==============================================================================
# HÀM GỌI API ĐỂ LẤY POSTER PATH
# ==============================================================================
def get_poster_path(movie_id):
    """
    Gọi API lấy thông tin chi tiết để lấy poster_path
    (Vì file CSV cũ của bạn chưa lưu đường dẫn poster)
    """
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            return r.json().get("poster_path")
        return None
    except:
        return None

# ==============================================================================
# MAIN PROCESS
# ==============================================================================
def process_row(row):
    """
    Hàm xử lý cho từng dòng dữ liệu (được gọi bởi ThreadPool)
    """
    movie_id = row.get("id")
    
    # 1. Lấy đường dẫn poster (nếu chưa có trong CSV thì phải gọi API)
    # Giả sử file CSV hiện tại chưa có cột 'poster_path'
    poster_path = get_poster_path(movie_id)
    
    features = {
        "poster_brightness": np.nan,
        "poster_saturation": np.nan,
        "poster_dom_r": np.nan,
        "poster_dom_g": np.nan,
        "poster_dom_b": np.nan
    }
    
    if poster_path:
        full_url = IMG_BASE_URL + poster_path
        extracted = extract_poster_features(full_url)
        if extracted:
            features = extracted
            
    # Gộp features mới vào dòng dữ liệu cũ
    return {**row, **features}

def main():
    print("=== BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG POSTER ===")
    
    # 1. Đọc dữ liệu cũ
    if not os.path.exists(INPUT_FILE):
        print(f"Không tìm thấy file {INPUT_FILE}")
        return
        
    df = pd.read_csv(INPUT_FILE)
    print(f"Đã tải {len(df)} dòng phim.")
    
    # Chuyển đổi DataFrame thành list of dicts để xử lý đa luồng dễ hơn
    rows = df.to_dict("records")
    new_rows = []
    
    processed_count = 0
    start_time = time.time()
    
    # 2. Xử lý song song (Multithreading)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit các task
        future_to_row = {executor.submit(process_row, row): row for row in rows}
        
        for future in as_completed(future_to_row):
            result_row = future.result()
            new_rows.append(result_row)
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Đã xử lý: {processed_count}/{len(df)} phim...")

    # 3. Tạo DataFrame mới và Lưu
    df_result = pd.DataFrame(new_rows)
    
    # Sắp xếp lại thứ tự cột cho đẹp nếu cần
    cols = list(df.columns) + ["poster_brightness", "poster_saturation", "poster_dom_r", "poster_dom_g", "poster_dom_b"]
    # Chỉ giữ lại các cột có trong dữ liệu thực tế
    final_cols = [c for c in cols if c in df_result.columns]
    df_result = df_result[final_cols]
    
    df_result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    duration = time.time() - start_time
    print(f"\n=== HOÀN TẤT TRONG {duration:.2f} GIÂY ===")
    print(f"File mới đã lưu tại: {OUTPUT_FILE}")
    print("Preview dữ liệu:")
    print(df_result[["title", "poster_brightness", "poster_saturation", "poster_dom_r"]].head())

if __name__ == "__main__":
    main()