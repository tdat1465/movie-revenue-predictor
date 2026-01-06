from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# --- 1. CẤU HÌNH & LOAD RESOURCES ---

def load_resources():
    resources = {}
    try:
        resources['model'] = joblib.load('xgb_model.joblib')
        resources['kmeans'] = joblib.load('kmeans.joblib')
        resources['scaler'] = joblib.load('scaler.joblib')
        
        # Load các file phụ trợ (nếu có)
        if os.path.exists('year_medians.pkl'): resources['year_medians'] = joblib.load('year_medians.pkl')
        if os.path.exists('cast_scores.pkl'): resources['cast_score'] = joblib.load('cast_scores.pkl')
        if os.path.exists('director_scores.pkl'): resources['director_score'] = joblib.load('director_scores.pkl')
        if os.path.exists('company_scores.pkl'): resources['company_score'] = joblib.load('company_scores.pkl')
        
        print("✅ Đã tải Model và Resources thành công!")
    except Exception as e:
        print(f"⚠️ Cảnh báo: {str(e)}")
    return resources

RESOURCES = load_resources()

# DANH SÁCH FEATURE CHUẨN (Lấy từ thông báo lỗi của bạn - Model Expects)
# Tuyệt đối không thêm bớt dòng nào ở đây
MODEL_FEATURES = [
    'budget', 'runtime', 'poster_brightness', 'poster_saturation', 'poster_dom_r', 
    'poster_dom_g', 'poster_dom_b', 'release_year', 'release_month', 'release_quarter', 
    'budget_relative', 'budget_sq', 'is_franchise', 
    'cast_score', 'director_score', 'genre_score', 'company_score', # Lưu ý: company_score chứ không phải production_company_score
    'movie_cluster', 'budget_x_cast', 'budget_x_cluster', 
    # Genre Columns
    'genre_Action', 'genre_Adventure', 'genre_Animation', 'genre_Comedy', 'genre_Crime', 
    'genre_Documentary', 'genre_Drama', 'genre_Family', 'genre_Fantasy', 'genre_History', 
    'genre_Horror', 'genre_Music', 'genre_Mystery', 'genre_Romance', 'genre_Science_Fiction', 
    'genre_TV_Movie', 'genre_Thriller', 'genre_War', 'genre_Western'
]

# --- 2. HÀM TRA CỨU ĐIỂM SỐ ---

def get_score(text_input, resource_key, default_val=13.0):
    if not text_input or str(text_input).strip() == "": return default_val
    items = [x.strip().lower() for x in text_input.split(',')]
    scores = []
    db = RESOURCES.get(resource_key, {})
    for item in items:
        scores.append(db.get(item, default_val))
    
    if scores:
        return 0.7 * np.max(scores) + 0.3 * np.mean(scores)
    return default_val

# --- 3. ROUTES FLASK ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'model' not in RESOURCES:
        return render_template('index.html', result=True, title="Lỗi", revenue="Chưa load được Model")

    try:
        # Khởi tạo DataFrame với đúng các cột Model cần (số 0.0)
        df = pd.DataFrame(0.0, index=[0], columns=MODEL_FEATURES)
        
        # --- A. CƠ BẢN ---
        title = request.form.get('title', 'Untitled')
        
        budget_raw = float(request.form.get('budget', 0))
        budget_log = np.log1p(budget_raw)
        df['budget'] = budget_log
        df['budget_sq'] = budget_log ** 2
        
        runtime = float(request.form.get('runtime', 90))
        df['runtime'] = runtime
        
        # --- B. THỜI GIAN ---
        date_str = request.form.get('release_date')
        if date_str:
            d = datetime.strptime(date_str, '%Y-%m-%d')
            year = d.year
            df['release_year'] = year
            df['release_month'] = d.month
            df['release_quarter'] = (d.month - 1) // 3 + 1
        else:
            year = 2026
            df['release_year'] = 2026
            df['release_month'] = 1
            df['release_quarter'] = 1

        # --- C. RELATIVE BUDGET ---
        year_medians = RESOURCES.get('year_medians', {})
        median_budget = year_medians.get(year, 10000000)
        df['budget_relative'] = budget_raw / (median_budget + 1)
        
        # --- D. SCORES ---
        df['cast_score'] = get_score(request.form.get('actors'), 'cast_score', 15.0)
        df['director_score'] = get_score(request.form.get('directors'), 'director_score', 14.0)
        
        # [QUAN TRỌNG] Model cần cột 'company_score', không phải 'production_company_score'
        df['company_score'] = get_score(request.form.get('production_companies'), 'company_score', 13.0)
        
        # Genre Score mặc định
        df['genre_score'] = 14.5

        # --- E. K-MEANS CLUSTERING ---
        if 'kmeans' in RESOURCES and 'scaler' in RESOURCES:
            # Input KMeans phải đúng thứ tự lúc train [budget_log, runtime, year]
            cluster_input = pd.DataFrame([[budget_log, runtime, year]], 
                                         columns=['budget', 'runtime', 'release_year'])
            scaled_input = RESOURCES['scaler'].transform(cluster_input)
            cluster_id = RESOURCES['kmeans'].predict(scaled_input)[0]
            
            df['movie_cluster'] = cluster_id
            df['budget_x_cluster'] = budget_log * cluster_id
        else:
            df['movie_cluster'] = 0
            df['budget_x_cluster'] = 0

        # --- F. POSTER ---
        # Chỉ điền 5 cột cơ bản mà model cần
        df['poster_brightness'] = 127.0
        df['poster_saturation'] = 127.0
        df['poster_dom_r'] = 127.0
        df['poster_dom_g'] = 127.0
        df['poster_dom_b'] = 127.0

        # --- G. GENRE & FRANCHISE ---
        genres_input = request.form.get('genres', '')
        for g in genres_input.split(','):
            clean_g = g.strip().replace(' ', '_')
            col_name = f'genre_{clean_g}'
            if col_name in MODEL_FEATURES:
                df[col_name] = 1.0

        df['is_franchise'] = 1.0 if request.form.get('franchise') else 0.0
        
        # Interaction cuối
        df['budget_x_cast'] = budget_log * df['cast_score']

        # --- H. DỰ BÁO ---
        # Lọc đúng các cột theo thứ tự Model yêu cầu
        final_input = df[MODEL_FEATURES]
        
        pred_log = RESOURCES['model'].predict(final_input)[0]
        revenue = np.expm1(pred_log)
        low = np.expm1(pred_log - 2.28)
        high = np.expm1(pred_log + 3.42)

        return render_template('index.html', 
                               result=True, title=title,
                               revenue=f"${revenue:,.0f}",
                               low=f"${low:,.0f}", high=f"${high:,.0f}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', result=True, title="Lỗi", revenue=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)