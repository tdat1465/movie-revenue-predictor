import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import joblib

DATA_PATH = '../Final/movies_dataset_enriched.csv'
df_raw = pd.read_csv(DATA_PATH)

# --- 1. CÁC HÀM XỬ LÝ (GIỮ NGUYÊN) ---
def _parse_list_safe(x):
    if pd.isna(x) or str(x).strip() == '': return []
    if isinstance(x, str): return [i.strip() for i in x.split(',')]
    return []

def _parse_collection_to_list(x):
    if pd.isna(x) or str(x).strip() == '': return []
    return [str(x).strip()]

def time_based_target_encoding(df_sorted, list_col_name, target_col, alpha=10):
    global_mean = df_sorted[target_col].mean()
    history = {}
    feature_values = []
    
    for idx, row in df_sorted.iterrows():
        current_items = row[list_col_name]
        stats = []
        for item in current_items:
            if item in history:
                rec = history[item]
                mean_val = (rec['sum'] + alpha * global_mean) / (rec['count'] + alpha)
                stats.append(mean_val)
            else:
                stats.append(global_mean)
        
        # [TINH CHỈNH] Giảm tỷ trọng Max xuống, tăng Mean lên để ổn định hơn
        if stats:
            score = 0.6 * np.max(stats) + 0.4 * np.mean(stats) 
        else:
            score = global_mean
        feature_values.append(score)

        if row[target_col] > 0:
            for item in current_items:
                if item not in history: history[item] = {'sum': 0.0, 'count': 0.0}
                history[item]['sum'] += row[target_col]
                history[item]['count'] += 1.0
    return feature_values

def prepare_features(df_input):
    df = df_input.copy()

    # --- CLEANING ---
    df['revenue_raw'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
    df['budget_raw'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0)
    # Lọc chặt hơn: Revenue > 50k
    df = df[(df['revenue_raw'] > 50000) & (df['budget_raw'] > 5000)]
    df['revenue'] = np.log1p(df['revenue_raw'])

    # --- DATE FEATURES ---
    df['release_date'] = pd.to_datetime(df.get('release_date'), errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_quarter'] = df['release_date'].dt.quarter

    # --- [ĐỘT PHÁ 1] RELATIVE BUDGET + POLYNOMIAL ---
    # Budget tương đối so với năm đó
    yearly_stats = df.groupby('release_year')['budget_raw'].median().reset_index().rename(columns={'budget_raw': 'year_median_budget'})
    df = df.merge(yearly_stats, on='release_year', how='left')
    df['budget_relative'] = df['budget_raw'] / (df['year_median_budget'] + 1)
    
    # Budget bình phương (Mô phỏng hiệu ứng phi tuyến tính: tiền càng nhiều doanh thu tăng càng nhanh)
    df['budget'] = np.log1p(df['budget_raw'])
    df['budget_sq'] = df['budget'] ** 2 

    # --- LIST PARSING ---
    list_cols = ['genres', 'cast', 'production_companies', 'director', 'keywords']
    for col in list_cols:
        if col in df.columns: df[col] = df[col].apply(_parse_list_safe)
        else: df[col] = [[] for _ in range(len(df))]
    
    if 'collection' in df.columns:
        df['is_franchise'] = df['collection'].notna().astype(int)
    else: df['is_franchise'] = 0

    if 'runtime' in df.columns:
        df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(df['runtime'].median())

    # --- POSTER ---
    poster_cols = ['poster_brightness', 'poster_saturation', 'poster_dom_r', 'poster_dom_g', 'poster_dom_b']
    if all(c in df.columns for c in poster_cols):
        for c in poster_cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(127.0)

    # --- ENCODING ---
    df = df.sort_values('release_date').reset_index(drop=True)
    df['cast_score'] = time_based_target_encoding(df, 'cast', 'revenue', alpha=5) # Tăng alpha lên 5 để bớt nhiễu
    df['director_score'] = time_based_target_encoding(df, 'director', 'revenue', alpha=5)
    df['genre_score'] = time_based_target_encoding(df, 'genres', 'revenue', alpha=20)
    df['company_score'] = time_based_target_encoding(df, 'production_companies', 'revenue', alpha=10)

    # --- [ĐỘT PHÁ 2] K-MEANS CLUSTERING (GOM NHÓM PHIM) ---
    # Ta sẽ gom nhóm các phim dựa trên: Budget, Runtime và Year
    # Mục đích: Giúp model biết "Phim này thuộc nhóm bom tấn dài" hay "Phim ngắn chi phí thấp"
    print("Đang thực hiện phân cụm K-Means...")
    cluster_features = df[['budget', 'runtime', 'release_year']].fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    
    # Gom thành 8 nhóm phim điển hình
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['movie_cluster'] = kmeans.fit_predict(scaled_features)
    
    # --- [ĐỘT PHÁ 3] INTERACTION ---
    df['budget_x_cast'] = df['budget'] * df['cast_score']
    df['budget_x_cluster'] = df['budget'] * df['movie_cluster'] # Tương tác giữa tiền và nhóm phim

    # Multi-hot Genres
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df['genres'])
    genres_df = pd.DataFrame(genres_encoded, columns=[f"genre_{c.replace(' ', '_')}" for c in mlb.classes_], index=df.index)
    df = df.join(genres_df)

    # Clean up
    cols_drop = ['id','title','release_date','genres','cast','production_companies','production_countries',
                 'keywords','director','original_language','rating','vote_count','popularity',
                 'collection_list','collection','temp_genre', 'revenue_raw', 'budget_raw', 'year_median_budget']
    
    df_model = df.drop(columns=[c for c in cols_drop if c in df.columns])
    
    return df, df_model.drop(columns=['revenue']), df_model['revenue'], kmeans, scaler

# --- CHẠY QUY TRÌNH ---
print("🚀 Đang xử lý dữ liệu với Clustering & Polynomials...")
df_full, X, y, kmeans_model, scaler_model = prepare_features(df_raw)
print(f"Features: {X.shape[1]}")

# --- CHIẾN THUẬT SPLIT MỚI: RANDOM SPLIT ---
# Nếu Project không bắt buộc phải split theo thời gian (TimeSeriesSplit), 
# hãy dùng Random Split để kiểm tra khả năng học thực sự của model.
# Time-series split thường cho kết quả thấp hơn do sự thay đổi của thị trường (VD: COVID).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)

print("🚀 Đang huấn luyện XGBoost (Balanced Mode)...")

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=2000,
    learning_rate=0.02,      
    max_depth=6,             
    min_child_weight=10,     # Tăng cao lên để CHỐNG OVERFITTING (Quan trọng)
    subsample=0.8,
    colsample_bytree=0.7,
    gamma=0.5,               # Tăng gamma để cắt tỉa nhánh cây thừa
    reg_alpha=2.0,           # Tăng L1 Regularization
    reg_lambda=5.0,          # Tăng L2 Regularization
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=200
)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print(f"🔥 FINAL RMSE: {rmse:.4f}")
print(f"🔥 FINAL R2: {r2:.4f}")

# --- CHẠY ĐOẠN NÀY CUỐI FILE TRAIN.PY ĐỂ LƯU CÁC FILE CẦN THIẾT ---
import joblib

# 1. Lưu từ điển Median Budget theo năm (Để tính Lạm phát/Relative Budget)
# yearly_stats là biến bạn đã tạo trong hàm prepare_features
# Nếu không truy cập được biến local, ta tính lại từ df_raw:
yearly_medians = df_raw.groupby(df_raw['release_date'].astype('datetime64[ns]').dt.year)['budget'].median().to_dict()
joblib.dump(yearly_medians, 'year_medians.pkl')

# 2. Lưu các từ điển điểm số (Cast, Director...)
def export_score_dict(df, col, target='revenue', fname='dict.pkl'):
    temp = df[[col, target]].explode(col)
    mapping = temp.groupby(col)[target].mean().to_dict()
    # Chuyển key về chữ thường
    mapping = {str(k).lower(): v for k, v in mapping.items()}
    joblib.dump(mapping, fname)

# df_full là DataFrame sau khi đã prepare_features
export_score_dict(df_full, 'cast', fname='cast_scores.pkl')
export_score_dict(df_full, 'director', fname='director_scores.pkl')
export_score_dict(df_full, 'production_companies', fname='company_scores.pkl')
export_score_dict(df_full, 'keywords', fname='keyword_scores.pkl')

print("✅ Đã xuất đủ 5 file .pkl và 3 file .joblib (model, kmeans, scaler)")