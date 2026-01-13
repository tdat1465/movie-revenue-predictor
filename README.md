# Movie Revenue Prediction (Dự đoán Doanh thu Phim)

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Computation-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-red)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Fast_Boosting-green)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App_Framework-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
---

## Mục lục
1. [Giới thiệu Bài toán](#-giới-thiệu-bài-toán)
2. [Cấu trúc dự án](#cấu-trúc-dự-án)
3. [Cài đặt & Hướng dẫn chạy](#cài-đặt--hướng-dẫn-chạy)

---

## Giới thiệu Bài toán

Phim ảnh là một ngành kinh doanh rủi ro cao. Một bộ phim bom tấn có thể tiêu tốn hàng trăm triệu USD để sản xuất và quảng bá. Việc dự đoán sớm khả năng thành công về mặt tài chính giúp các nhà sản xuất. 

Dự án này xây dựng một hệ thống Machine Learning để dự đoán doanh thu phòng vé toàn cầu của một bộ phim trước khi ra mắt. Đặc biệt, mô hình kết hợp cả **thông tin truyền thống** (ngân sách, diễn viên, đạo diễn) và **đặc trưng thị giác từ Poster phim** (màu sắc, độ sáng, độ bão hòa) để đưa ra dự báo.


**Thách thức chính:** Doanh thu phim phụ thuộc vào nhiều yếu tố định tính (sức hút ngôi sao, sự hưởng ứng của khán giả, ...) và các yếu tố phi cấu trúc (hình ảnh poster, trailer).

Dự án này giải quyết bài toán hồi quy (Regression):
$$f(X) \to \text{Revenue (USD)}$$
Với $X$ bao gồm ngân sách, thời lượng, thể loại, ekip sản xuất, và các tham số màu sắc trích xuất từ poster.

Dữ liệu được thu thập từ **TMDB API** bao gồm các phim phát hành từ năm **2000** đến **2024**.

Các nhóm đặc trưng chính:

- `budget`: Ngân sách sản xuất.

- `title`: Tiêu đè bộ phim.

- `release_date`: Ngày phát hành.

- `revenue`: doanh thu - mục tiêu cần dự đoán.

- `runtime`: thời lượng của bộ phim.

- `rating`: Điểm đánh giá bộ phim.

- `vote_count`: Số lượt bình chọn.

- `popularity`: Độ phổ biến.

- `genres`: Thể loại của phim.

- `production_companies` và `production_countries`: Công ty/Đất nước sản xuất bộ phim.

- `director` và `cast`: Đạo diễn/Diễn viên
Poster (Visual Features).

- `keywords`: Từ khóa liên quan để tìm kiếm bộ phim.

- `collection`: Series mà bộ phim thuộc về, nếu có.

- Các đặc trưng về poster: `poster_brightness` - Độ sáng, 
`poster_saturation` - Độ rực rỡ màu sắc, ...


---

## Cấu trúc dự án

```text
movie-revenue-prediction/
├── README.md
├── crawler
│   ├── Enriched_Poster_Features.py           
│   ├── crawler.py
│   └── crawler2.py
├── data
│   ├── movies_dataset_enriched.csv
│   └── movies_dataset_revenue.csv
├── notebooks
│   ├── Data_Exploration.ipynb
│   ├── Data_Preprocessing.ipynb
│   └── Meaningful_Question.ipynb
├── requirements.txt
└── web_app
    ├── app.py
    ├── artifacts
    │   ├── encoders.json
    │   ├── feature_columns.json
    │   ├── feature_medians.json
    │   ├── model.joblib
    │   └── poster_medians.json
    ├── feature_engineering.py
    └── train_and_export.py
```

## Cài đặt & Hướng dẫn chạy

### 1. Clone project

```bash
# Clone repository
git clone https://github.com/tdat1465/movie-revenue-predictor.git
cd movie-revenue-prediction
```

### 2. Cài đặt các thư viện cần thiết

```bash
# Cài đặt thư viện
pip install -r requirements.txt
```

### 3. Chạy lần lượt các file trong folder /notenooks

```bash
Run All
```
Hoặc
```bash
Restart & Run All
```

### 4. (Optional) - Chạy web app dự đoán doanh thu

#### 4.1. Huấn luyện và Export mô hình

```bash
python web_app/train_and_export.py
```

Artifacts được lưu ở `web_app/artifacts/`:
- `model.joblib`
- `feature_columns.json`
- `encoders.json`
- `poster_medians.json`

#### 4.2. Chạy web app

```bash
python -m streamlit run web_app/app.py
```

#### Ghi chú
- App cho phép nhập thông tin phim + upload poster để trích xuất các feature màu sắc.
- Mô hình dự đoán `log1p(revenue)` và app sẽ convert ngược ra `revenue (USD)`.