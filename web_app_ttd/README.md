# Web app: Movie Revenue Prediction

## 1) Cài dependencies

Từ thư mục repo:

```bash
pip install -r web_app/requirements.txt
```

## 2) Train + export model artifacts

```bash
python web_app/train_and_export.py
```

Artifacts được lưu ở `web_app/artifacts/`:
- `model.joblib`
- `feature_columns.json`
- `encoders.json`
- `poster_medians.json`

## 3) Chạy web app

```bash
streamlit run web_app/app.py
```

## Ghi chú
- App cho phép nhập thông tin phim + upload poster để trích xuất các feature màu sắc.
- Mô hình dự đoán `log1p(revenue)` và app sẽ convert ngược ra `revenue (USD)`.
