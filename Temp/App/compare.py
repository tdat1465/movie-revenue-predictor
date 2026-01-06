import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_3_models(models_dict, X_train, y_train, X_test, y_test, is_log_scale=True):
    """
    models_dict: Dictionary chứa 3 model. VD: {'XGBoost': model1, 'RandomForest': model2, 'Linear': model3}
    is_log_scale: Nếu True, sẽ chuyển ngược log về USD để tính MAE cho dễ hiểu.
    """
    results = []
    
    plt.figure(figsize=(18, 5))
    plot_idx = 1

    for name, model in models_dict.items():
        # 1. Dự báo
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 2. Tính toán Metrics (Trên Log Scale để so sánh khoa học)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_test = r2_score(y_test, y_test_pred)
        
        # Tính MAE thực tế (USD)
        if is_log_scale:
            mae_real = mean_absolute_error(np.expm1(y_test), np.expm1(y_test_pred))
        else:
            mae_real = mean_absolute_error(y_test, y_test_pred)

        # Lưu kết quả
        results.append({
            'Model': name,
            'RMSE Train': rmse_train,
            'RMSE Test': rmse_test,
            'Overfit Gap': rmse_test - rmse_train,
            'R2 Score': r2_test,
            'MAE (Real USD)': mae_real
        })

        # 3. Vẽ biểu đồ Residuals (Quan trọng nhất để bắt bệnh)
        plt.subplot(1, 3, plot_idx)
        residuals = y_test - y_test_pred
        sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'{name} Residuals')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plot_idx += 1

    plt.tight_layout()
    plt.show()
    
    # Hiển thị bảng so sánh
    df_results = pd.DataFrame(results).set_index('Model')
    
    # Tô màu để dễ nhìn: Min là tốt (RMSE, MAE), Max là tốt (R2)
    return df_results.style.highlight_min(subset=['RMSE Test', 'MAE (Real USD)', 'Overfit Gap'], color='lightgreen')\
                           .highlight_max(subset=['R2 Score'], color='lightgreen')

# --- CÁCH SỬ DỤNG ---
# Giả sử bạn đã train xong 3 model: xgb_model, rf_model, linear_model
models = {
    'XGBoost Tuned': model,  # Model tốt nhất của bạn
    'XGBoost Basic': model_cu, # Model cũ chưa chỉnh
    'Baseline': dummy_model    # Model đoán toàn số trung bình
}

evaluate_3_models(models, X_train, y_train, X_test, y_test)