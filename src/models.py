# src/models.py
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.metrics import calculate_regression_metrics
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_regressors():
    return {
        "SVR_RBF": SVR(kernel='rbf', C=60, gamma=0.001, epsilon=0.005),
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1),
        "MLP": MLPRegressor(hidden_layer_sizes=(512, 256), activation="relu",
                            learning_rate_init=1e-3, max_iter=200,
                            early_stopping=True, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8,
                                objective="reg:squarederror", random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, n_jobs=-1)
    }

def train_and_select_best(X_train, y_train, X_test, y_test):
    regressors = get_regressors()
    best_model, best_name, best_mse = None, None, float("inf")

    for name, reg in regressors.items():
        model = MultiOutputRegressor(reg) if name in ["SVR_RBF", "XGBoost", "LightGBM"] else reg
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, y_pred)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"🚀 {name}: MSE={mse:.4f}, R²={r2:.4f}")

        if mse < best_mse:
            best_mse, best_model, best_name = mse, model, name

        with open(f"output/{name}_results.txt", "w") as f:
            f.write(f"Results for {name}\n{'='*40}\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.6f}\n")

    with open("output/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"\n🎯 Best model: {best_name} (MSE={best_mse:.4f})")
    return best_model, best_name
