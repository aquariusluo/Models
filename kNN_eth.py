import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import onnx
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# ✅ Step 1: Load and Preprocess Data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values properly
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df

# ✅ Step 2: Train-Test Split
def train_test_split(df):
    X = df.drop(columns=["target_ETHUSDT"])  # Features
    y = df["target_ETHUSDT"]  # Target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, scaler

# ✅ Step 3: Train kNN Model with Optimization
def train_knn(X_train, y_train):
    """
    Trains an optimized kNN model using GridSearchCV.
    Uses time series cross-validation for best hyperparameters.
    """
    print("\n🚀 Training kNN Model with Grid Search...")

    # Define hyperparameter grid
    param_grid = {
       "n_neighbors": [500, 750, 1000],  # Increase k further
       "weights": ["uniform", "distance"],  # Try both weighting strategies
       "metric": ["minkowski", "manhattan"]  # Check performance of Manhattan distance
    }
    # Define kNN model
    knn = KNeighborsRegressor()

    # Use time series cross-validation (5 splits)
    tscv = TimeSeriesSplit(n_splits=5)

    # Optimize model based on MAE
    grid_search = GridSearchCV(
        knn,
        param_grid,
        cv=tscv,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        n_jobs=-1,  # Use all CPU cores
        verbose=2  # Show detailed logs
    )

    # Train model
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_knn = grid_search.best_estimator_

    # Print best hyperparameters
    print(f"\n✅ Best k: {best_knn.n_neighbors}, Metric: {best_knn.metric}, Weighting: {best_knn.weights}")

    return best_knn

# ✅ Step 4: Evaluate Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"✅ Mean Absolute Error (MAE): {mae:.6f}")
    print(f"✅ Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"✅ R² Score: {r2:.6f}")

    return predictions

# ✅ Step 5: Convert to ONNX
def convert_to_onnx(model, input_dim, output_path="ethusdt_1hour_knn_model.onnx"):
    initial_type = [("candles", FloatTensorType([None, input_dim]))]
    onnx_model = onnxmltools.convert_sklearn(model, initial_types=initial_type)
    onnxmltools.utils.save_model(onnx_model, output_path)
    print(f"✅ ONNX model saved to {output_path}")
    return output_path

# ✅ Step 6: Verify ONNX Model
def verify_onnx_model(model_path, X_test, model):
    """
    Verify ONNX model against the trained kNN model using Mean Absolute Difference.
    """
    ort_session = ort.InferenceSession(model_path)
    test_subset = X_test[:5]
    ort_inputs = {"candles": test_subset.astype(np.float32)}

    onnx_output = ort_session.run(None, ort_inputs)[0].flatten()
    original_output = model.predict(test_subset)

    mean_diff = np.mean(np.abs(original_output - onnx_output))
    print("\n=== Verifying ONNX Model ===")
    print(f"✅ Original Model Predictions: {original_output[:5]}")
    print(f"✅ ONNX Model Predictions:     {onnx_output[:5]}")
    print(f"✅ Mean Absolute Difference: {mean_diff:.8f}")

    if mean_diff < 0.0001:
        print("✅ ONNX model predictions match closely!")
    else:
        print("❌ ONNX model predictions show noticeable deviation!")

# 🚀 Main Execution
if __name__ == "__main__":
    file_path = "../data/ETHUSDT_1h_spot_forecast_training_new.csv"
    
    df = load_data(file_path)
    X_train, X_test, y_train, y_test, scaler = train_test_split(df)

    knn_model = train_knn(X_train, y_train)
    
    evaluate_model(knn_model, X_test, y_test)

    onnx_model_path = convert_to_onnx(knn_model, input_dim=X_train.shape[1])

    verify_onnx_model(onnx_model_path, X_test, knn_model)
