from model_utils import (
    DecisionTree, GradientBoosting, GridSearchRidge,
    RandomForest, SimpleLR, XGBoost
)

def model_evaluation(X_train, y_train, X_test, y_test):
    _, mae_dt = DecisionTree(X_train, y_train, X_test, y_test)
    _, mae_gb = GradientBoosting(X_train, y_train, X_test, y_test)
    _, mae_gsr = GridSearchRidge(X_train, y_train, X_test, y_test)
    _, mae_rf = RandomForest(X_train, y_train, X_test, y_test)
    _, mae_lr = SimpleLR(X_train, y_train, X_test, y_test)
    _, mae_xgb = XGBoost(X_train, y_train, X_test, y_test)

    mae_data = {
        "DecisionTree": mae_dt, 
        "GradientBoosting": mae_gb,
        "GridSearchRidge": mae_gsr,
        "RandomForest": mae_rf,
        "SimpleLR": mae_lr,
        "XGBoost": mae_xgb
    }
    # Find the model with the minimum MAE
    min_key = min(mae_data, key=mae_data.get)
    min_value = mae_data[min_key]

    print(f"The model with the minimum MAE is {min_key} with an MAE of {min_value}")

