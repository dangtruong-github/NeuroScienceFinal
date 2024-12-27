import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def XGBoost(X_train, y_train, X_test, y_test):
    # Define the model with MAE as the objective
    xgb_model = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, random_state=42)

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_model.predict(X_test)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)

    return xgb_model, mae