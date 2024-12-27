from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def RandomForest(X_train, y_train, X_test, y_test):
    # Define the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    # MAE
    mae = mean_absolute_error(y_test, y_pred)

    return rf_model, mae