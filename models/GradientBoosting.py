from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

def GradientBoosting(X_train, y_train, X_test, y_test):

    # Define the model
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    # Train the model
    gb_model.fit(X_train, y_train)

    y_pred = gb_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(mae)

    return gb_model, mae