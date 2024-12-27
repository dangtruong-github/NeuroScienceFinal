from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def DecisionTree(X_train, y_train, X_test, y_test):
    # Create and train the decision tree model
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = tree_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return tree_model, mae

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

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

def GridSearchRidge(X_train, y_train, X_test, y_test):
    # Define the model
    ridge = Ridge()

    # Define the parameter grid
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'solver': ['auto', 'svd', 'cholesky']
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    # best_params = grid_search.best_params_
    # best_score = -grid_search.best_score_

    # Evaluate the best model on the test set
    best_model_grid_search = grid_search.best_estimator_
    y_pred = best_model_grid_search.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return best_model_grid_search, mae

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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def SimpleLR(X_train, y_train, X_test, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # evaluate on train set
    y_pred = lr_model.predict(X_test)

    # MAE
    mae = mean_absolute_error(y_test, y_pred)
    
    return lr_model, mae

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