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