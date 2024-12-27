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