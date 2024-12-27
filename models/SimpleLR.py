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