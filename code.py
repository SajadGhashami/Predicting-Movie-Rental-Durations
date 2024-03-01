import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# Import any additional modules and start coding below
from datetime import date

# Importing data
data = pd.read_csv('rental_info.csv', parse_dates = ["rental_date","return_date"])
# Take a look at the data
data.info()

# Create "rental_length_days"
format = "%Y-%m-%dT%H:%M:%S.%f"
data["rental_date"] = pd.to_datetime(data["rental_date"], format= format)
data["return_date"] = pd.to_datetime(data["return_date"], format= format)
data["rental_length_days"] = (data["return_date"] - data["rental_date"]).dt.days

#df_rental["special_features"].str.contains("Deleted Scenes")
#data["special_features"].str.contains("deleted_scenes")
data["deleted_scenes"] = np.where(data["special_features"].str.contains("Deleted Scenes"), 1, 0 )
data["behind_the_scenes"] = np.where(data["special_features"].str.contains("Behind the Scenes"), 1, 0)

X = data.drop(columns = ["special_features", "rental_length_days", "rental_date", "return_date"])
y= data["rental_length_days"]
X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size= 0.2, stratify =y, random_state = 9)

model_dict = {
    'LogisticRegression' : LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor()
}
dict = {}
for name, model in model_dict.items():
    print (name, model)
    model.fit(X_train, y_train)
    Y_pred= model.predict(X_test)
    error= mean_squared_error(Y_test, Y_pred)
    dict[name] = error
    
#dict
final = pd.DataFrame(dict.items(), columns=["model","MSE"] ).sort_values("MSE")
best_model = DecisionTreeRegressor
best_mse = 2.176183713

lasso_model = Lasso()
lasso_param = {
    "alpha" : np.linspace(0.00001, 1, 20)
}
kf = KFold(n_splits= 10, random_state= 9, shuffle=True)
lasso_cv = GridSearchCV(lasso_model, param_grid= lasso_param, cv = kf)
lasso_cv.fit(X_train, y_train)
best_model = lasso_cv.best_estimator_
y_pred = best_model.predict(X_test)
error= mean_squared_error(Y_test, y_pred)
print(error)

check = best_model.get_params()
check
