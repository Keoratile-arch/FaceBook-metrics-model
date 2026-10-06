from ucimlrepo import fetch_ucirepo


facebook_metrics = fetch_ucirepo(id=368)


X = facebook_metrics.data.features
y = facebook_metrics.data.targets

import pandas as pd 

df=pd.read_csv("//Users/khutsosefolosha/Desktop/dataset_Facebook.csv")

import numpy as np 
import pandas as pd

X['Type'] = np.where(X['Type'] == 'Photo', 0
                     , np.where(X['Type'] == 'Status', 1
                                , np.where(X['Type'] == 'Link', 2, 3)))

print(X.isnull().sum())

y=y.rename(columns={'Total Interactions': 'Label'})

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                      train_size=0.7, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

#Linear regression :)
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

LR=make_pipeline(SimpleImputer(strategy="median"),LinearRegression())
LR.fit(X_train_scaled, y_train)

#Lasso Regression :)

from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

lasso=make_pipeline(SimpleImputer(strategy="median"),Lasso())
lasso.fit(X_train_scaled,y_train)

#Random forest :)
from sklearn.ensemble import RandomForestRegressor
rf=make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestRegressor(random_state=42)
)
rf.fit(X_train_scaled, y_train)

from sklearn.metrics import r2_score

LR_pred = LR.predict(X_test_scaled)
lasso_pred = lasso.predict(X_test_scaled)
rf_pred = rf.predict(X_test_scaled)

LR_r2 = r2_score(y_test, LR_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("Linear Regression R²:", LR_r2)
print("Lasso Regression R²:", lasso_r2)
print("Random Forest R²:", rf_r2)
