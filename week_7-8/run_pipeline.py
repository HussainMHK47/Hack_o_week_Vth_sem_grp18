import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score

housing = fetch_california_housing(as_frame=True)
df = housing.frame

features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
X = df[features]
y_reg = df['MedHouseVal']
y_clf = (y_reg > y_reg.quantile(0.70)).astype(int)

X_train, X_test, y_train_r, y_test_r, y_train_c, y_test_c = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("=== REGRESSION RESULTS ===")
lr = LinearRegression().fit(X_train_s, y_train_r)
p_lr = lr.predict(X_test_s)
print("Linear Reg    -> RMSE:", round(np.sqrt(mean_squared_error(y_test_r, p_lr)), 4), "| R2:", round(r2_score(y_test_r, p_lr), 3))

poly = PolynomialFeatures(degree=2, include_bias=False)
X_tr_p = poly.fit_transform(X_train_s)
X_te_p = poly.transform(X_test_s)

p_reg = LinearRegression().fit(X_tr_p, y_train_r)
p_poly = p_reg.predict(X_te_p)
print("Poly Reg (d2) -> RMSE:", round(np.sqrt(mean_squared_error(y_test_r, p_poly)), 4), "| R2:", round(r2_score(y_test_r, p_poly), 3))

ridge = Ridge(alpha=100.0).fit(X_tr_p, y_train_r)
p_ridge = ridge.predict(X_te_p)
print("Ridge Reg     -> RMSE:", round(np.sqrt(mean_squared_error(y_test_r, p_ridge)), 4), "| R2:", round(r2_score(y_test_r, p_ridge), 3))

lasso = Lasso(alpha=0.05, max_iter=5000).fit(X_tr_p, y_train_r)
p_lasso = lasso.predict(X_te_p)
print("Lasso Reg     -> RMSE:", round(np.sqrt(mean_squared_error(y_test_r, p_lasso)), 4), "| R2:", round(r2_score(y_test_r, p_lasso), 3))

print("\n=== CLASSIFICATION RESULTS ===")
clf = LogisticRegression().fit(X_train_s, y_train_c)
p_clf = clf.predict(X_test_s)
pr_clf = clf.predict_proba(X_test_s)[:, 1]
print("Logistic Reg  -> Acc:", round(accuracy_score(y_test_c, p_clf), 3), "| F1:", round(f1_score(y_test_c, p_clf), 3), "| ROC-AUC:", round(roc_auc_score(y_test_c, pr_clf), 3))

knn = KNeighborsClassifier(n_neighbors=7).fit(X_train_s, y_train_c)
p_knn = knn.predict(X_test_s)
pr_knn = knn.predict_proba(X_test_s)[:, 1]
print("KNN (k=7)     -> Acc:", round(accuracy_score(y_test_c, p_knn), 3), "| F1:", round(f1_score(y_test_c, p_knn), 3), "| ROC-AUC:", round(roc_auc_score(y_test_c, pr_knn), 3))