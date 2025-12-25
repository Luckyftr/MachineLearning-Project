from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def tune_models(X_train, y_train):
    models = {}

    print("\n=== Hyperparameter Tuning Running... ===")

    dt_params = {"max_depth": [3,5,7,None]}
    dt = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, scoring="f1", cv=5)
    dt.fit(X_train, y_train)
    models["Decision Tree"] = dt.best_estimator_

    knn_params = {"n_neighbors": [3,5,7,9]}
    knn = GridSearchCV(KNeighborsClassifier(), knn_params, scoring="f1", cv=5)
    knn.fit(X_train, y_train)
    models["KNN"] = knn.best_estimator_

    xgb_params = {"n_estimators": [100,200], "max_depth": [3,5], "learning_rate": [0.05,0.1]}
    xgb = GridSearchCV(
        XGBClassifier(eval_metric="logloss"),
        xgb_params, scoring="f1", cv=5
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb.best_estimator_

    print("\nBest Models Selected")
    return models
