# Best parameters found: {'gamma': 0.25, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500}        
# Best cross-validation accuracy: 0.7538768634493577
# Test set accuracy: 0.7533064573166419

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("diabetes_50_50.csv")

# Separate target and features
diagnosis = df["Diabetes_binary"]
factors = df.drop(columns="Diabetes_binary")

# Train/test split
factors_train, factors_test, diagnosis_train, diagnosis_test = train_test_split(
    factors, diagnosis, test_size=0.2, stratify=diagnosis, random_state=0
)

# Base model
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=0
)

# Parameter grid for Grid Search
param_grid = {
    "n_estimators": [100, 500, 1000],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.5],
    "gamma": [0, 0.25, 0.5]
}

# Set up Grid Search with cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,            # 5-fold cross-validation
    verbose=2,       #will print intermediate procceses
    n_jobs=-1        # use all CPU cores - will probably make your computer very slow
)

# Run Grid Search
print("Starting Grid Search...")
grid_search.fit(factors_train, diagnosis_train)

# Show best parameters and score
print("\nBest parameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Test on held-out test set
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(factors_test)
test_accuracy = accuracy_score(diagnosis_test, test_predictions)
print("Test set accuracy:", test_accuracy)
