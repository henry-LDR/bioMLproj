#high score = Average Accuracy over 3 runs: 0.7552 
#parameters = Estimators: 5000  Depth: 5  Learning Rate: 0.05  Gamma: 0.47


import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



#Two/Three modes: Ensemble strategies like XGBoost and Random Forest (which are similar)
#A neural network.

#Data import: Take all of the data from the csv and read it into a Pandas DataFrame
df=pd.read_csv("diabetes_5050.csv")
#Diagnosis is 1 or 0 (diabetes or not)
diagnosis=df["Diabetes_binary"]
#Factors is everything else (BP, BMI, etc.)
factors=df.drop(columns="Diabetes_binary")
#Data cleaning: It's not necessary for this because there are no NaNs, but this is 
#not very realistic and we should anticipate the need to do so in the future.
#print(df.isnull().any().any())

factors_train,factors_test,diagnosis_train,diagnosis_test=train_test_split(factors, diagnosis, test_size=0.2,stratify=diagnosis)

# XGBoost parameters
estimators = 5000
depth = 5
learning_rate = 0.05 #scales the contribution of each new tree - 
gamma_value = 0.49    #determines if split is made based on loss reduction value - default = 0  - higher value helpts to prevent overfitting
accuracies = []

for run in range(3):
    # Create new train/test split each time for variation
    factors_train, factors_test, diagnosis_train, diagnosis_test = train_test_split(
        factors, diagnosis, test_size=0.2, stratify=diagnosis, random_state= run
    )

    # Initialize model
    diabetes_xgb = XGBClassifier(
        n_estimators=estimators,
        max_depth=depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=run,
        gamma = gamma_value
    )

    print(f"Training XGBoost model - run {run+1}")
    diabetes_xgb.fit(factors_train, diagnosis_train)
    print("Model trained successfully")

    # Predict and calculate accuracy
    results_test = diabetes_xgb.predict(factors_test)
    accuracy = accuracy_score(diagnosis_test, results_test)
    print(f"Run {run+1} Accuracy: {accuracy:.4f}")

    accuracies.append(accuracy)

avg_accuracy = sum(accuracies) / len(accuracies)
print(f"\nAverage Accuracy over 3 runs: {avg_accuracy:.4f} Estimators: {estimators}  Depth: {depth}  Learning Rate: {learning_rate}  Gamma: {gamma_value}")

# importances = diabetes_xgb.feature_importances_
# features = factors.columns
# importance_df = pd.Series(importances, index=features).sort_values(ascending=True)

# # Plot
# plt.figure(figsize=(10, 6))
# importance_df.plot(kind='barh')
# plt.title("Feature Importances (Random Forest)")
# plt.xlabel("Importance Score")
# plt.tight_layout()
# plt.show()

