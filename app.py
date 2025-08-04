import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#import torch

#Two/Three modes: Ensemble strategies like XGBoost and Random Forest (which are similar)
#A neural network.

#Data import: Take all of the data from the csv and read it into a Pandas DataFrame
df=pd.read_csv("diabetes_data.csv")
#Diagnosis is 1 or 0 (diabetes or not)
diagnosis=df["Diabetes_binary"]
#Factors is everything else (BP, BMI, etc.)
factors=df.drop(columns="Diabetes_binary")
#Data cleaning: It's not necessary for this because there are no NaNs, but this is 
#not very realistic and we should anticipate the need to do so in the future.
#print(df.isnull().any().any())

#Create train/test split
factors_train,factors_test,diagnosis_train,diagnosis_test=train_test_split(factors, diagnosis, test_size=0.2,stratify=diagnosis)

#Ensemble strategies
#Random forest
RandomForestClassifier(n_estimators=200,criterion=entropy, )