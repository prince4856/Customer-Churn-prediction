#-----------------------------------(Day-01)-------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets--
df=pd.read_csv(r'c:\Users\HP\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv')
# print(df)

# Display 1st 5 rows
print(df.head())

# Display last 5 row
print(df.tail())

print(df.shape)     # shape of dataset
print(df.size)       # size of dataset

print(df.info())    # Complete information of dataset
print(df.describe())  # statistical summary of dataset

# checking missing value
print(df.isnull().sum())

# check Duplicate row
print(df.duplicated().sum())

# Target Distribution
print(df['Churn'].value_counts())

# check percentage
print(df['Churn'].value_counts(normalize=True)*100)


#--------------------------------------(Day-02)-----------------------------------------------
print(df.columns)

# Remove customer id
df.drop("customerID",axis=1,inplace=True)
print(df.columns)

# Separate numerical and categorical column
numerical_col=df.select_dtypes(include=['number']).columns
categorical_col=df.select_dtypes(include=["object"]).columns

print("numerical column-",numerical_col)
print("categorical column-",categorical_col)

# check categorical value
for column in categorical_col:
    print(df[column].value_counts())


# Encode target---
df['Churn']=df['Churn'].map({
    "No":0,
    "Yes":1

})

# Final verification :--
print("cleaned datasets=",df.head())
print("Final shape=",df.shape)
print("Missing value=",df.isnull().sum())
print("Check duplicate rows=",df.duplicated().sum())
print("Churn distribution=",df["Churn"].value_counts())

# ==================================(Day-03, Exploratory Data Analytics(EDA))=========================
# EDA stands for Exploratory Data Analysis. 
# It is the process of analyzing and visualizing data to understand its distributions, relationships, patterns, and possible anomalies before model building.


# Calculate churn count and percentage :----
print(df["Churn"].value_counts())

print(df["Churn"].value_counts(normalize=True)*100)

# plot churn Distribution
sns.countplot(x="Churn",data=df)
plt.title("Customer churn Distribution")
plt.xlabel("Churn")
plt.ylabel("number of customer")
plt.show()

# Analyze contract vs churn
print(pd.crosstab(df['Contract'],df["Churn"],normalize="index")*100)

# Visualization
sns.countplot(x="Contract",hue="Churn",data=df)
plt.xlabel("contract")
plt.ylabel("nunber of customer")
plt.title("Contract vs churn")
plt.show()

# Analyze Tenure vs churn
sns.boxplot(x="Churn",y="tenure",data=df)
plt.title("Tenure vs churn")
plt.xlabel("churn")
plt.ylabel("tenure")
plt.show()

# mean (useful Analysis)
print(df.groupby("Churn")["tenure"].mean())


# Analyze monthly charges vs churn
sns.boxplot(x="Churn",y="MonthlyCharges",data=df)
plt.title("monthlyCharges vs churn")
plt.xlabel("Churn")
plt.ylabel("monthlyCharges")
plt.show()

# mean
print(df.groupby("Churn")["MonthlyCharges"].mean())

# Analyze internet services vs churn
print(pd.crosstab(df["InternetService"],df["Churn"],normalize="index")*100)

# Visualize
sns.countplot(x="InternetService",hue="Churn",data=df)
plt.title("Internet services vs churn")
plt.show()

#Analyze payment method vs churn
sns.countplot(x="PaymentMethod",hue="Churn",data=df)
plt.title("PaymentMethod vs Churn")
plt.xticks(rotation=30)
plt.show()

payment_churn = pd.crosstab(df["PaymentMethod"],df["Churn"],normalize="index") * 100
print(payment_churn)


# Create Numerical Correlation heatmap
numerical_df=df.select_dtypes(include="number")
print(numerical_df.corr())

# Visualization
plt.figure(figsize=(8,6))
sns.heatmap(numerical_df.corr(),annot=True,cmap='coolwarm')
plt.title("Correlation heatmap")
plt.show()


# check datatype
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].dtypes)

## Convert TotalCharges from string to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values created during conversion
print("Missing TotalCharges:", df["TotalCharges"].isnull().sum())

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Check datatype
print(df["TotalCharges"].dtype)

# print grouped numerical Summary
summary = df.groupby("Churn")[
    ["tenure", "MonthlyCharges", "TotalCharges"]
].mean()

print("\nGrouped Numerical Summary:")
print(summary)

# ============================================================
# DAY 03 - OBSERVATIONS
# ============================================================

# Observation 1:
# Month-to-month contract customers have the highest churn rate
# of 42.71%, while two-year contract customers have only 2.83% churn.

# Observation 2:
# Churned customers have an average tenure of 17.98 months,
# while non-churned customers have an average tenure of 37.57 months.

# Observation 3:
# Churned customers have higher average monthly charges (74.44)
# compared with non-churned customers (61.27).

# Observation 4:
# Fiber-optic customers have the highest churn rate of 41.89%,
# compared with DSL customers (18.96%) and customers with
# no internet service (7.40%).


#====================================(Day-04 , Topic :-Feature Engineering + Categorical Encoding + Feature/Target Separation)===

# Check categorical column
categorical_col = df.select_dtypes(include=["object"]).columns
print("Categorical column=",categorical_col)

# Check unique value
for column in categorical_col:
    print("Column=",df[column].unique())


# Binary Encoding :--
binary_columns=["Partner","Dependents","PhoneService","PaperlessBilling"]

for column in binary_columns:
    df[column]=df[column].map({
        "No":0,
        "Yes":1
    })

# Gender Encoding
df["gender"] = df["gender"].map({
    "Female":0,
    "Male":1
})

# Remaining categories 
remaining_categories=df.select_dtypes(include=["object"]).columns
print("Remaining categories=",remaining_categories)

# One Hot Encoding :-
df=pd.get_dummies(df, columns=remaining_categories,drop_first=True,dtype=int) 


# Verify Encoding
print("\nDataset after Encoding")
print(df.head())

print("Shape=",df.shape)

print("\n remaining object column")
print(df.select_dtypes(include=["object"]).columns)

print("\n Total missing value")
print(df.isnull().sum().sum())


# separate column
x= df.drop("Churn",axis=1)
y=df["Churn"]
print("x Shape=",x.shape)
print("y Shape=",y.shape)

# verify the target column
print("churn counts=",y.value_counts())
print("Churn percentage=",y.value_counts(normalize=True)*100)

#=====================(Day-05, Topic:-> Train-Test Split + Feature Scaling + First ML Model)=======

# select input feature and target variable
x=df.drop("Churn",axis=1)
y=df["Churn"]

# Train test split 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

print("X_train Shape:", x_train.shape)
print("X_test Shape:", x_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# --------------------------------
# 2. Check target distribution
# --------------------------------

print("\nTraining Target Percentage:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting Target Percentage:")
print(y_test.value_counts(normalize=True) * 100)


# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

# create model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)

# train the model on training dataset
model.fit(x_train_scaled,y_train)

# prediction
y_pred=model.predict(x_test_scaled)

# model Evaluation
from sklearn.metrics import accuracy_score ,classification_report,confusion_matrix
print("Accuracy=",accuracy_score(y_test,y_pred))
print("Classificatio report=",classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix:",cm)


# Visualization of confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm, annot=True,fmt="d")
plt.title("Confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show() 

#=====================(Day-05, Topic-> Multiple model Training and model Comparision)======
#
# Taking three model to comparision :-1.-Logistic Regression ,KNN,Random Forest

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

# Taking LogisticRegression as Baseline
log_model=LogisticRegression(max_iter=1000)

# Train the model
log_model.fit(x_train_scaled,y_train)

# make prediction
log_pred=log_model.predict(x_test_scaled)

# print Accuracy score
print("Logistic Accuracy scoe=",accuracy_score(y_test,log_pred))


# Train KNN
KNN_model=KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(x_train_scaled,y_train)

# make predictions
KNN_pred=KNN_model.predict(x_test_scaled)

# Evaluation
print("KNN Accuracy Score=",accuracy_score(y_test,KNN_pred))

# 3rd model 
rf=RandomForestClassifier(n_estimators=100,random_state=42)

# Train the model
rf.fit(x_train,y_train)   

# make predicton
rf_pred=rf.predict(x_test)

# model Evaluation 
print("rf Accuracy score:",accuracy_score(y_test,rf_pred))

# Evaluate Every  model properly
print("Logistic Classification report=",classification_report(y_test,log_pred))
print("KNN classification report=",classification_report(y_test,KNN_pred))
print("Random Forest Classification report=",classification_report(y_test,rf_pred))


# create model comparision table
result={
    "model":["Logistic Regression","KNN","Random Forest"],
    "accuracy":[
    accuracy_score(y_test,log_pred),
    accuracy_score(y_test,KNN_pred),
    accuracy_score(y_test,rf_pred)],

    "precision":[
        precision_score(y_test,log_pred),
        precision_score(y_test,KNN_pred),
        precision_score(y_test,rf_pred)

    ],

    "recall":[
        recall_score(y_test,log_pred),
        recall_score(y_test,KNN_pred),
        recall_score(y_test,rf_pred) ],
                       
    "f1 score":[
        f1_score(y_test,log_pred),
        f1_score(y_test,KNN_pred),
        f1_score(y_test,rf_pred)
    ] ,   

}


# convert into dataframe
result_df=pd.DataFrame(result)
print(result_df)

# Find Highest metrics :--
print("Highest Accuracy",result_df.loc[result_df["accuracy"].idxmax()])
print("Highest Recall",result_df.loc[result_df["recall"].idxmax()])

# Check confusion matrix :--
print("Logistic regression CM:",confusion_matrix(y_test,log_pred))
print("KNN CM:",confusion_matrix(y_test,KNN_pred))
print("Random Forest CM:",confusion_matrix(y_test,rf_pred))

# Visualize comparision :--
import matplotlib.pyplot as plt

result_df.set_index("model")[["accuracy","precision","recall","f1 score"]].plot(kind="bar")
plt.title("Model performance comparision")
plt.ylabel("score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#================================( Day-07 ,Hyperparameter Tuning and model improvement)=====================

# create baseline random Forest :-
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100,random_state=42)

# Train the baseline
rf.fit(x_train,y_train)

# make prediction Baseline
rf_pred=rf.predict(x_test)

# Model Evaluation :-
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

print("Accuracy:",accuracy_score(y_test,rf_pred))
print("Precision:",precision_score(y_test,rf_pred))
print("Recall:",recall_score(y_test,rf_pred))
print("F1 score:",f1_score(y_test,rf_pred))

# model Tuning
from sklearn.model_selection import GridSearchCV

# create parameter grid
params_grid={
    "n_estimators":[50,100,200],
    "max_depth":[None,10,20],
    "min_samples_split":[2,5],
    "min_samples_leaf":[1,2],
}

# create gridsearch CV
grid_search=GridSearchCV(estimator=rf,param_grid=params_grid ,cv=5,scoring="accuracy", n_jobs=-1)

# train the Tuned 
grid_search.fit(x_train,y_train)



# Find best parameters :--
print("best parameters:",grid_search.best_params_)

# Find best cross-validation score :--
print("Best cross validation score:",grid_search.best_score_)

# Get best model
best_rf=grid_search.best_estimator_
print(best_rf)

# prediction Tuned
best_rf_pred=best_rf.predict(x_test)

# Model Evaluation :--
print("Accuracy:",accuracy_score(y_test,best_rf_pred))
print("precision:",precision_score(y_test,best_rf_pred))
print("recall:",recall_score(y_test,best_rf_pred))
print("F1 Score:",f1_score(y_test,best_rf_pred))


# Compare BaseLine vs Tuned model :--
comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ],

    "Baseline RF": [
        accuracy_score(y_test, rf_pred),
        precision_score(y_test, rf_pred),
        recall_score(y_test, rf_pred),
        f1_score(y_test, rf_pred)
    ],

    "Tuned RF": [
        accuracy_score(y_test, best_rf_pred),
        precision_score(y_test, best_rf_pred),
        recall_score(y_test, best_rf_pred),
        f1_score(y_test, best_rf_pred)
    ]
})

print(comparison)


#================( Day-08 , Topic=> Feature importance and model interpretation)=========================

# prediction using Tuned Random Forest 
best_rf_pred = best_rf.predict(x_test)

# Get Feature importance :--
importance = best_rf.feature_importances_

# create DataFrame :--
feature_importance=pd.DataFrame({
    "Feature":x_train.columns,
    "importance":importance
})

# sort by importance
feature_importance=feature_importance.sort_values(by="importance",ascending=False)
print("Sort Feature importance=",feature_importance)

# Top 10 Feature
top_features=feature_importance.head(10)
print("Top_feature:",top_features)

# Most importance feature
most_important_feature = feature_importance.iloc[0]
print("Most important feature:",most_important_feature)

# visualization Top 10 features
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(
    x="Feature",
    y="importance",data=top_features
)

plt.title("Top 10 Feature importance")
plt.xlabel("Feature")
plt.ylabel("importance")
plt.show()

# ============================(Day-09 , Topic name :-ROC-AUC + Prediction Threshold + Model Evaluation)======

#  Calculate Churn Probablity
y_prob = best_rf.predict_proba(x_test)[:,1]
print("Churn Probablity=",y_prob)

# print first 10 churn probablity
print("First 10 churn probablity=",y_prob[:10])

# Calculate Roc :--
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test,y_prob)
print("Roc Curve=",roc_auc)

# Plot Roc-curve :---
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr,tpr,threshold = roc_curve(y_test,y_prob)

plt.figure(figsize=(8,6))

plt.plot(fpr,tpr, label=f"Random Forest(AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],linestyle="--",label="Random Classifier")
plt.title("ROC-CURVE")
plt.xlabel("False Positive Rate")
plt.ylabel("True positive Rate")
plt.legend()
plt.show()

# Experiment with Threshold
y_pred=best_rf.predict_proba(x_test)[:,1]

# Create prediction using threshold 0.5
y_pred_50 = (y_prob >=0.5).astype(int)

# Evaluate
from sklearn.metrics import precision_score, recall_score, f1_score

print("Threshold = 0.50")
print("Precision=",precision_score(y_test,y_pred_50))

print("Recall=",recall_score(y_test,y_pred_50))
print("F1=",f1_score(y_test,y_pred_50))

# Try Threshold 0.40
y_pred_40 = (y_prob >= 0.40).astype(int)

print("Threshold = 0.40")

print("Precision:",
      precision_score(y_test, y_pred_40))

print("Recall:",
      recall_score(y_test, y_pred_40))

print("F1:",
      f1_score(y_test, y_pred_40))

# Try Threshold 0.60
y_pred_60 = (y_prob >= 0.60).astype(int)

print("Threshold = 0.60")

print("Precision:",
      precision_score(y_test, y_pred_60))

print("Recall:",
      recall_score(y_test, y_pred_60))

print("F1:",
      f1_score(y_test, y_pred_60))

# Create a Threshold Comparision DataFrame

threshold_to_test = [0.3,0.4,0.5,0.6,0.7]
threshold_result = []

for threshold in threshold_to_test:
    prediction =(y_prob >=threshold).astype(int)
    threshold_result.append({
        "Threshold":threshold,
        "precision":precision_score(y_test,prediction),
        "recall": recall_score(y_test,prediction),
        "f1":f1_score(y_test,prediction)
    })

threshold_df =pd.DataFrame(threshold_result)    
print(threshold_df)

#==============================(Day-10)==================================================
# Topic name :-- Classification report , Confusion Matrix , Final Accuracy/ precision/recall/f1 score/ROC-AUC

# Final prediction
final_pred = best_rf.predict(x_test)

# Final Churn probablity
final_prob =best_rf.predict_proba(x_test)[:,1]

# Classification report
from sklearn.metrics import classification_report
print("Classification report=",classification_report(y_test,final_pred))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,final_pred)
print("Confusion Matrix=",cm)

# Visualization of confusion matrix

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.title("Final Random forest- Confusion matrix")
plt.show()


# Extract TP,FP,FN,TN 
tn,fp,fn,tp = cm.ravel()

print("True Negative=",tn)
print("False Positive=",fp)
print("False Negative=",fn)
print("True Positive=",tp)

# Final metrics 

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

Accuracy=accuracy_score(y_test,final_pred)
Precision=precision_score(y_test,final_pred)
Recall=recall_score(y_test,final_pred)
F1_score=f1_score(y_test,final_pred)
ROC_AUC_SCORE=roc_auc_score(y_test,final_prob)

print("Accuracy=",Accuracy)
print("Precision score=",Precision)
print("Recall score=",Recall)
print("F1 Score=",F1_score)
print("ROC-AUC-SCORE=",ROC_AUC_SCORE)

# Final summary table
final_summary = pd.DataFrame({
    "Metrics":["Accuracy","Precision","Recall","F1_score","ROC_AUC_SCORE"],
    "Score":[Accuracy,Precision,Recall,F1_score,ROC_AUC_SCORE]
    })
print("Final summary=",final_summary)

# Error Analysis 
error_analysis = x_test.copy()

error_analysis["Actual"]=y_test
error_analysis["Prediction"]=final_pred
error_analysis["Churn Prediction"]= final_prob

# All incorrect prediction 
error = error_analysis[error_analysis["Actual"]!= error_analysis["Prediction"]]

print("Total incorrect prediction=",len(error))

# False Negative 
false_negative = error_analysis[(error_analysis["Actual"]==1) & (error_analysis["Prediction"]==0)]
print("False Negative=",false_negative)
print(false_negative.head())

# False Positive 
false_positive = error_analysis[(error_analysis["Actual"]==0) & (error_analysis["Prediction"]==1)]
print("False Positive=",false_positive)
print(false_positive.head())


#==================(Day-11, Topic name :- Customer Risk Segmentation  and  Churn Prediction insight)========

# Generate churn probability
final_prob = best_rf.predict_proba(x_test)[:,1]
print("Final Probability=",final_prob[:10])

# create customer Risk level
import numpy as np

risk_level = np.select([
    final_prob<0.30,
    final_prob<0.60],
["Low Risk","Medium Risk"],default="High Risk"
)

# Create Risk Dataframe
risk_df = x_test.copy()

risk_df["Churn Probability"]=final_prob
risk_df["Risk Level"]=risk_level
risk_df["Actual Churn"]=y_test.values

print("Risk DataFrame=",risk_df.head())

# Risk count
print("Risk Distribution=",risk_df["Risk Level"].value_counts())

# Risk percentage
print("Risk Percentage=",risk_df["Risk Level"].value_counts(normalize=True)*100)

# Visualization of risk  Distribution

import seaborn as sns
sns.countplot(x="Risk Level",data=risk_df)

plt.xlabel("risk_level")
plt.ylabel("No of customers")
plt.title("Customer Churn Risk Distribution")
plt.show()

# High Risk Customer
high_risk = risk_df[risk_df["Risk Level"]=="High Risk"]
print("High Risk=",high_risk.head())
print("Total High Risk Customer=",len(high_risk))

# Sort High Risk Customer
high_risk = high_risk.sort_values(by="Churn Probability",ascending=False)
print("Top 10 High risk Customer=",high_risk.head(10))

# Compare Risk Group
print("Risk Group Numerical summary")
print(risk_df.groupby("Risk Level")[["tenure","MonthlyCharges"]].mean())

#=============(Day-12 ,Topic name :- Model Saving and Reusable prediction function)=========

# import libraries
import pandas as pd
import numpy as np
import joblib
import os

# Save the Trained Random Forest Model
joblib.dump(best_rf,"churn_model.pkl")
print("model saved succesfully")

# save the Feature name
feature_name = x_train.columns.tolist()
joblib.dump(feature_name,"churn_feature.pkl")
print("Feature name saved Successfully")
print("Total Feature=",len(feature_name))

# Load the Saved model
loaded_model=joblib.load("churn_model.pkl")
loaded_feature=joblib.load("churn_feature.pkl")

print("Model loaded successfully")
print("Total expected features:",len(loaded_feature))

# Verify the model expected feature
print("Expected features:",loaded_feature) 



# create a reusable predicted function
def predict_customer_churn(customer_data):

    # Check input type
    if not isinstance(customer_data,pd.DataFrame):
        raise TypeError(
            "Input Must be pandas DataFrame."
        )

    # Check missing feature
    missing_feature=[
        feature
        for feature in loaded_feature
        if feature not in customer_data.columns
        ]

    if missing_feature:
        raise ValueError(
            f"Missing required Feature:{missing_feature}"
        )

    # Arrange feature in correct order
    customer_data=customer_data[loaded_feature]

    # predict churn probablity
    churn_probablity=loaded_model.predict_proba(customer_data)[:,1]

    # predict churn class
    churn_class=loaded_model.predict(customer_data)

    # Assign Customer risk level
    risk_levels=np.select([
        churn_probablity<0.30,
        churn_probablity<0.60
    ],
    ["Low Risk","Medium Risk"],default="High Risk"
        
    )

    # create prediction result
    result=pd.DataFrame({
        "churn_probablity":churn_probablity,
        "predicted churn":churn_class,
        "Risk level":risk_levels},index=customer_data.index
    )
    return result

# predict churn for one customer
sample_customer=x_test.iloc[[0]]
single_prediction = predict_customer_churn(sample_customer)
print("Single customer predictions:",single_prediction)

# Step 8:predict churn for five customers
sample_customer=x_test.iloc[:5]

multiple_predictions=predict_customer_churn(sample_customer)
print("Multiple Predictions=",multiple_predictions)

# Verify original and loaded model predictions
original_predictions=best_rf.predict(sample_customer)

loaded_prediction = loaded_model.predict(sample_customer)

predictions_match=np.array_equal(original_predictions,loaded_prediction)


# model Verification
print("Original predictions=",original_predictions)
print("Loaded model predictions=",loaded_prediction)
print("predictions match=",predictions_match)

# check the saved file
print("Model file exist=",os.path.exists("churn_model.pkl"))
print("Feature file exist=",os.path.exists("churn_features.pkl"))


# Display final summary
print("Model Saving: completed")
print("model loading: completed")
print("predicted Function: created")
print("Single customer prediction: completed")
print("Multiple customer predictions: completed")
print("Model verification:",predictions_match)

#=========================(Day-13, Topic name :-Build a customer churn predictions applications)========

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and feature
@st.cache_resource
def load_model():
    model=joblib.load("churn_model.pkl")
    feature=joblib.load("churn_feature.pkl")
    return model,feature

model,feature = load_model()

# Application title
st.title("customer churn prediction system")
st.write(
    "Enter customer information to estimate"
    "The probablity of customer churn"
)

# customer input form
tenure=st.number_input(
    "Tenure(months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges=st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges=st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)

contract=st.selectbox(
    "Contract",
    ["Month-to-Month","One year","Two-year"]
)

internet_service=st.selectbox(
    "Internet Service",
    ["DSL","Fibre-optic","No"]
)

payment_method=st.selectbox(
    "Payment Method",
    [
        "Electrinic check",
        "Mailed check",
        "Bank transfer(automatic)",
        "Credit card (automatic)"
    ]
)

# create input DataFrame

if st.button("predict churn"):
    customer_data=pd.DataFrame({
        "tenure":[tenure],
        "Monthly charges":[monthly_charges],
        "Total charges":[total_charges],
        "Contract":[contract],
        "Interner services":[internet_service],
        "Payment method":[payment_method]
    })

    st.write("customer_input")
    st.dataframe(customer_data)


    st.warning(
        "This demonstration form contains only selected "
        "features. The saved model requires its complete "
        "training feature set and matching encoding."
    )

    st.info(
        "Complete the preprocessing integration before "
        "using this application for actual predictions.")


#===============================(Day-14 , Topic name-Complete machine Learning pipeline)==========

# Import libraries
# We need pandas for data handling and scikit-learn for preprocessing, model training, and evaluation.

import pandas as pd
import numpy as np
import joblib



from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report,confusion_matrix


# Load the datasets
df = pd.read_csv(
    r"C:\Users\HP\Downloads\WA_Fn-UseC_-Telco-Customer-Churn (1).csv"
)

print(df.head())
print("Dataset shape:", df.shape)
print(df.columns.tolist())

# clean the datasets
# Remove the customer ID
df=df.drop(columns=["customerID"])

# Convert total charges into numerical values
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

# Remove row with the missing value
df=df.dropna().copy() 

# Encode the target variable
df["Churn"]=df["Churn"].map({
    "No":0,
    "Yes":1
})

print("missing value=",df.isnull().sum())
print("Duplicate rows=",df.duplicated().sum())
print(df.head())

# separate input feature and target variable
x=df.drop(columns=["Churn"])
y=df["Churn"]

print("x.shape=",x.shape)
print("y shape=",y.shape)

# Split the datasets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# Identify the numerical and categorical column
num_feature=x_train.select_dtypes(include=["number"]).columns.tolist()
cat_feature=x_train.select_dtypes(include=["object","category","bool"]).columns.tolist()

print("Numerical features=",num_feature)
print("Categorical feature=",cat_feature)


# Build the processing pipeline
numerical_transformer="passthrough"
categorical_transformer= OneHotEncoder(handle_unknown="ignore")

preprocessor=ColumnTransformer(transformers=[("num",numerical_transformer,num_feature),("cat",categorical_transformer,cat_feature)])

# Build the complete ml pipeline
# we will combine preprocessing and Random Forest into one pipeline.

rf_model=RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced")
churn_pipeline=Pipeline(steps=[("preprocessor",preprocessor),("model",rf_model)])


# Train the pipeline
churn_pipeline.fit(x_train,y_train)
print("complete pipeline trained successfully")


from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "churn_pipeline.pkl"

joblib.dump(churn_pipeline, MODEL_PATH)

print("PASS: Pipeline saved successfully")
print("Saved at:", MODEL_PATH)


# Evaluate the model
# Predict classes
y_pred = churn_pipeline.predict(x_test)

# Predict churn probabilities
y_prob = churn_pipeline.predict_proba(x_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#=====================(Day-15, Topic name:- Streamlit web application for customer churn prediction )========================

# application.py

import pandas as pd
import joblib
import streamlit as st
from pathlib import Path


# page configuration
st.set_page_config(
    page_title="Customer Churn prediction",
    page_icon="📊",
    layout="wide"
    )

# Load saved ml pipeline
MODEL_PATH=("churn_pipeline.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Application title
st.title("📊 Customer Churn Prediction System")

st.write(
    "Enter customer details to estimate churn probability "
    "using a trained Machine Learning pipeline."
)

st.info(
    "This application estimates churn risk. "
    "It does not guarantee that a customer will or will not leave."
)


# Check model file
from pathlib import Path
MODEL_PATH=Path("churn_pipeline.pkl")
if not MODEL_PATH.exists():
    st.error(
        "Model not found .please run train_pipeline.py first"
             )
    st.stop()

try:
    model=load_model()
except Exception as error:

    st.error(
        f"Unable to load the saved model:{error}"
    )

    st.stop()

# Collect customer information
st.header("Enter Customer information")

with st.form("customer_form"):
    col1,col2,col3=st.columns(3)

    # Customer information

    with col1:

        gender=st.selectbox(
            "Gender",
            ["Male","Female"]
        )

        senior_citizen=st.selectbox(
            "senior citizen",
            [0,1],
            format_func=lambda value: ("Yes" if value==1 else "No")
        )

        partner=st.selectbox(
            "Has Partner?",
            ["Yes","No"]
        )

        dependents=st.selectbox(
            "Has Dependent?",
            ["Yes","No"]
        )

        tenure=st.number_input(
            "Tenure(months)",
            min_value=0,
            max_value=100,
            value=12
         )

        # Service information
    with col2:

            phone_service=st.selectbox(
                "Phone Service",
                ["Yes","No"]
            )

            multiple_lines=st.selectbox(
                "Multiple Lines",
                ["Yes","No","No phone Service"]
            )

            internet_service=st.selectbox(
                "Internet service",
                ["DSL","Fiber optic","No"]
            )

            online_security=st.selectbox(
                "Online Security",
                ["Yes","No","No internet service"]
            )

            online_backup=st.selectbox(
                "Online Backup",
                ["Yes","No","No internet service"]
            )

            device_protection=st.selectbox(
                "Device Protection",
                ["Yes","No","No internet service"]
            )

            # Contract and billing information
    with col3:

            tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

            streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

            streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

            contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

            paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

            payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        # Additional Billing input:-
            col4, col5 = st.columns(2)

    with col4:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )

    with col5:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=float(tenure * monthly_charges),
            step=10.0
        )

    submit = st.form_submit_button(
        "Predict Churn"
    )


# --------------------------------------------------
# 6. Make Prediction
# --------------------------------------------------

if submit:

    # Prepare data using original training column names
    customer_data = pd.DataFrame(
        [{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }]
    )

    try:

        # Predict churn class
        prediction = int(
            model.predict(customer_data)[0]
        )

        # Predict probability of churn
        probability = float(
            model.predict_proba(customer_data)[0, 1]
        )

        # Assign risk category
        if probability < 0.30:

            risk_level = "Low Risk"

        elif probability < 0.60:

            risk_level = "Medium Risk"

        else:

            risk_level = "High Risk"

        # Display results
        st.divider()

        st.header("Prediction Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )

        with result_col2:

            st.metric(
                "Risk Category",
                risk_level
            )

        with result_col3:

            prediction_text = (
                "Likely to Churn"
                if prediction == 1
                else "Not Predicted to Churn"
            )

            st.metric(
                "Model Prediction",
                prediction_text
            )

        # Risk explanation
        if risk_level == "High Risk":

            st.error(
                "The model estimates a relatively high churn risk. "
                "Consider reviewing this customer's needs "
                "and available retention options."
            )

        elif risk_level == "Medium Risk":

            st.warning(
                "The model estimates a moderate churn risk. "
                "Consider monitoring this customer."
            )

        else:

            st.success(
                "The model estimates a relatively low churn risk."
            )

        # Display submitted customer details
        with st.expander("View Customer Details"):

            st.dataframe(
                customer_data,
                use_container_width=True
            )

        st.caption(
            "Risk thresholds (30% and 60%) are illustrative. "
            "They should be validated against business objectives "
            "and model performance before production use."
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )

        st.info(
            "Check that the input columns and category values "
            "match those used to train the pipeline."
        )


# --------------------------------------------------
# 7. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction System | "
    "Python • Scikit-learn • Random Forest • Streamlit")


#===================(Day-16,Topic- Application Testing ,Validation and error Handling )============================== 

# Test the saved pipeline

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Locate saved pipeline in project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "churn_pipeline.pkl"

# Check model file
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}. "
        "Run the Day 14 training code first."
    )

# Load model
model=joblib.load(MODEL_PATH)
print("PASS: model loaded successfully")

# Sample Customer
customer_data=pd.DataFrame([{
    "gender":"Male",
    "SeniorCitizen":0,
    "Partner":"Yes",
    "Dependents":"No",
    "tenure":12,
    "PhoneService":"Yes",
    "MultipleLines":"No",
    "InternetService":"DSL",
    "OnlineSecurity":"No",
    "OnlineBackup":"Yes",
    "DeviceProtection":"No",
    "TechSupport":"No",
    "StreamingTV":"No",
    "StreamingMovies":"No",
    "Contract":"Month-to-month",
    "PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check",
    "MonthlyCharges":70.0,
    "TotalCharges":840.0
}])

# match model input columns
preprocessor=model.named_steps["preprocessor"]
expected_columns=list(preprocessor.feature_names_in_)

missing=set(expected_columns)-set(customer_data.columns)
extra=set(customer_data.columns)-set(expected_columns)

assert not missing,f"Missing column:{missing}"
assert not extra,f"Unexpected columns:{extra}"

customer_data=customer_data[expected_columns]

# Genearte predictions
prediction=model.predict(customer_data)
probabilities=model.predict_proba(customer_data)

assert len(prediction)==1
assert probabilities.shape==(1,2)
assert prediction[0] in [0,1]
assert 0<=probabilities[0,1]<=1

print("PASS: Input schema validated")
print("PASS: Predictions generations")
print("PASS: Probability is within valid range")

print("predicted class=",prediction[0])
print(f"churn probability:{probabilities[0,1]:.2%}")


# ==============(Day-17, Topic name:- Feature importance churn Analysis  )==============================================

# Today's main goal
# Understand why the Random Forest model predicts that a customer may leave.


from pathlib import Path   # it helps in create the path file location
import joblib              # Load the trained model
import pandas as pd
import matplotlib.pyplot as plt

#  Project root: one level above notebooks/
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "churn_pipeline.pkl"

# Load the complete trained Pipeline (Preprocessing and random forest model both included in pipeline)
churn_pipeline=joblib.load(MODEL_PATH)
print("Pipeline loaded Successfully")

# Extract the trained random forest
rf_model=churn_pipeline.named_steps["model"]

# Extract the preprocessing component
preprocessor=churn_pipeline.named_steps["preprocessor"]

print("Random forest extracted successfully")


# Get the feature names after preprocessing
feature_names=preprocessor.get_feature_names_out()
print("Total processed feature:",len(feature_names))
print(feature_names)


# create a feature importance tables

# Get importance score from random forest
importances=rf_model.feature_importances_

# check that names and score match
assert len(feature_names)==len(importances)

# create a DataFrame
importance_df=pd.DataFrame({
    "Feature":feature_names,
    "Importance":importances
})

# sort from highest to lowest
importance_df=importance_df.sort_values(by="Importance",ascending=False).reset_index(drop=True)

print(importance_df.head(10))

# Visualize the Top 10 features
top_10=importance_df.head(10).sort_values(by="Importance",ascending=True)

# plot a Horizontal bar plot
plt.figure(figsize=(10,6))
plt.barh(top_10["Feature"],top_10["Importance"])

plt.xlabel("Importance score")
plt.ylabel("Feature")
plt.title("Top_10 Feature importances- Churn prediction")

plt.tight_layout()
plt.show()

# save the result
OUTPUT_PATH=BASE_DIR /"feature_importance.csv"
importance_df.to_csv(OUTPUT_PATH,index=False)
print("Feature importance saved at:",OUTPUT_PATH)


#==================(Day-18 , Topic name -permutation and model Evaluation)==================

# Today's main goal
# Check how much the model's performance changes when a feature's values are shuffled.


from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


# set the file path
BASE_DIR= Path(__file__).resolve().parent.parent
MODEL_PATH=BASE_DIR/"churn_pipeline.pkl"
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = BASE_DIR / "permutation_importance.csv"
CHART_PATH = BASE_DIR / "permutation_importance.png"


# load the trained pipeline
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found :{MODEL_PATH}"
        "run your training script first  and save churn_pipeline.pkl"
    )

churn_pipeline=joblib.load(MODEL_PATH)
print("Trained pipeline loaded successfully")


# Load the datasets
if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Datasets not found:{DATA_PATH}"
        "Update DATA_PATH to your actual dataset location"
    )

df=pd.read_csv(DATA_PATH)
print("Data loaded successfully")
print("Original dataset shape:",df.shape)

# Data preprocesssing

# Remove customer identifier
df=df.drop(columns=["customerID"])

# Convert total charges to numeric
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

# convert target label to Binary
df["Churn"]=df["Churn"].map({
    "No":0,
    "Yes":1
})

# Remove row with missing value
df=df.dropna()

# Separate feature and target
x=df.drop(columns=["Churn"])
y=df["Churn"]

print("Feature shape:",x.shape)
print("Target shape:",y.shape)

# Recreate test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

print("Test split created successfuly")

# calculate Baseline ROC-AUC

# predict churn probability for test datasets
y_pred=churn_pipeline.predict_proba(x_test)[:,1]

# calculate baseline ROC-AUC
Baseline_auc=roc_auc_score(y_test,y_pred)
print("Baseline ROC-AUC=",Baseline_auc)

# calculate permutation importance
result=permutation_importance(
    estimator=churn_pipeline,
    X=x_test,
    y=y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1)
print("Permutation importance calculated successfully!")

# create importance dataframe
permutation_df=pd.DataFrame({
    "Feature":x_test.columns,
    "Importance_Mean":result.importances_mean,
    "Importance_std":result.importances_std
})

# Sort features from highest to lowest importance
permutation_df=permutation_df.sort_values(
    by="Importance_Mean",
    ascending=False).reset_index(drop=True)

print("\n===========Top 10 Important permutation features============")
print(permutation_df.head(10).to_string(index=False))

# Visualize top 10 features
top_10=permutation_df.head(10).sort_values(by="Importance_Mean",ascending=True)
plt.figure(figsize=(10,6))

plt.barh(top_10["Feature"],top_10["Importance_Mean"],xerr=top_10["Importance_std"])
plt.xlabel("Mean Decrease in ROC-AUC")
plt.ylabel("Original Feature")
plt.tight_layout()

# save chart
plt.savefig(CHART_PATH,dpi=150,bbox_inches="tight",)
plt.show()

print("\n saved chart at:",CHART_PATH)

# Saved results
permutation_df.to_csv(OUTPUT_PATH , index=False)
print("Permutation importance saved at:",OUTPUT_PATH)

# Final summary
print("\n========== DAY 18 SUMMARY ==========")
print(f"Baseline ROC-AUC: {Baseline_auc:.4f}")
print("Total features analyzed:", len(permutation_df))
print("Top feature:", permutation_df.iloc[0]["Feature"])
print(
    "Top feature importance:",
    round(permutation_df.iloc[0]["Importance_Mean"], 4)
)

print("\nDay 18 analysis completed successfully!")


# ===============(Day-19, Topic name :-Customer churn risk analysis )===================================

# Today's main goal
# Use the trained pipeline to predict churn probabilities, assign Low/Medium/High risk levels, and create a report that helps identify customers who may leave.


# steps :--
# Load saved model and customer data

# Predict churn probability

# Assign Low / Medium / High risk

# Save CSV report and analyze high-risk customers


from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# set the project path
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "churn_pipeline.pkl"
DATA_PATH = (
    BASE_DIR
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
OUTPUT_PATH = BASE_DIR / "churn_risk_report.csv"
CHART_PATH = BASE_DIR / "churn_risk_distribution.png"

# Check and Load the model
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found:{MODEL_PATH}") 

churn_pipeline=joblib.load(MODEL_PATH)
print("MOdel loaded successfully")

# Load the datasets
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found:{DATA_PATH}")

df=pd.read_csv(DATA_PATH) 
print("Dataset loaded successfully")

#3- Keep customerID for the final report
customer_ids=df["customerID"].copy()

# 4- Prepare features
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

# Remove ID and target column from model input
x=df.drop(columns=["customerID","Churn"])

# Remove row with missing feature value
valid_row=x.notna().all(axis=1)
x=x.loc[valid_row].copy()

customer_ids=customer_ids.loc[valid_row]

# Predict churn probablity
churn_probability=churn_pipeline.predict_proba(x)[:,1]

# Assign risk level
risk_level=[]

for probability in churn_probability:
    if probability < 0.30:
        risk_level.append("Low")
    elif probability <0.60:
        risk_level.append("Medium")
    else:
        risk_level.append("High")

# Create the customer risk report
risk_report=pd.DataFrame({
    "customerID":customer_ids.to_numpy(),
    "Churn_probability":churn_probability,
    "Risk_level":risk_level
})    

risk_report["Churn_probability"]=(
    risk_report["Churn_probability"].round(4)
)

risk_report=risk_report.sort_values(by="Churn_probability",ascending=False)

# 8. save the report
risk_report.to_csv(OUTPUT_PATH,index=False)

# Display summary :--
print(risk_report["Risk_level"].value_counts())

print("\n Top 10 High risk customer")
print(risk_report.head(10).to_string(index=False))

# Plot risk distribution :--
risk_order=["Low","Medium","High"]
risk_counts=(risk_report["Risk_level"].value_counts().reindex(risk_order,fill_value=0))

risk_counts.plot(kind="bar")
plt.title("Customer churn risk distribution")
plt.xlabel("Risk_level")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(CHART_PATH,dpi=150)
plt.show()

print("Report saved at:",OUTPUT_PATH)
print("CHART saved at:",CHART_PATH)
print("Day-19 completed successfully")


#====================(Day-20, Topic name :- Customer Retention strategy and Business Recommendations)===========
# Use the churn prediction results to decide which customers need attention and what actions a company should take to retain them.
# हिंदी: Churn prediction के results का इस्तेमाल करके यह तय करना कि किन customers पर ध्यान देना चाहिए और उन्हें company छोड़ने से रोकने के लिए क्या कदम उठाने चाहिए।

from pathlib import Path
import pandas as pd

#1. Set the project path
BASE_DIR=Path(__file__).resolve().parent.parent
INPUT_PATH=BASE_DIR/"churn_risk_report.csv"
OUTPUT_PATH=BASE_DIR/"retention_recommendations.csv"

# Check and load the day-19 report
if not INPUT_PATH.is_file():
    raise FileNotFoundError(
        f"Report not found:{INPUT_PATH}"
    )

df=pd.read_csv(INPUT_PATH)

print("Day-19 report loaded successfully")
print("Report shape:",df.shape)
print("Availabel columns:",df.columns.tolist())

# Find the Risk level column
column_lookup={
    col.strip().lower().replace(" ","_"):col
    for col in df.columns
}

possible_risk_columns=["risk_level","churn_risk","risk","risk_category"]

risk_column = next(
    (column_lookup[c] for c in possible_risk_columns
     if c in column_lookup),
    None
)

if risk_column is None:
    raise ValueError(
        "Risk column not found. Check the printed column names."
    )

# Create a retention recommentation
def recommend_action(risk):
    risk=str(risk).strip().lower()

    if risk=="high":
        return("prioritize personal outreach ;investigate"
               "Service issue and retention option.")

    elif risk=="medium":
        return("Review customer feedback,billing, and"
               "contract needs")

    elif risk=="low":
        return("continue normal service and monitor"
               "for changes in customer behavior")
    else:
        return "Review customer risk level manually"
        

df["Retention_Action"]= df[risk_column].apply(recommend_action)

# 5. Summarize customers by risk level
print("\nCustomer count by risk level:")
print(df[risk_column].value_counts(dropna=False))

# 6. Save the recommendations
df.to_csv(OUTPUT_PATH, index=False)

print("\nRetention recommendations saved to:")
print(OUTPUT_PATH)
print("\nDay 20 task completed!")


#========(Day-21 , Topic name :- Customer Retention Dashboard and Business insight)==========

# Today's main goal: Convert customer churn predictions and retention recommendations into 
# a simple dashboard that helps a business understand customer risk.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# set project path
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR/ "retention_recommendations.csv"
OUTPUT_PATH = BASE_DIR/ "churn_dashboard.png"


#2. Load the retention report
if not INPUT_PATH.is_file():
    raise FileNotFoundError(
        f"Report not found:{INPUT_PATH}"
        "Run the day 20 script first"
    )

df=pd.read_csv(INPUT_PATH)

if df.empty:
    raise ValueError("The retention report is empty")

print("Retention report loaded successfully")
print("Total records:",len(df)) 
print("Available columns:",df.columns.tolist())

# Identify the risk level column
column_lookup = {
    col.strip().lower().replace(" ","_"): col
    for col in df.columns
}

possible_column=[
    "risk_level",
    "churn_risk",
    "risk",
    "risk_category"
]

risk_column = next(
    (column_lookup[col]
     for col in possible_column
        if col in column_lookup
    ),None
)

if risk_column is None:
    raise ValueError(
        "Risk level column not found"
        "Check the available column name "
    )

# Standardize the risk level
def standardize_risk(value):
    value=str(value).strip().lower()

    if "high" in value:
        return "High"
    elif "medium" in value:
        return "Medium"
    elif "low" in value:
        return "Low"
    else:
        return "Unknown"

df["Risk_Group"]=df[risk_column].apply(standardize_risk)    


# Calculate dashboard matrics
total_customers=len(df)
risk_order=["High","Medium","Low","Unknown"]

risk_counts = (df["Risk_Group"].value_counts().reindex(risk_order,fill_value=0))

high_risk=int(risk_counts["High"])
medium_risk = int(risk_counts["Medium"])
low_risk=int(risk_counts["Low"])
unknow_risk=int(risk_counts["Unknown"])

print("\n CHURN DASHBOARD SUMMARY")
print("Total Customers:",total_customers)
print("High risk customer:",high_risk)
print("Medium risk customer:",medium_risk)
print("Low risk customer:",low_risk)
print("Unknown risk customer:",unknow_risk)

# Create a dashboard
fig=plt.figure(figsize=(12,7))

fig.suptitle("Customer churn risk dashboard",fontsize=17)

# KPI card
kpis=[
    ("Total customers:",total_customers),
    ("High Risk:",high_risk),
    ("Medium Risk:",medium_risk),
    ("Low Risk:",low_risk)
]

for i, (label,value) in enumerate(kpis):
    ax = fig.add_subplot(2, 4, i + 1)
    ax.axis("off")

    ax.text(
        0.5, 0.65, str(value),
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold"
    )

    ax.text(
        0.5, 0.25, label,
        ha="center",
        va="center",
        fontsize=11
    )

# Risk distribution chart
ax = fig.add_subplot(2, 1, 2)

plot_counts = risk_counts[
    ["High", "Medium", "Low"]
]

ax.bar(
    plot_counts.index,
    plot_counts.values
)

ax.set_title("Customer Distribution by Risk Level")
ax.set_xlabel("Risk Level")
ax.set_ylabel("Number of Customers")

ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.show()


# 7. Save a summary CSV
summary = risk_counts.rename_axis("Risk_Level").reset_index(
    name="Customer_Count"
)

summary.to_csv(
    BASE_DIR / "churn_dashboard_summary.csv",
    index=False
)

print("\nDashboard image saved to:", OUTPUT_PATH)
print(
    "Summary CSV saved to:",
    BASE_DIR / "churn_dashboard_summary.csv"
)

print("\nDay 21 dashboard completed successfully!")


# =====================( Day-22 , Topic :Classification model)==================
# Today's focus will be Logistic Regression, the first classification model in our learning sequence.
# 3. Today's Practical Task
# Goal: Train a Logistic Regression model using your Telco Customer Churn dataset and evaluate its performance.

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score, classification_report,confusion_matrix

# set project path
BASE_DIR=Path(__file__).resolve().parent.parent
DATA_PATH=Path(BASE_DIR/"data",r"C:\Users\HP\Downloads\WA_Fn-UseC_-Telco-Customer-Churn (1).csv")
REPORTS_PATH = BASE_DIR / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH=BASE_DIR/"Logisticre_gression_churn_pipeline.pkl"
METRICS_PATH = REPORTS_PATH / "day22_logistic_regression_metrics.csv"


# Load the dataset
df=pd.read_csv(DATA_PATH)
print("Datasets shape:",df.shape)
print(df.head())

# clean total charges
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

df=df.dropna(subset=["TotalCharges"]).copy()


# Prepare target variable 
y=df["Churn"].map({"No":0,"Yes":1})

# Remove ID and target from input features
x=df.drop(columns=["customerID","Churn"])

# Identify numerical and categorical column
numerical_col=x.select_dtypes(include="number").columns.tolist()
categorical_col=x.select_dtypes(include=["object","category","bool"]).columns.tolist()

print("Numerical_column:",numerical_col)
print("Categorical column:",categorical_col)

# create preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num",StandardScaler(),numerical_col),
    ("cat",OneHotEncoder(handle_unknown="ignore"),
    categorical_col),
])

# split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# create a Logistic regression pipeline
model=Pipeline(steps=[
    ("Preprocessor",preprocessor),
    ("model",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42))
     ])

# train the model on training data
model.fit(x_train,y_train)

# make prediction on unseen data
y_pred=model.predict(x_test)
y_pred_proba=model.predict_proba(x_test)[:,1]


# Evaluate the model
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1_Score": f1_score(y_test, y_pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, y_pred_proba)
}

print("\n Model Evaluation:",)
for name,value in metrics.items():
    print(f"{name}: {value:.4f}")

print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("classification report:",classification_report(y_test,y_pred,zero_division=0))

# save the evaluation result
pd.DataFrame([metrics]).to_csv(METRICS_PATH,index=False)

# save the complete pipeline
joblib.dump(model,MODEL_PATH)


#================(Day-23, Topic name :- Classification model (KNN classifier))===================
# 1. Today's topics

# 1. Introduction to K-Nearest Neighbors (KNN).
# 2. How the KNN algorithm works.
# 3. Distance measurement — Euclidean Distance.
# 4. Choosing the value of K.
# 5. Why feature scaling is important in KNN.
# 6. Implementing KNN on our Customer Churn dataset.
# 7. Evaluating the model using Accuracy, Precision, Recall, F1-score and ROC-AUC.
# 8. Saving the trained KNN pipeline.

# 2. Today's main goal
# English: Build a KNN classification model that predicts whether a customer is likely to leave the company.

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report


#1. set project path
BASE_DIR=Path(__file__).resolve().parent.parent
DATA_PATH=Path(r"C:\Users\HP\Customer-Churn-Prediction\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")
REPORTS_PATH = BASE_DIR / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

METRICS_PATH = REPORTS_PATH / "day23_knn_metrics.csv"
MODEL_PATH=BASE_DIR/"knn_churn_pipeline.pkl"


#2. Load the datasets
df=pd.read_csv(DATA_PATH)
print("Dataset shape:",df.shape)
print(df.head())

#3 clean total charges
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

df=df.dropna(subset=["TotalCharges"]).copy()

# 4 Prepare feature and target
x=df.drop(columns=["customerID","Churn"])
y=df["Churn"].map({"No":0,"Yes":1})

#5 identify numerical and categorical column
numerical_cols= x.select_dtypes(include="number").columns.tolist()
categorical_col=x.select_dtypes(include=["object","category","bool"]).columns.tolist()

print("Numerical columns:",numerical_cols)
print("Categorical column:",categorical_col)

#6 create preprocessing
preprocessor=ColumnTransformer(transformers=[
    ("num",StandardScaler(),numerical_cols),
     ("cat",OneHotEncoder(handle_unknown="ignore"),
     categorical_col)
])

# 7. split the datasets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#8. Create Knn pipeline
model=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",KNeighborsClassifier(n_neighbors=5,metric="euclidean"))
               ])

# 9. train the model
model.fit(x_train,y_train)
print("Knn Model completed")

# 10. Make predictions
y_pred=model.predict(x_test)
y_pred_proba=model.predict_proba(x_test)[:,1]

# Model Evaluation
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1_Score": f1_score(y_test, y_pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, y_pred_proba)
}

print("\nKNN Model Evaluation:")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# 12. Save evaluation metrics
pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

# 13. Save complete pipeline
joblib.dump(model, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)
print("Metrics saved at:", METRICS_PATH)

#===============(Day-24 , Topic name :- Decision tree classifier)===================

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics  import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report

#  1. Set project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = Path(
    r"C:\Users\HP\Customer-Churn-Prediction\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_PATH = BASE_DIR / "decision_tree_churn_pipeline.pkl"
REPORTS_PATH = BASE_DIR / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

METRICS_PATH = REPORTS_PATH / "day24_decision_tree_metrics.csv"

#2.-load dataset
df=pd.read_csv(DATA_PATH)
print("Dataset shape:",df.shape)

#3 clean Total charges
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df=df.dropna(subset=["TotalCharges"]).copy()

#4 create input feature and target variable
x=df.drop(columns=["customerID","Churn"])
y=df["Churn"].map({"No":0,"Yes":1})

#5.- Identify numerical column and categorical column
numerical_col=x.select_dtypes(include="number").columns.tolist()
categorical_col=x.select_dtypes(include=["object","category","bool"]).columns.tolist()

#6. Preprocessing
preprocessor=ColumnTransformer(transformers=[
    ("num",StandardScaler(),numerical_col),
    ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_col)
    
    ])

# Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# create Decision tree pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=5,
                random_state=42
            )
        )
    ]
)

# 9. Train model
model.fit(x_train, y_train)

print("\nDecision Tree training completed!")

# 10. Make predictions
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

# 11. Evaluate model
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "F1_Score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "ROC_AUC": roc_auc_score(
        y_test, y_pred_proba
    )
}

print("\nDecision Tree Evaluation:")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# 12. Save metrics
pd.DataFrame([metrics]).to_csv(
    METRICS_PATH,
    index=False
)

# 13. Save complete pipeline
joblib.dump(model, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)
print("Metrics saved at:", METRICS_PATH)
print("Day 24 completed")

#=================(Day-25, Topic:- Random Forest classifier)=====================

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report

# set project path
# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = Path(
    r"C:\Users\HP\Downloads\WA_Fn-UseC_-Telco-Customer-Churn (1).csv"
)

MODEL_PATH = BASE_DIR / "random_forest_churn_pipeline.pkl"
METRICS_PATH = BASE_DIR / "reports" / "day25_random_forest_metrics.csv"

pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

#2.Load the dataset
df=pd.read_csv(DATA_PATH)
print("Dataset shape:",df.shape)

#3.clean totalcharges
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df=df.dropna(subset=["TotalCharges"]).copy()

#4.input feature and Target variable
x=df.drop(columns=["customerID","Churn"])
y=df["Churn"].map({"No":0,"Yes":1})

#5.- Identify numerical and categorical column
numerical_col=x.select_dtypes(include="number").columns.tolist()
categorical_col=x.select_dtypes(include=["object","category","bool"]).columns.tolist()

#6 preprocessing
preprocessor=ColumnTransformer(transformers=[
    ("num","passthrough",numerical_col),
    ("cat",OneHotEncoder(handle_unknown="ignore"),
     categorical_col)
])

#7 split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#8 create random forest pipeline
model=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42,n_jobs=-1))

])

#9 Train the model
model.fit(x_train,y_train)

#10 make prediction
y_pred=model.predict(x_test)
y_pred_proba=model.predict_proba(x_test)[:,1]

# 11.Evaluate model
metrics={
    "Accuracy":accuracy_score(y_test,y_pred),
    "Precision":precision_score(y_test,y_pred,zero_division=0),
    "Recall":recall_score(y_test,y_pred,zero_division=0),
    "F1-score":f1_score(y_test,y_pred,zero_division=0),
    "Roc-auc":roc_auc_score(y_test,y_pred_proba)
}

for name,value in metrics.items():
    print(f"{name}:{value:.4f}")

print("Confusion matrix:",confusion_matrix(y_test,y_pred))    
print("Classification matrix:",classification_report(y_test,y_pred,zero_division=0))

# save metrics
pd.DataFrame([metrics]).to_csv(METRICS_PATH,index=False)

# save complete pipeline
joblib.dump(model,MODEL_PATH)

print("Model saved at:",MODEL_PATH)
print("Metrics saved at:",METRICS_PATH)


# ==========(Day-26, Topic name:- SVM(support vector machine))=====================

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = Path(
    r"C:\Users\HP\Downloads\WA_Fn-UseC_-Telco-Customer-Churn (1).csv"
)

MODEL_PATH = BASE_DIR / "svm_churn_pipeline.pkl"
REPORTS_PATH = BASE_DIR / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

METRICS_PATH = REPORTS_PATH / "day26_svm_metrics.csv"

# 2. Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# 3. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["TotalCharges"]).copy()

# 4. Prepare features and target
x = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 5. Identify numerical and categorical columns
numerical_cols = x.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = x.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# 6. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ]
)

# 7. Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Create SVM pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=42
            )
        )
    ]
)

# 9. Train model
model.fit(x_train, y_train)

print("\nSVM training completed!")

# 10. Make predictions
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

# 11. Evaluate model
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(
        y_test, y_pred_proba
    )
}

print("\nSVM Evaluation:")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# 12. Save metrics
pd.DataFrame([metrics]).to_csv(
    METRICS_PATH,
    index=False
)

# 13. Save complete pipeline
joblib.dump(model, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)
print("Metrics saved at:", METRICS_PATH)

#============(Day-26, Topic name:-Gradient Boosting classifier)===================
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report

# 1. set project path
BASE_DIR=Path(__file__).resolve().parent.parent
DATA_PATH=Path(r"C:\Users\HP\Downloads\WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

MODEL_PATH=BASE_DIR/"gradient_boosting_churn_pipeline.pkl"
REPORTS_PATH = BASE_DIR / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

METRICS_PATH = REPORTS_PATH / "day_27_gradient_boosting_metrics.csv"

#2. Load the datasets
df=pd.read_csv(DATA_PATH)
print("Dataset shape:",df.shape)

#3.clean total charges
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

df=df.dropna(subset=["TotalCharges"]).copy()

# Prepare input feature and target variable
x=df.drop(columns=["customerID","Churn"])
y=df["Churn"].map({"No":0,"Yes":1})

# Identify the numerical and categorical column
numerical_col=x.select_dtypes(include="number").columns.tolist()
categorical_col=x.select_dtypes(include=["object","category","bool"]).columns.tolist()

# preprocessing
preprocess=ColumnTransformer(transformers=[
    ("num",StandardScaler(),numerical_col),
    ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_col)])

# Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# create Gradient Boosting pipeline
model=Pipeline(steps=[
    ("Preprocessor",preprocess),
    ("model",GradientBoostingClassifier(
        n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42
    ))
])

# train the model
model.fit(x_train,y_train)

# prediction on unseen data
y_pred=model.predict(x_test)
y_pred_proba=model.predict_proba(x_test)[:,1]


# Model Evaluation
metrics={
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(
        y_test, y_pred_proba
    )
}

print("\nGradient Boosting Evaluation:")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# save metrics
pd.DataFrame([metrics]).to_csv(METRICS_PATH,index=False)

# save complete pipeline
joblib.dump(model,MODEL_PATH)

print("Model saved at:",MODEL_PATH)
print("Metrics saved at:",METRICS_PATH)

#==============(Day-28, Topic name:-XGBoostclassifier)=================


import pandas as pd
import joblib

from pathlib import Path
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# 1. File paths
PROJECT_PATH = Path(r"C:\Users\HP\Customer-Churn-Prediction")

DATA_PATH = Path(
    r"C:\Users\HP\Downloads\WA_Fn-UseC_-Telco-Customer-Churn (1).csv"
)

MODEL_PATH = PROJECT_PATH / "models" / "xgboost_churn_pipeline.pkl"

METRICS_PATH = PROJECT_PATH / "reports" / "day_28_xgboost_metrics.csv"


# 2. Load dataset
df = pd.read_csv(DATA_PATH)


# 3. Clean data
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.dropna(subset=["TotalCharges"]).copy()


# 4. Prepare features and target
X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({"No": 0, "Yes": 1})


# 5. Identify column types
numerical_cols = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)


# 8. Create XGBoost model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# 9. Create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)


# 10. Train model
pipeline.fit(X_train, y_train)


# 11. Predictions
y_pred = pipeline.predict(X_test)

y_predict_proba = pipeline.predict_proba(X_test)[:, 1]


# 12. Evaluation metrics
metrics = {
    "Model": "XGBoost",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1_Score": f1_score(y_test, y_pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, y_predict_proba)
}


# 13. Display evaluation
print("\nXGBoost Model Evaluation")
print(pd.DataFrame([metrics]).to_string(index=False))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 14. Create output folders
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


# 15. Save trained pipeline
joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)


# 16. Save evaluation metrics
metrics_df = pd.DataFrame([metrics])

metrics_df.to_csv(METRICS_PATH, index=False)

print("\nMetrics saved at:", METRICS_PATH)


# 17. Verify saved files
print("\nModel file exists:", MODEL_PATH.exists())
print("Metrics file exists:", METRICS_PATH.exists())

print("\nDay 28 completed successfully!")

#===========(Day-29, Today topic:- Model Comparision and performance analysis)==============

# Day 29 का मुख्य उद्देश्य: पता लगाना कि कौन-सा model customer churn prediction के लिए सबसे अच्छा perform कर रहा है।

import pandas as pd
from pathlib import Path

#1. project path
PROJECT_PATH=Path(r"C:\Users\HP\Customer-Churn-Prediction")

REPORTS_PATH=PROJECT_PATH/"reports"
OUTPUT_PATH=REPORTS_PATH/"day_29_model_comparision.csv"

#2. Metrics files
metrics_files = {
    "Logistic Regression": "day22_logistic_regression_metrics.csv",
    "KNN": "day23_knn_metrics.csv",
    "Decision Tree": "day24_decision_tree_metrics.csv",
    "Random Forest": "day25_random_forest_metrics.csv",
    "SVM": "day26_svm_metrics.csv",
    "Gradient Boosting": "day_27_gradient_boosting_metrics.csv",
    "XGBoost": "day_28_xgboost_metrics.csv"
}

#3. Read and collect metrics
results=[]

for model_name,file_names in metrics_files.items():
    file_path=REPORTS_PATH/file_names
    if file_path.exists():
        df=pd.read_csv(file_path)
        df["Model"]=model_name
        results.append(df)

    else:
      raise FileNotFoundError(
         f"Metrics file not found: {file_path}"
    )


# Combine model result
if not results:
    raise FileNotFoundError(
        "No metrics file found ,check your reports folder and file names"
    )

comparison_df=pd.concat(results,ignore_index=True)

# 5. select important columns
required_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

comparison_df = comparison_df[required_columns]

#6. sort by roc-auc :
comparison_df=comparison_df.sort_values(by="ROC_AUC",ascending=False)

#7. Display comparision
print("Model performance comparision")

print(comparison_df.to_string(index=False))

#8 Find Best model
best_model=comparison_df.iloc[0]
print("Best model based on ROC_AUC")

print("Model:",best_model["Model"])
print("ROC-AUC:",best_model["ROC_AUC"])

# 9. save comparision report
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

comparison_df.to_csv(OUTPUT_PATH, index=False)

print("\nComparison report saved successfully!")

print("Saved at:", OUTPUT_PATH)

print("File exists:", OUTPUT_PATH.exists())

# ================(Day-30 , Topic name :-Model selection and performance insight)=================
# Today's objective: सातों ML models में से best-performing model select करना और उसकी performance analyze करना।

from pathlib import Path
import pandas as pd
import joblib

# set project path
BASE_DIR=Path(__file__).resolve().parent.parent
REPORT_PATH=BASE_DIR/"reports"
MODELS_PATH=BASE_DIR/"models"
COMPARISON_PATH=REPORTS_PATH/"day_29_model_comparision.csv"
OUTPUT_PATH=REPORTS_PATH/"day_30_model_selection.csv"

# Load model comparison-----------
comparison_df=pd.read_csv(COMPARISON_PATH) 

# Sort model by Roc-auc














































 

        













































    
































































    








