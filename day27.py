# ================================
# Loan Prediction Using ML (SVM)
# ================================

# Importing Dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score

# ================================
# Data Collection
# ================================

loan_dataset = pd.read_csv("loandata.csv")

# ================================
# Data Cleaning
# ================================

loan_dataset = loan_dataset.dropna()

# Encode target variable
loan_dataset['Loan_Status'] = loan_dataset['Loan_Status'].map({'N': 0, 'Y': 1})

# Replace '3+' dependents with 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# Encode binary categorical columns
loan_dataset['Married'] = loan_dataset['Married'].map({'No': 0, 'Yes': 1})
loan_dataset['Gender'] = loan_dataset['Gender'].map({'Male': 1, 'Female': 0})
loan_dataset['Self_Employed'] = loan_dataset['Self_Employed'].map({'No': 0, 'Yes': 1})

# ================================
# Data Visualization (Optional)
# ================================

sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
plt.show()

sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
plt.show()

# ================================
# Feature & Target Separation
# ================================

X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

# ================================
# One-Hot Encoding
# ================================

categorical_cols = ['Dependents', 'Education', 'Property_Area']
X[categorical_cols] = X[categorical_cols].astype(str)
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Ensure alignment
X = X.dropna()
Y = Y.loc[X.index]

# Encode target
le = LabelEncoder()
Y = le.fit_transform(Y)

# ================================
# Train-Test Split
# ================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=2
)

# ================================
# Model Training (SVM)
# ================================

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# ================================
# Model Evaluation
# ================================

train_pred = classifier.predict(X_train)
test_pred = classifier.predict(X_test)

print("Training Accuracy:", accuracy_score(Y_train, train_pred))
print("Testing Accuracy :", accuracy_score(Y_test, test_pred))

# ================================
# Prediction System
# ================================

feature_names = X.columns

def loan_prediction_system(
    Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area
):
    input_data = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    prediction = classifier.predict(input_df)

    if prediction[0] == 1:
        return "Loan Approved"
    else:
        return "Loan Not Approved"

# ================================
# Sample Prediction
# ================================

result = loan_prediction_system(
    Gender=1,
    Married=1,
    Dependents='1',
    Education='Graduate',
    Self_Employed=0,
    ApplicantIncome=5000,
    CoapplicantIncome=2000,
    LoanAmount=150,
    Loan_Amount_Term=360,
    Credit_History=1.0,
    Property_Area='Urban'
)

print("Prediction Result:", result)
