import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import joblib

filepath = "C:/DSP-2_Final_Project/data/HR_comma_sep.csv"

# Load dataset
df = pd.read_csv(filepath)

# Handle duplicates
df = df.drop_duplicates()

# Ensure column names are clean
df.columns = df.columns.str.strip()

# Debugging: Check available columns
print("Columns in DataFrame:", df.columns.tolist())

# Encode categorical variables
salary_mapping = {"low": 0, "medium": 1, "high": 2}
df["salary"] = df["salary"].map(salary_mapping)

encoder = OrdinalEncoder()

# Ensure reshaping for OrdinalEncoder
df["Departments"] = encoder.fit_transform(df[["Departments"]].values.reshape(-1, 1))

# Feature Engineering: Create a new feature
df["work_hours_per_project"] = df["average_montly_hours"] / df["number_project"]

# Define features and target
features = ["satisfaction_level", "last_evaluation", "number_project",
            "average_montly_hours", "time_spend_company", "Work_accident",
            "promotion_last_5years", "Departments", "salary", "work_hours_per_project"]
target = "left"

# Splitting into training and testing sets (80-20 split)
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df["left"])

# Save train and test sets
os.makedirs("data", exist_ok=True)
train_set.to_csv("data/train.csv", index=False)
test_set.to_csv("data/test.csv", index=False)

# Reload training set
train_set = pd.read_csv("data/train.csv")

# Splitting into features (X) and target (y)
X_train = train_set.drop("left", axis=1)
y_train = train_set["left"].copy()

# Creating a validation set (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes("object").columns.tolist()

print(f"Numerical columns: {num_cols}", f"Categorical columns: {cat_cols}")

# Handle missing values
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Normalize numerical features
scaler = StandardScaler()

X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_val[num_cols] = num_imputer.transform(X_val[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])

# Apply categorical imputations (if any categorical columns exist)
if cat_cols:
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = cat_imputer.transform(X_val[cat_cols])

print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict on validation set
y_pred_log = log_reg.predict(X_val)

# Evaluate model
log_accuracy = accuracy_score(y_val, y_pred_log)
print(f"Logistic Regression Accuracy: {log_accuracy}")

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=120, random_state=42)
rf.fit(X_train, y_train)

# Predict on validation set
y_pred_rf = rf.predict(X_val)

# Evaluate model
rf_accuracy = accuracy_score(y_val, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy}")


#save the model
os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/encoder.pkl")
joblib.dump(num_imputer, "models/num_imputer.pkl")
joblib.dump(cat_imputer, "models/cat_imputer.pkl")

print("Model and preprocessing objects saved successfully!")