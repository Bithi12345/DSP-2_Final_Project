import joblib
import pandas as pd

# Define a sample employee record
sample_employee = {
    "satisfaction_level": 0.37,
    "last_evaluation": 0.53,
    "number_project": 2,
    "average_montly_hours": 150,
    "time_spend_company": 3,
    "Work_accident": 0,
    "promotion_last_5years": 0,
    "department": "sales",
    "salary": "high"
}

# Convert to DataFrame
sample_employee_df = pd.DataFrame([sample_employee])

# Encode categorical features manually (if necessary)
salary_mapping = {"low": 0, "medium": 1, "high": 2}
sample_employee_df["salary"] = sample_employee_df["salary"].map(salary_mapping)

print("Sample Employee Data:")
print(sample_employee_df)