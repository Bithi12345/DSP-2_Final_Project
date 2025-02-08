import streamlit as st
import pandas as pd
import joblib

# Set up the title
st.title("Employee Attrition Prediction App")

# Sidebar for user input
st.sidebar.header("Please enter Employee Details")

# User inputs
satisfaction_level = st.sidebar.slider("Satisfaction Level", 0.0, 1.0, 0.38)
last_evaluation = st.sidebar.slider("Last Evaluation Score", 0.0, 1.0, 0.53)
number_project = st.sidebar.number_input("Number of Projects", min_value=1, max_value=10, value=2)
average_montly_hours = st.sidebar.number_input("Average Monthly Hours", min_value=50, max_value=350, value=157)
time_spend_company = st.sidebar.number_input("Years Spent in Company", min_value=1, max_value=20, value=3)
Work_accident = st.sidebar.selectbox("Work Accident", [0, 1])
promotion_last_5years = st.sidebar.selectbox("Promotion in Last 5 Years", [0, 1])
Departments = st.sidebar.selectbox("Departments", ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management'])
salary = st.sidebar.selectbox("Salary Level", ["low", "medium", "high"])

work_hours_per_project = average_montly_hours / number_project if number_project > 0 else 0

# Encode categorical variables
salary_mapping = {"low": 0, "medium": 1, "high": 2}
salary_encoded = salary_mapping[salary]

# Create DataFrame for prediction
input_data = {
    "satisfaction_level": satisfaction_level,
    "last_evaluation": last_evaluation,
    "number_project": number_project,
    "average_montly_hours": average_montly_hours,
    "time_spend_company": time_spend_company,
    "Work_accident": Work_accident,
    "promotion_last_5years": promotion_last_5years,
    "Departments": Departments,
    "salary": salary_encoded,
    "work_hours_per_project": work_hours_per_project
}

input_data_df = pd.DataFrame([input_data])

# Load trained model
model_path = "..\models\model_with_pipeline.pkl"


try:
    model = joblib.load(model_path)
    
    # Make a prediction
    result = model.predict(input_data_df)

    # Display input data
    st.table(input_data_df)

    # Display prediction result
    prediction = "Employee Will Leave" if result[0] == 1 else "Employee Will Stay"
    #st.metric("Prediction", prediction)
    st.metric('Predicted Loan Default:', f'{result[0]:,.2f}')

except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Train and save the model first.")