This project focuses on predicting employee attrition (whether an employee will leave or stay) using machine learning techniques. The dataset used is HR_comma_sep.csv, containing employee-related features such as satisfaction level, number of projects, working hours, promotions, salary, and department.


1. Loaded and Preprocessed Data: Read the HR dataset (`HR_comma_sep.csv`) and performed data cleaning, including handling missing values and encoding categorical features.  

2. Feature Engineering: Created new features such as "Work Hours Per Project" by dividing average monthly hours by the number of projects.  

3. Data Splitting: Divided the dataset into training and testing sets to ensure proper model evaluation.  

4. Data Transformation Pipeline: Built a preprocessing pipeline using `Pipeline` and `ColumnTransformer` to handle numerical and categorical features efficiently.  

5. Model Training: Trained multiple models, including Linear Regression and Random Forest to predict employee attrition.  

6. Model Evaluation: Used Root Mean Squared Error (RMSE) and other metrics to evaluate model performance on the validation set.  

7. Model Saving and Deployment: Saved the trained model using `joblib` for later use in predictions.  

8. Streamlit Web App: Developed an interactive web application where users can input employee details to predict attrition.  

9. User-Friendly Interface: Used sliders, dropdowns, and tables in Streamlit to make it easy for users to enter data and view predictions.  

10. Deployment Readiness: Prepared the model and app for deployment on  Streamlit Cloud for real-world usage.
