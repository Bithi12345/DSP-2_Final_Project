import joblib
import pandas as pd


fpath = "C:/DSP-2_Final_Project/src/models/hr_model_pipeline.pkl"
# Load the trained model
model = joblib.load(fpath)

# Load test dataset
ffpath = "C:/DSP-2_Final_Project/data/test.csv"
test_data = pd.read_csv(ffpath)

# Splitting features and target
X_test = test_data.drop("left", axis=1)
y_test = test_data["left"].copy()

# Predict using the loaded model
y_preds = model.predict(X_test)

from sklearn.metrics import accuracy_score

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_preds)

print(f"Test Accuracy: {accuracy:.4f}")

# Get a random sample from test data
sample = test_data.sample(1).to_dict()

print("Random Test Sample:", sample)