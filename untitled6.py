import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
dataset_path = r"C:\Users\eng\Desktop\data\diabetes.csv"
dataset = pd.read_csv(dataset_path)

# Extract the numeric columns
numeric_columns = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
dataset = dataset.dropna()

# Convert the data to NumPy array
X = dataset[numeric_columns].values
y = dataset['Outcome'].values.reshape(-1, 1)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the dimensions for the new test_data
Pregnancies_value = 5
BloodPressure_value = 120
SkinThickness_value = 30
Insulin_value = 150
BMI_value = 25
DiabetesPedigreeFunction_value = 0.5
Age_value = 35

test_data = np.array([[Pregnancies_value, BloodPressure_value, SkinThickness_value, Insulin_value, BMI_value, DiabetesPedigreeFunction_value, Age_value]])

# Scale the new test_data
test_data = scaler.transform(test_data)

# Build the model
hidden_size = 16
model = MLPClassifier(hidden_layer_sizes=(hidden_size,), max_iter=1000, random_state=1)

# Train the model
model.fit(X, y)

# Neural network predictions
predictions_nn = model.predict(X)

# Expert system predictions
predictions_expert = ...  # Expert system rules

# Calculate model accuracy
accuracy = accuracy_score(y, predictions_nn)
print("Accuracy (Neural Network):", accuracy)

# Print expert system predictions
print("Expert System Predictions:", predictions_expert)

# Final system prediction
final_prediction = ...  # Final system prediction

print("Final Prediction:", final_prediction)