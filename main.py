import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data from Excel file (replace 'filename.xlsx' with your actual file name)
# data = pd.read_excel('diabetes.xlsx')
data = pd.read_excel('hos_data_diabities.xlsx')

# Separate features and labels
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(12, input_shape=(8,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Use the model to predict outcomes for new patients
# Load data for new patients (replace 'new_patients.xlsx' with the actual file name)
new_data = pd.read_excel('hos_data_diabities_without_outcome.xlsx')

# Preprocess the new data similarly as done before
new_X_scaled = scaler.transform(new_data)

# Predict outcomes for new patients
predictions = model.predict(new_X_scaled)
predictions_rounded = (predictions > 0.5).astype(int).flatten()

# Save predictions to an Excel file
predictions_df = pd.DataFrame({'Predictions': predictions_rounded})
predictions_df.to_excel('predictions.xlsx', index=False)
