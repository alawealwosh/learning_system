import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_excel('Dataset_of_Diabetes_Iraq.xlsx')

# Label encode the target variable
label_encoder = LabelEncoder()
df['CLASS'] = label_encoder.fit_transform(df['CLASS'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

X = df.drop('CLASS', axis=1)  # Replace 'TargetColumn' with the name of your target column
y = df['CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

# Assuming new_data is your new Excel data loaded and preprocessed (without the target variable)

new_data = pd.read_excel('Dataset_of_Diabetes_Iraq_No_Class.xlsx')

# One-hot encode categorical columns in the new data
# new_data = pd.get_dummies(new_data, columns=['Gender'])
# new_data['CLASS'] = label_encoder.fit_transform(new_data['CLASS'])
new_data['Gender'] = label_encoder.fit_transform(new_data['Gender'])


new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)

# Convert predictions to labels (e.g., 0 and 1)
predicted_labels = (predictions > 0.5).astype(int)

# Map predictions back to original class labels
predicted_labels = label_encoder.inverse_transform(predicted_labels.flatten())

predictions_df = pd.DataFrame(predicted_labels, columns=['Predicted_Class'])

# Concatenate original DataFrame with predictions DataFrame
result_df = pd.concat([df, predictions_df], axis=1)

# Export the result to a new Excel file
result_df.to_excel('predicted_results__Iraqi_hospital.xlsx', index=False)
