import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from datetime import datetime
current_dateTime = datetime.now()

formatted_datetime = current_dateTime.strftime("%Y-%m-%d_%H-%M-%S")

# Load dataset
df = pd.read_excel('d_1.xlsx')


X = df.drop('Outcome', axis=1)  # Replace 'TargetColumn' with the name of your target column
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



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



new_data = pd.read_excel('d_2.xlsx')

new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)

# Convert predictions to labels (e.g., 0 and 1)
predicted_labels = (predictions > 0.5).astype(int)

predictions_df = pd.DataFrame(predicted_labels, columns=['Predicted_Outcome'])

# Concatenate original DataFrame with predictions DataFrame
result_df = pd.concat([df, predictions_df], axis=1)

# Export the result to a new Excel file
result_df.to_excel(f'predicted_results_date_{formatted_datetime}.xlsx', index=False)