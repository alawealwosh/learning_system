import numpy as np

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias1 = np.random.randn(self.hidden_size)
        self.bias2 = np.random.randn(self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, X):
        self.hidden_layer = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights2) + self.bias2)
        return self.output_layer

    def backpropagation(self, X, y, learning_rate):
        error = y - self.output_layer
        delta_output = error * self.output_layer * (1 - self.output_layer)
        error_hidden = np.dot(delta_output, self.weights2.T)
        delta_hidden = error_hidden * self.hidden_layer * (1 - self.hidden_layer)
        self.weights2 += learning_rate * np.dot(self.hidden_layer.T, delta_output)
        self.bias2 += learning_rate * np.sum(delta_output, axis=0)
        self.weights1 += learning_rate * np.dot(X.T, delta_hidden)
        self.bias1 += learning_rate * np.sum(delta_hidden, axis=0)

# Define the expert system rules
def expert_system(age, bmi, glucose, insulin):
    if age >= 40 and bmi >= 30 and glucose >= 140 and insulin >= 100:
        return "Positive"
    elif age >= 40 and bmi >= 30 and glucose >= 140:
        return "Positive"
    elif age >= 40 and bmi >= 30 and insulin >= 100:
        return "Positive"
    else:
        return "Negative"

# Define the main function
def main():
    # Load the dataset
    dataset_path = r'C:\Users\eng\Desktop\data\diabetes.csv'
    dataset = np.loadtxt(dataset_path, delimiter=',', skiprows=1)
    X = dataset[:, :-1]
    y = dataset[:, -1].reshape(-1, 1)

    # Normalize the input features
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X = (X - mean_X) / std_X

    # Initialize the neural network
    input_size = X.shape[1]
    hidden_size = 3
    output_size = 1
    neural_network = NeuralNetwork(input_size, hidden_size, output_size)

    # Train the neural network using backpropagation
    epochs = 1000
    learning_rate = 0.1
    for epoch in range(epochs):
        neural_network.feedforward(X)
        neural_network.backpropagation(X, y, learning_rate)

    # Test the neural network
    test_data = np.array([[48, 34.9, 125, 82]])
    test_data = (test_data - mean_X) / std_X.reshape(1, -1)
    prediction = neural_network.feedforward(test_data)
    print("Neural Network Prediction:", prediction)

    # Use the expert system to make a final prediction
    age = test_data[0][0]
    bmi = test_data[0][1]
    glucose = test_data[0][2]
    insulin = test_data[0][3]
    expert_prediction = expert_system(age, bmi, glucose, insulin)
    print("Expert System Prediction:", expert_prediction)

# Run the main function
if __name__ == "__main__":
    main()
