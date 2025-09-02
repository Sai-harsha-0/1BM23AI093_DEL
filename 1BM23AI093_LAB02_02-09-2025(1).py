#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt


def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def relu_activation(x):
    return np.maximum(0, x)

class Perceptron:
    def __init__(self, activation_function):
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)
        self.activation_function = activation_function
        
    def predict(self, inputs):
        total_input = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(total_input)
    
    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            for input_vector, target in zip(inputs, targets):
                prediction = self.predict(input_vector)
                error = target - prediction
                self.weights += learning_rate * error * input_vector
                self.bias += learning_rate * error

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {np.mean(np.square(targets - self.predict(inputs)))}")

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])  

targets = np.array([0, 0, 0, 1])

activation_functions = {
    "Sigmoid": sigmoid_activation,
    "Tanh": tanh_activation,
    "ReLU": relu_activation,
}

for name, activation_function in activation_functions.items():
    print(f"\nTraining Perceptron with {name} Activation:")
    perceptron = Perceptron(activation_function)
    perceptron.train(inputs, targets, epochs=1000, learning_rate=0.1)
    predictions = [perceptron.predict(x) for x in inputs]
    print(f"Predictions using {name} activation: {predictions}")

def plot_activation_function():
    x = np.linspace(-10, 10, 400)
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 2)
    plt.plot(x, sigmoid_activation(x), label='Sigmoid')
    plt.title('Sigmoid Activation Function')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(x, tanh_activation(x), label='Tanh')
    plt.title('Tanh Activation Function')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(x, relu_activation(x), label='ReLU')
    plt.title('ReLU Activation Function')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_activation_function()


# In[11]:


import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size)
        self.bias = 0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return self.activation(linear_output)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            errors = 0
            print(f"Epoch {epoch+1}/{self.epochs}")
            for idx, x_i in enumerate(X):
                prediction = self.predict(x_i)
                update = self.learning_rate * (y[idx] - prediction)
                if update != 0:
                    errors += 1
                self.weights += update * x_i
                self.bias += update
                print(f"  Sample {idx+1}: Prediction={prediction}, Actual={y[idx]}, Weights={self.weights}, Bias={self.bias}")
            print(f"Epoch {epoch+1} completed with {errors} errors.\n")
            if errors == 0:
                print("Training converged.")
                break;
if __name__ == "__main__":
    
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 0, 0, 1]) 

    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
    perceptron.fit(X, y)

    print("Testing trained perceptron:")
    for sample in X:
        print(f"Input: {sample}, Predicted: {perceptron.predict(sample)}")


# In[ ]:




