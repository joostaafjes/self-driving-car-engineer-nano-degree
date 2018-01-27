import numpy as np

def sigmoid(x):
    # Implement sigmoid function
    return 1 / (1 + np.exp(-x))

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

# Calculate the output
output = sigmoid(np.sum(np.dot(inputs, weights)) + bias)

print('Output:')
print(output)