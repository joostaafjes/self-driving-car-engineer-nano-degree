import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

print(cross_entropy([1], [0.6]))

print(cross_entropy([1], [0.9]))

print(cross_entropy([1, 0], [0.1, 0.9]))

print(cross_entropy([0, 1], [0.5, 0.5]))

print(cross_entropy([1,0,1,1], [0.4,0.6,0.1,0.5]))