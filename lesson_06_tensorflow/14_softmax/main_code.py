import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    if x.ndim == 1:
        sum = np.sum(np.exp(x))
    elif x.ndim == 2:
        sum = np.sum(np.exp(x), axis=0)

    return np.exp(x) / sum

    # udacity solution -> easier
    #return np.exp(x) / np.sum(np.exp(x), axis=0)


logits = [3.0, 1.0, 0.2]
print(softmax(logits))

# logits is a two-dimensional array
logits = np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]])
# softmax will return a two-dimensional array with the same shape
print(softmax(logits))