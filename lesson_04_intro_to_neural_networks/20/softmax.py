import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    L_exp = np.exp(L)
    L_sum = sum(L_exp)
    return_list = []
    for i in L_exp:
        return_list.append(i / L_sum)
    return return_list


print(softmax([1]))

print(softmax([1, 1]))

print(softmax([1, 2]))
