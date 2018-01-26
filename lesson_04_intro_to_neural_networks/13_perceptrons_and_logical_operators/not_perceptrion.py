import pandas as pd
from generate_and_check import generate_and_check

# TODO: Set weight1, weight2, and bias
weight1 = 0.0
weight2 = -1.0
bias = 0.0


# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [True, False, True, False]

generate_and_check(test_inputs, correct_outputs, weight1, weight2, bias)
