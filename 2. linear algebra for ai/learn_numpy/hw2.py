import numpy as np

def no_numpy_scalar(v1, v2):
    #param v1, v2: lists of 3 ints
    #YOUR CODE: please do not use numpy


    result = []

    for index in range(len(v1)):
        result.append(v1[index] * v2[index])

    return result


def numpy_scalar (v1, v2):
    #param v1, v2: np.arrays[3]
    #YOUR CODE

    result = v1 * v2
    return result

print(no_numpy_scalar([2,2,2], [1,2,3]))
print(numpy_scalar(np.array([2,2,2]), np.array([1,2,3])))