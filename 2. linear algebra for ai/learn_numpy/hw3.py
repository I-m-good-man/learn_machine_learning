import numpy
import numpy as np

def transform(X, a=1):
    """
    param X: np.array[batch_size, n]
    """

    copy_X = np.ndarray.copy(X)

    def string_perfromance(string):
        last_string = np.ndarray.copy(string)
        for index in range(len(string)):
            if index % 2 == 1:
                string[index] = a
            else:
                string[index] **= 3

        string = string[::-1]
        merged_strings = np.concatenate([last_string, string])
        return merged_strings

    X_shape = X.shape
    new_X_shape = (X_shape[0], X_shape[1]*2)
    new_X = np.zeros(new_X_shape)

    for index1, string in enumerate(copy_X):
        for index2, el in enumerate(string_perfromance(string)):
            new_X[index1][index2] = el
    #YOUR CODE

    return new_X

print(transform(np.array([[1,2,3,4,5],[1,2,3,4,5]])))







