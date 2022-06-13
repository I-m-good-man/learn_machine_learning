import numpy as np  # не стирать!


def diag_2k(a):
    # param a: np.array[size, size]
    # YOUR CODE

    new_a = np.array([])

    for index in range(len(a)):

        if a[index, index] % 2 == 0:
            new_a = np.append(new_a, a[index, index])

    if new_a.size == 0:
        return 0
    else:
        return new_a.sum()



print(diag_2k(np.array([[1,2], [3,4]])))


