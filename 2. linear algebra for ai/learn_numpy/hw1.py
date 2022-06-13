import numpy as np

def no_numpy_mult(first, second):
    """
    param first: list of "size" lists, each contains "size" floats
    param second: list of "size" lists, each contains "size" floats
    """

    #YOUR CODE: please do not use numpy

    def get_column_list_from_matrix(num, matrix):

        result = []
        for i in matrix:
            result.append(i[num])
        return result

    result = [[] for i in range(len(first))]

    for string_index in range(len(first)):
        string_elems = first[string_index]
        for column_index in range(len(second)):
            column_elems = get_column_list_from_matrix(column_index, second)
            elem = sum([i*j for i, j in zip(string_elems, column_elems)])
            result[string_index].append(elem)
    return result


def numpy_mult(first, second):
    """
    param first: np.array[size, size]
    param second: np.array[size, size]
    """

    #YOUR CODE: please use numpy

    result = first @ second
    return result

print(no_numpy_mult([[2 for i in range(3)] for i in range(3)], [[3 for i in range(3)] for i in range(3)]))





