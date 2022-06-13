import numpy as np

a= np.array([])
a = np.append(a, 1)
print(a)


def encode(a):
    #YOUR CODE

    unique_array = np.array([a[0]])

    for index in range(1, len(a)):
        if unique_array[-1] != a[index]:
            unique_array = np.append(unique_array, a[index])

    num_array = np.array([])

    current_el = None
    flag = False
    current_counter = 0

    for index in range(len(a)):
        if index == len(a)-1:
            if a[index] == current_el:
                current_counter += 1
                num_array = np.append(num_array, current_counter)
            else:
                num_array = np.append(num_array, current_counter)
                num_array = np.append(num_array, 1)


        else:

            if flag:
                if a[index] == current_el:
                    current_counter += 1
                else:
                    num_array = np.append(num_array, current_counter)
                    current_counter = 1
                    current_el = a[index]
                    flag = True
            else:
                flag = True
                current_el = a[index]
                current_counter += 1

    return unique_array, num_array


a = np.array([1, 2, 2, 3, 3, 1, 1, 5, 5, 2, 3, 3])
print(encode(a))





