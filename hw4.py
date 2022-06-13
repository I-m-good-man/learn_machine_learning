def cumsum_and_erase(A, erase=1):

    B = []

    for index in range(len(A)):
        B.append(sum(A[:index+1]))

    C = list(filter(lambda el: True if el != erase else False, B))
    return C



A = [5, 1, 4, 5, 14]
B = cumsum_and_erase(A, erase=10)
# assert B == [5, 6, 15, 29], "Something is wrong! Please try again"







