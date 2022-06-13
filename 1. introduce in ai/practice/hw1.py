def almost_double_factorial(n):
    if n == 0:
        return 1
    else:
        proizv = 1
        for i in range(1, n + 1):
            if i % 2 != 0:
                proizv *= i
        return proizv


print(almost_double_factorial(7))

