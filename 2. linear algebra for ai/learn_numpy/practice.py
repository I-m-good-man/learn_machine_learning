"""
numpy полезен тем, что позволяет работать с массивами быстрее чем с дефолтными структурами 
данных в питоне за счет низкоуровневой организей из за того что написан на C.
"""
import numpy as np
import math


"""
Чтобы создать numpy-массив, нужно чтобы исходный список был однородным, т.е. чтобы тип данных
был одинаковым, т.к. numpy-массив может хранить данные только одного типа.
Также можно указать тип данных, который будет использован в numpy-массиве, например float64.

"""
python_num_list = [1, 2, 3, 4, 5]
np_num_array = np.array(python_num_list, dtype="float64")
print(type(python_num_list), type(np_num_array))
print(python_num_list, np_num_array)

print('------------------')

"""
Посмотрим все методы numpy-массива, которых нет у обычного списка.
"""
print([method for method in dir(np_num_array) if method not in dir(python_num_list)])

print('------------------')

# воспользуемся методом, который показывает вес массива. 5 чисел, каждое весит 4 байта, значит
# размер массива будет 20 байтов, верно)
np_int32_array = np.array([1, 2, 3, 4, 5], dtype='int32')
print(np_int32_array.nbytes)

print('------------------')

# методы и атрибуты массива
multi_size_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
np_multi_size_array = np.array(multi_size_array, dtype='int32')
print(f'Количество элементов по первой оси(3): {len(np_multi_size_array)}')
print(f'Всего элементов в матрице (9): {np_multi_size_array.size}')
print(f'Количество элементов по каждой оси: {np_multi_size_array.shape}')
print(f'Размерность матрицы, т.е. уровней вложенности(2): {np_multi_size_array.ndim}')
print(f'Тип элементов массива: {np_multi_size_array.dtype}')

print('------------------')

# обращение к элементам по индексу
print(np_int32_array[0])
# их можно изменять
np_int32_array[0] = 11
print(np_int32_array[0])

"""
numpy-массив можно итерировать, но лучше не надо, потому что это будет очень медленно,
теряется главное преимущество
"""
for i in np_int32_array:
    print(i, end=' ')

print('------------------')

# можно создавать массивы, состоящие только из нулей или единиц
zero_array = np.zeros(10, dtype='int64')
ones_array = np.ones(10, dtype='float32')
print(zero_array, ones_array)

print('------------------')

# аналог функции range в numpy
# интервалы полуоткрытые
for i in np.arange(1, 10, 2):
    print(i, end=' ')

print('')

# можно использовать нецелые числа
for i in np.arange(0, 1.1, 0.1):
    print(i, end=' ')

print('')

print('------------------')

"""
если нам нужно построить последовательность чисел, для которой известный левые и правые границы
а также количество чисел, которые должны находится между этими границами, притом границы
включаются в этот массив, то мы можем воспользоваться следющим методом
"""
array1 = np.linspace(10, 11, 5)
print(array1)

print('------------------')

"""
Над одномернымимассивами можно производить арифметические операции (+-/*), но только если массивы
одномерны и их размеры равны. Арифм. операции используются попеременно.
"""
a = np.linspace(10, 70, 7)
b = np.arange(1.0, 4.5, 0.5)
print(a)
print(b)

print(a+b)
print(a-b)
print(a/b)
print(a*b)

# также можно умножать массив на скаляр и прибавить скаляр
print(a*10)
print(a+10)

# еще можно возводить какое то число в степень массива, в этом случае результатом будем массив,
# элементы которого будут степенями, взятыми из того массива, данного числа
print(2**np.array([1, 2, 3, 4, 5], dtype='int32'))

# также можно возводить в степень массив, это будет эквивалентно умножению массива на массив
print(np.array([1, 2, 3, 4, 5], dtype='int32')**2)

print('-------------------')

# есть еще элементарные функции - это всякие тригонометрические функции, экспонента и тд
print(np.cos(math.pi))
# если применить элементарную функцию к массиву, то мы получим массив с обработанными элементами
print(np.exp(b))

print('--------------------')

# кванторы всеобщности(A-перевернутая) и существования(E-развернутая)
array_of_1 = np.ones(10, 'int32')
# в массиве array_of_1 любой элемент равен 1
print(f'Квантор всеобщности: {array_of_1.all()==1}')

# в массиве array_of_1 существует элемент, равный 1
print(f'Квантор существования: {array_of_1.any()==1}')

# в массиве array_of_1 существует элемент, равный 0
print(f'Квантор существования: {array_of_1.any()==0}')

print('------------------')

# есть удобные константы
print(np.pi)
print(np.exp(1))

print('------------------')

# cumsum - камулятивная сумма, аналог факториала, но только для суммы
print(a)
print(a.cumsum())

print('------------------')

# сортировка массива
unsorted_array = np.arange(100, -1, -10)
print(unsorted_array)

sorted_array = np.sort(unsorted_array)
print(sorted_array)

unsorted_array.sort()
print(unsorted_array)

print('------------------')

# можно объединять массивы
a = np.arange(10)
b = np.arange(10, 20)
c = np.arange(20, 30)

print(a, b, c)
d = np.hstack((a, b, c))
print(d)
a = np.append(a, 100)
print(a)
print(d)

print('------------------')

# можно расщеплять массивы по индексам
a, b, c = np.hsplit(d, [10, 20])
print(d)
print(a, b, c)

print('----------------')

# двумерные массивы
dvumer_array = np.array([[1, 2], [3, 4]])
print(dvumer_array)
print(dvumer_array[0][0])

print('------------------')
# можно менять многомерность массива, получается что то типо линейного оператора
b = np.arange(10)
print(b)
b.shape = 5, 2
print(b)
b.shape = 2, 5
print(b)

print('-------------------')

# можно приводить многомерный массив к одномерному виду
print(b)
b = b.ravel()
print(b)

print('-------------------')

# можно создавать многомерный массив из нулей и единиц, указав shape матрицы
shape_of_matrix = (3, 4)
mnogomern_array_of_ones = np.ones(shape_of_matrix)
print(mnogomern_array_of_ones)
# аналогично с нулями
mnogomern_array_of_zeroes = np.zeros(shape_of_matrix)
print(mnogomern_array_of_zeroes)

print('-------------------')

# единичная квадратная матрица, указываем ее порядок
c = np.eye(5)
print(c)

print('-------------------')

# можно создавать диагональные матрицы, указываем элементы на диагонали
diag_matrix = np.diag([1, 2, 3, 4])
print(diag_matrix)

print('-------------------')

# поэлементное умножение матриц, умножается каждый жлемент на какой то коэффициент
a = np.ones((5,5))
a *= 5
print(a)

b = np.ones((5,5))
b *= 2
print(b)

c = a*b
print(c)

print('-------------------')

# матричное умножение, нужно чтобы число столбцов одной матрицы было равно числу строк другой матрицы
a = np.ones((5,5))
a *= 5
print(a)

b = np.ones((5,5))
b *= 2
print(b)

# матричное умноежние
c = a @ b
d = a.dot(b) # аналогично a @ b. тут же важен порядок множителей
print(c)
print(c == d)

print(np.dot(a, b))
print('\n'*10)
print('-------------------')

"""
Можно использовать маски. маска - это массив той же самой формы, что и исходный, но его элементы - 
это булевы значения, т.е. True и False. Составляется маска на основе какого то условия
"""

array_of_0_and_1 = np.diag([1 for i in range(10)])
print(array_of_0_and_1)
mask_of_array_of_0_and_1 = array_of_0_and_1 == 1
print(mask_of_array_of_0_and_1)

"""
также маску можно передавать в качестве индекса и тогда мы получим массив, содержащий только те
элементы, которым соответствуют элементы True маски
"""
print(array_of_0_and_1[mask_of_array_of_0_and_1])

print('-------------------')

diag_matrix = np.diag([i for i in range(10)])
print(diag_matrix)
# метод trace считает сумму диагональных элементов
print(diag_matrix.trace(), sum([i for i in range(10)]))

print('-------------------')

# тензор - это многомерный массив
odnomern_array = np.arange(64)
print(odnomern_array)
# метод reshape меняет размер матрицы
tensor = odnomern_array.reshape(8, 2, 4)
print(tensor)
print(tensor.size, tensor.shape, len(tensor), tensor.ndim)

print('-------------------')

# сумма тензора
print(np.sum(tensor))

# можной находить суммы тензора по разным его осям
# на нулевой оси все списки содерат в себе по 2 элемента, поэтому происходит поэлементное
# складывание соответствующих элементов в соответствующих списках
print(np.sum(tensor, axis=0))
# аналогично для остальных осей
print(np.sum(tensor, axis=1))
print(np.sum(tensor, axis=2))

print('--------------')

"""

Линейная алгебра в numpy. Она находится в numpy.linalg

"""

matrix_2_to_2 = np.array([[1, 2], [3, 4]])
print(matrix_2_to_2)
det_matrix_2_to_2 = np.linalg.det(matrix_2_to_2)
print(det_matrix_2_to_2)

print('-----------')

# обратная матрица
if det_matrix_2_to_2 != 0:
    inv_matrix_2_to_2 = np.linalg.inv(matrix_2_to_2)
    print(inv_matrix_2_to_2)

# если умножить прямую матрицу на обратную, то получим еиничную матрицу
eye_matrix = inv_matrix_2_to_2 @ matrix_2_to_2
print(eye_matrix)

print('-----------')

# иногда Numpy может неправильно высчитывать определитель матрицы, он может представлять его как
# очень маленькое, близкое  к нулю число, но не нуль, и поэтому нужно внимательнее проверять резы
zero_det_matrix = np.array([[1, 2], [1, 2]])
det_of_zero_det_matrix = np.linalg.det(zero_det_matrix)
print(det_of_zero_det_matrix)

print('-----------')

# решение неоднородных линейных уравнений, т.е. A*X=B
matrix_of_factors = np.array([[1, -2, 1], [2, -1, 0], [3, 2, -1]])
matrix_of_free_members = np.array([0, 1, 4])
matrix_of_solve = np.linalg.solve(matrix_of_factors, matrix_of_free_members)
print(matrix_of_solve)

# решим это уравнение сами
matrix_of_solve = (np.linalg.inv(matrix_of_factors)) @ matrix_of_free_members
print(matrix_of_solve)

print('---------------')

# транспонирование матрицы
diag_matrix = np.array([[1,1,1,1], [0,0,0,0], [0,0,0,0]])
trans_diag_matrix = diag_matrix.T
print(diag_matrix)
print(trans_diag_matrix)




