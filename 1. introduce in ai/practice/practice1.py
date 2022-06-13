"""

функция map

Общий вид: map(func, iterables)

"""

num_list = [i for i in range(1,21)]


def sq(num):
    return num ** 2


changed_num_list = list(map(sq, num_list))
print(changed_num_list)


string_list = [sym for sym in 'abcdefghij']

# мы можем применять к элементам коллекции методы типов, указав предварительно тип
changed_string_list = list(map(str.upper, string_list))
print(changed_string_list)

# также можно работать одновременно с несколькими коллекциями, функция принимает соотв. число элементов
# из коллекций берутся элементы, пока в одной из них они не кончатся

strange_list = list(map(lambda sym, num: f'{num}{sym}', string_list, num_list))
print(strange_list)


"""

функция filter

Функция фильтр предлагает элегантный вариант фильтрации последовательности.
Общий вид: filter(func, iterables)

Передаваемая функция должна возвращать True или False!

"""


sym_list = ['apple iphone 13', 'samsung galaxy s10', 'xiaomi mi 11', 'oneplus 8t', 'oneplus 9pro']

filtred_sym_list = list(filter(lambda sym: 'oneplus' in sym, sym_list))
print(filtred_sym_list)


"""

функция zip

общий вид: zip(iterables)
Функция заключает переданные в нее коллекции в последовательность.

"""

l1 = [i for i in range(10)]
l2 = [i for i in range(10, 20)]
l3 = [i for i in range(20, 30)]

zipped_list = list(zip(l1, l2, l3))
print(zipped_list)


"""

reduce

Эта функция нужна для того, чтобы свести все элементы какой то последовательности к единств. знач.

общий вид: reduce(function, iterable)

function - функция, которая в свой первый аргумент принимает так называемое аккумулирующее значение,
а вторым элементом - элементы последовательности

"""

num_list = [i for i in range(10)]

# найдем сумму списка num_list

from functools import reduce

sum_of_num_list = reduce(lambda acc, el: acc+el, num_list)

print(sum_of_num_list)


"""

Можно использовать tqdm для осознания прогресса итерации какой либо коллекции.

"""


from tqdm import tqdm as show_progress
from tqdm import trange


for i in show_progress(range(10000000)):
    pass


for i in trange(10000000):
    pass


for i in trange(10):
    for j in trange(5):
        pass









