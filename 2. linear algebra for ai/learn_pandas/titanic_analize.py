import pandas as pd
import numpy as np


desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',100)

# pass_link = 'https://www.dropbox.com/s/lyzcuxu1pdrw5qb/titanic_data.csv?dl=1'
# titanic_passengers = pd.read_csv(pass_link)

titanic_passengers = pd.read_csv('titanic_data.csv')

# выводим первые 10 строк
first_10_strings = titanic_passengers.head(10)
print(first_10_strings)

print('--------------')

# выводим последние 10 строк
last_10_strings = titanic_passengers.tail(10)
print(last_10_strings)

print('---------------------------')

# считываем заново, чтобы указать индексами колонку PassengerId
titanic_passengers = pd.read_csv('titanic_data.csv', index_col='PassengerId')
print(titanic_passengers.tail(10))

# shape показывает сначала количество строк, затем количество столбцов
print(titanic_passengers.shape)

# можно также выводить подробную информацию о столбцах
print(titanic_passengers.info)

# можно выводить статистическую информацию о столбцах численного типа
print(titanic_passengers.describe())

print('-------------------------')

print(titanic_passengers.head(10))

print('-------------------------')

# выводим разные значения серии при помощи разным методов
print(titanic_passengers['Age'].min(), titanic_passengers['Age'].max(),
      titanic_passengers['Age'].mean())

# value_counts выводит число, сколько раз встречается определеноое значение в серии
print(titanic_passengers['Sex'].value_counts())

print('------------------')

# можно группировать дата фрейм по какому то столбцу. т.е. все возможные значения серии Pclass
# становятся индексами нового датафрейма. из этого объекта можно вытащить интересные статистики
print(titanic_passengers.groupby('Pclass').mean())

print('---------------------------')

df2 = pd.read_csv('titanic_surv.csv')
print(df2.head(10))

print('---------------------------')

"""
Можно сливать два датафрейма в один при помощи метода join(). Этот метод сопоставляет совпадающие
индексы двух датафреймов и добавляет в один датафрейм новые столбцы с элементами.
"""

df1 = pd.read_csv('titanic_data.csv')
df2 = pd.read_csv('titanic_surv.csv')
print(df1.head(10))
print(df2.head(10))

# вот так мы присоединили к df1 df2
df3 = df1.join(df2)
print(df3.head(10))

# можно суммировать все значения серии при помощи метода sum()
print(df3['Age'].sum())

print('----------------------------------')

# создали новый дата фрейм corr_data
corr_data = df3[['Sex', 'Age', 'Survived']]
# создали маску, если женщина, то True
mask_for_Sex_column = corr_data['Sex'] == 'female'
print(mask_for_Sex_column.head(10))
# меняем все значения bool на соотв. числа, т.е. True на 1, False на 0
int_sex_column = mask_for_Sex_column.astype(int)
print(int_sex_column.head(10))
# меняем столбец в датафрейме для корреляции
corr_data['Sex'] = int_sex_column
print(corr_data.head(10))

print(corr_data.corr())






