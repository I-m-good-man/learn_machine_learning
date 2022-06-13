import pandas as pd
import numpy as np

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',100)

dataframe = pd.read_csv('BlackFriday.csv')
print(dataframe.head())
print(dataframe)


