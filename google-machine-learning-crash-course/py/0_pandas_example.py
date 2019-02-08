from __future__ import print_function
import os
import pandas as pd
import numpy as np

dirname = os.path.dirname(__file__)

# print('Pandas Version: ' + pd.__version__)

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

dataframe = pd.DataFrame({ 'City name': city_names, 'Population': population })

# print(dataframe)

csv = os.path.join(dirname, '../datasets/california_housing_train.csv')
# california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = pd.read_csv(csv, sep=",")

# describe the DataFrame
# print(california_housing_dataframe.describe())

# display the first few records of the DataFrame
# print(california_housing_dataframe.head())

# Accessing Data

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
# print(type(cities['City name']))
# print(cities['City name'])

# print(type(cities['City name'][1]))
# print(cities['City name'][1])

# print(type(cities[0:2]))
# print(cities[0:2])

# Manipulating Data

# print(population / 1000)
# print(population.apply(lambda val: val > 1000000))

# Modifying DataFrames
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
# print(cities)

# Indexes
# print(city_names.index)
# print(cities.index)

# print(cities.reindex([2, 0, 1]))

# Shuffling DataFrame
# print(cities.reindex(np.random.permutation(cities.index)))
