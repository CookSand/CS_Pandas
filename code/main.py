import pandas as pd
import os
import numpy as np

## Convert multiple files into one file and create the outputs

import os
import glob
import pandas as pd
import numpy as np

os.chdir('./data/multiple_files')
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

print(all_filenames)

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f, encoding = "ISO-8859-1") for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')



""" 1. Basics 
1. Create the data series
2. Create the data frame

"""

# 1. Create DataSeries
## 1.1. From List
s = pd.Series([1, 3, 5, np.nan, 6, 8])


# 2. Create DataFrame
## 2.1. From Numpy
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

## 2.2. From dict
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

## 2.3. From external data
df = pd.read_csv('./data/superstore.csv', encoding = "ISO-8859-1")



## 2. Viewing data
df2.dtypes

df.head()
df.tail(4)

df.index
df.columns

df.describe()




## 3. Data Manipulation
### This chaptal is about how to slice data, there are multiple ways to do that and the data label in rwo is index and in column is columns

# Check the index and columns
df.index
df.columns

df = pd.read_csv("./data/superstore.csv", encoding = "ISO-8859-1")
df.sort_index(axis=1, ascending=False)
df.sort_values(by='B')

df = pd.DataFrame({
   ....:     'one': pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
   ....:     'two': pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),
   ....:     'three': pd.Series(np.random.randn(3), index=['b', 'c', 'd'])})

row = df.iloc[1]

column = df['two']
df.sub(row, axis='columns')

dfmi = df.copy()

In [27]: dfmi.index = pd.MultiIndex.from_tuples([(1, 'a'), (1, 'b'),
   ....:                                         (1, 'c'), (2, 'a')],
   ....:                                        names=['first', 'second'])
   ....: 

In [28]: dfmi.sub(column, axis=0, level='second')



Missing data:
df.add(df2, fill_value=0)

# Selection
df.Country
df['Country']
df[0:3]


## Selection by label
df2.loc[dates[0]]
df.loc[:, ['Country', 'Sales']]

# The usage for the index, the question is whehter it can be duplicated or it is just unique?
df.loc[dates[0], 'A']
df.at[dates[0], 'A']


## Selection by position

df.iloc[3]
df.iloc[3:5, 0:2]
df.iloc[[1, 2, 4], [0, 2]]


df.iloc[1, 1]
df.iat[1, 1]


## Boolean indexing

# Using a single column’s values to select data.
df[df.A > 0]

#Selecting values from a DataFrame where a boolean condition is met.
df[df > 0]

# Boolean reductions
In [48]: (df > 0).all()
Out[48]: 
one      False
two       True
three    False
dtype: bool

In [49]: (df > 0).any()
Out[49]: 
one      True
two      True
three    True
dtype: bool

(df > 0).any().any()

df1.equals(df2)


# Combining overlapping data sets - A way to deal with the missing values

df1.combine_first(df2)

The idxmin() and idxmax() functions on Series and DataFrame compute the index labels with the minimum and maximum corresponding values:

Value counts (histogramming) / mode
The value_counts() Series method and top-level function computes a histogram of a 1D array of values. It can also be used as a function on regular arrays:



In [126]: arr = np.random.randn(20)

In [127]: factor = pd.cut(arr, 4)

In [128]: factor
Out[128]: 
[(-0.251, 0.464], (-0.968, -0.251], (0.464, 1.179], (-0.251, 0.464], (-0.968, -0.251], ..., (-0.251, 0.464], (-0.968, -0.251], (-0.968, -0.251], (-0.968, -0.251], (-0.968, -0.251]]
Length: 20
Categories (4, interval[float64]): [(-0.968, -0.251] < (-0.251, 0.464] < (0.464, 1.179] <
                                    (1.179, 1.893]]

In [129]: factor = pd.cut(arr, [-5, -1, 0, 1, 5])

pipe()

df.apply(np.mean)
Out[141]: 
one      0.811094
two      1.360588
three    0.187958
dtype: float64

In [142]: df.apply(np.mean, axis=1)
Out[142]: 





## Setting, add one more column into the dataframe

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
df['F'] = s1



df1.dropna(how='any')
df1.fillna(value=5)
pd.isna(df1)


## Apply, apply a function to the dataframe

df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())



## Append is appending rows to the dataframe
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
s = df.iloc[3]
df.append(s, ignore_index=True)



## Grouping 
"""
By “group by” we are referring to a process involving one or more of the following steps:

Splitting the data into groups based on some criteria
Applying a function to each group independently
Combining the results into a data structure
"""



## Stack
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
   ....:                      'foo', 'foo', 'qux', 'qux'],
   ....:                     ['one', 'two', 'one', 'two',
   ....:                      'one', 'two', 'one', 'two']]))


## Pivot Table
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])


## Categoricals
df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                   "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})

df["grade"] = df["raw_grade"].astype("category")
df["grade"].cat.categories = ["very good", "good", "very bad"]


## Reorder the categories and simultaneously add the missing categories (methods under Series .cat return a new Series by default).

df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df.sort_values(by="grade")



