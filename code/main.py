import pandas as pd
import os
import numpy as np

! ls  
! pwd 

#
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




## 2. Viewing data
df2.dtypes

df.head()
df.tail(4)

df.index
df.columns

df.describe()




## 3. Data Manipulation
df = pd.read_csv("./data/superstore.csv", encoding = "ISO-8859-1")
df.sort_index(axis=1, ascending=False)
df.sort_values(by='B')


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