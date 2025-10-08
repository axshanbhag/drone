import pandas as pd
df = pd.read_csv('titanic.csv')
# the first 5 rows of the dataset
header = df.head()
# information about the dataset (columns, names, non-null counts (filled), data types)
df.info()
# summary statistics for each column of the dataset
df.describe(include = 'all')
age = 'Age'
# remove missing values in the Age column
removeAge = df.dropna(subset=[age])
# drop rows with any missing values
df.dropna()
# fill missing values with 0
df.fillna(value=0)
# filter rows where Age is greater than 30
older30 = df[df['Age'] > 30]
# select only the Name and Age columns
df[['Name', 'Age']]
# counts of unique values in the Sex column (male/female)
df['Sex'].value_counts()
# groups each sex and computes the mean of the survived column for each group
rate = df.groupby('Sex')['Survived'].mean()
