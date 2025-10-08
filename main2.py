import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('titanic.csv')
# range of age will be split into 20 equal intervals
# counts of passangers in each interval
# used to make the bottom start at 0
plt.hist(df['Age'], bins=20)
# scatter plot of Age vs Fare
plt.scatter(df['Age'], df['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare')
plt.show()
le = LabelEncoder()
# encode category with order, categories have unique integer to convert words into numbers
# find the unique categories in Sex and encode and replace
# column is added in replacement of Sex
df['Sex_encoded'] = le.fit_transform(df['Sex'])
# one-hot encode the Embarked column that is non-ordered
# basically creates a new column for each unique value in Embarked and fills with 0s and 1s
df_encoded = pd.get_dummies(df, columns=['Embarked'])
# Show what columns were created
print(df_encoded.columns)
# Show the first few rows of those new columns
print(df_encoded[['Embarked_C', 'Embarked_Q', 'Embarked_S']].head())
