import pandas as pd

df = pd.read_csv("data/Hotel_Reviews.csv")

print(df.shape)
print(df.columns)
print(df.head(3))

empty_positive = df['Positive_Review'].isnull().sum() + (df['Positive_Review'].str.strip() == '').sum()
empty_negative = df['Negative_Review'].isnull().sum() + (df['Negative_Review'].str.strip() == '').sum()

print(f"Empty positive reviews: {empty_positive}")
print(f"Empty negative reviews: {empty_negative}")