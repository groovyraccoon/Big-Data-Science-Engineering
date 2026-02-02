import pandas as pd
import matplotlib.pyplot as plt
from data.mongo_data import df 

# Basic info
print(df.info())
print(df.describe())

print(df.isnull().sum())

# Target distribution
print(df['sentiment_label'].value_counts())


df['sentiment_label'].value_counts().plot(kind='bar')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()

df['review_length'] = df['review_text'].apply(len)
print(df['review_length'].describe())

df['review_length'].hist(bins=50)
plt.xlabel('Number of characters')
plt.ylabel('Frequency')
plt.title('Review Length Distribution')
plt.show()
