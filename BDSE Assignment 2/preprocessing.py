import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
import re
from data.mongo_data import df 


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

df['clean_review'] = df['review_text'].apply(clean_text)

X_text = df['clean_review']
y = df['sentiment_label']

vectorizer = HashingVectorizer(n_features=5000, alternate_sign=False)
X = vectorizer.transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
