import dask.dataframe as dd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from data.mongo_data import df

ddf = dd.from_pandas(df, npartitions=4)

X_text = ddf['review_text'].compute()
y = ddf['sentiment_label'].compute()

vectorizer = HashingVectorizer(n_features=5000, alternate_sign=False)
X = vectorizer.transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Dask Logistic Regression Accuracy: {accuracy:.4f}")
