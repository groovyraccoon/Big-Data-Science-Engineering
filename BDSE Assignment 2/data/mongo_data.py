import pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

db = client["bdse_assignment"]
collection = db["reviews"]


cursor = collection.find({}, {"_id": 0, "review_text": 1, "sentiment_label": 1, "reviewer_score": 1})
df = pd.DataFrame(list(cursor))

# Quick check
print(df.shape)
print(df.head())

