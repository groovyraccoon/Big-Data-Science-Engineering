import pandas as pd

df = pd.read_csv("data/Hotel_Reviews.csv")

print(f"Original shape: {df.shape}")

# Keep relevant columns
df = df[['Negative_Review', 'Positive_Review', 'Reviewer_Score']]

# Combine positive and negative review text into a column
def combine_reviews(row):
    if str(row['Negative_Review']).strip() and str(row['Negative_Review']).strip() != "No negative":
        return row['Negative_Review'], 0
    else:
        return row['Positive_Review'], 1 

# Apply function to create review_text and sentiment_label
df[['review_text', 'sentiment_label']] = df.apply(lambda row: pd.Series(combine_reviews(row)), axis=1)

# Keep only the final columns
df = df[['review_text', 'Reviewer_Score', 'sentiment_label']]

# Remove empty reviews
df = df[df['review_text'].str.strip() != ""]

print(f"Cleaned shape: {df.shape}")
print(df.head())

# Save cleaned CSV
output_path = "data/kaggle_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"Saved cleaned Kaggle dataset to {output_path}")
