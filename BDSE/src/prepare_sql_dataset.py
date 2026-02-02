import pandas as pd


df_kaggle = pd.read_csv("data/kaggle_cleaned.csv")

# Remove empty reviews
df_kaggle = df_kaggle[df_kaggle['review_text'].str.strip() != ""]

sample_size = 50000
if len(df_kaggle) > sample_size:
    df_kaggle = df_kaggle.sample(n=sample_size, random_state=42)

# Making sure sentiment labels are integers
df_kaggle['sentiment_label'] = df_kaggle['sentiment_label'].astype(int)

print(f"Kaggle sample: {df_kaggle.shape}")


# Load scraped and handwritten reviews
df_scraped = pd.read_csv("data/scraped_reviews.csv")
df_handwritten = pd.read_csv("data/my_reviews.csv")

# Making sure sentiment labels are integers
for df in [df_scraped, df_handwritten]:
    df['sentiment_label'] = df['sentiment_label'].astype(int)

# Combine all datasets into one
final_df = pd.concat([df_kaggle, df_scraped, df_handwritten], ignore_index=True)
print(f"Total rows: {final_df.shape}")

# Save CSV
final_df.to_csv("data/sql_import_dataset.csv", index=False)
print("Saved dataset")
