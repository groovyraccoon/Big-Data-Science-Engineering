import mysql.connector
import pandas as pd

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",
    database="bdse_assignment"
)
cursor = conn.cursor()

query = """
SELECT review_text, sentiment_label, reviewer_score
FROM hotel_reviews
WHERE reviewer_score >= %s OR reviewer_score <= %s
"""
cursor.execute(query, (8.0, 4.0))


# Fetch results into a data frame
df_sql = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# Make sure sentiment labels are integers
df_sql["sentiment_label"] = df_sql["sentiment_label"].astype(int)

# Print quick overview
print(f"Fetched {df_sql.shape[0]} rows where reviewer_score >= {8.0}")
print(df_sql.head())

# Close connections
cursor.close()
conn.close()
