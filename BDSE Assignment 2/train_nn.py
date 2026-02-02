import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from data.mongo_data import df



X_text = df['review_text'].astype(str)
y = df['sentiment_label']

num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X_text)
X_seq = tokenizer.texts_to_sequences(X_text)

max_len = 200
X_pad = pad_sequences(X_seq, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42, stratify=y
)

mlp_model = Sequential([
    Embedding(input_dim=num_words, output_dim=32, input_length=max_len),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

cnn_model = Sequential([
    Embedding(input_dim=num_words, output_dim=32, input_length=max_len),
    Conv1D(64, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

mlp_acc = mlp_model.evaluate(X_test, y_test, verbose=0)[1]
cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)[1]

print(f"MLP Accuracy: {mlp_acc:.4f}")
print(f"CNN Accuracy: {cnn_acc:.4f}")
