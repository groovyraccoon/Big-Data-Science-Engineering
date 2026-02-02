from preprocessing import X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Convert sparse matrices to dense only for KNN
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Multinomial NB": MultinomialNB(),
    "Linear SVM": LinearSVC(max_iter=5000),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Store predictions and accuracies
predictions = {}
accuracies = {}

# Train models and evaluate
for name, model in models.items():
    print(f"{name}")
    # Use dense arrays for KNN, sparse for others
    if name == "KNN":
        model.fit(X_train_dense, y_train)
        y_pred = model.predict(X_test_dense)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    predictions[name] = y_pred
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# Plot confusion matrices for Logistic Regression and KNN
for model_name in ["Logistic Regression", "KNN"]:
    cm = confusion_matrix(y_test, predictions[model_name])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    cmap = plt.cm.Blues if model_name == "Logistic Regression" else plt.cm.Oranges
    disp.plot(cmap=cmap)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.show()

# Plot overall accuracy for all models
plt.figure(figsize=(8, 5))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red', 'purple', 'orange'])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Comparison of Model Accuracies")
plt.show()
