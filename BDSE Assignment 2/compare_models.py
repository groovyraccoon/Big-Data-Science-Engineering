import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

results = {
    'Dask Logistic Regression': 0.932,  
    'MLP': 0.949,
    'CNN': 0.955
}

print("Model Performance Comparison:")
for model, acc in results.items():
    print(f"{model}: Accuracy = {acc:.4f}")

# Bar plot for report
plt.figure(figsize=(6,4))
plt.bar(results.keys(), results.values(), color=['skyblue', 'lightgreen', 'salmon'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
for i, acc in enumerate(results.values()):
    plt.text(i, acc + 0.01, f"{acc:.3f}", ha='center')
plt.show()
