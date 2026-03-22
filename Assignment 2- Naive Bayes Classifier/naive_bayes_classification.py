import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset from UCI
def load_iris_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    df = pd.read_csv(url, names=column_names)
    return df

# Load dataset
df = load_iris_dataset()
print(f"Dataset loaded: {len(df)} samples")
print(f"\nShape: {df.shape}\n")

# Prepare data
X = df.iloc[:, :-1].values  # Features (4 columns)
y = df.iloc[:, -1].values   # Target (class column)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("DATA SPLITTING:")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}\n")

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gaussian Naive Bayes (best for continuous data)
print("TRAINING NAIVE BAYES CLASSIFIERS:")

# Gaussian NB - for continuous features
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
gnb_pred = gnb.predict(X_test_scaled)
gnb_accuracy = accuracy_score(y_test, gnb_pred)

print(f"Gaussian Naive Bayes Accuracy: {gnb_accuracy:.4f} ({gnb_accuracy*100:.2f}%)\n")

# Classification Report
print("CLASSIFICATION REPORT (GaussianNB):")
print(classification_report(y_test, gnb_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, gnb_pred)
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
            yticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title('Confusion Matrix - Naive Bayes Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Summary
print(f"""
Results:
- Training Accuracy: {gnb.score(X_train_scaled, y_train)*100:.2f}%
- Test Accuracy: {gnb_accuracy*100:.2f}%
""")

# Save predictions to CSV
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': gnb_pred,
    'Correct': y_test == gnb_pred
})
results_df.to_csv('naive_bayes_results.csv', index=False)