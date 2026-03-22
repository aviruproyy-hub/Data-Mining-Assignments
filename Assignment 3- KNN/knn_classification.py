import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

# 1. Load Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sl', 'sw', 'pl', 'pw', 'class'])

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 2. Split & Scale (Essential for distance-based KNN)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Find Best K using Cross-Validation on Training Set
best_k = 1
best_score = 0

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    # Using 5-fold cross-validation to find the most stable K
    score = cross_val_score(knn, X_train, y_train, cv=5).mean()
    if score > best_score:
        best_score = score
        best_k = k

# 4. Final Model Training
knn_final = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn_final.fit(X_train, y_train)

# 5. Final Evaluation
predictions = knn_final.predict(X_test)

print(f"Optimal K Found: {best_k}")
print(f"Final Test Accuracy: {knn_final.score(X_test, y_test):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nDetailed Report:")
print(classification_report(y_test, predictions))



# 6. Plotting the Validation Curve (Finding K)
k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Using 5-fold cross-validation on the training data
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

plt.figure(figsize=(10, 4))
plt.plot(k_range, cv_scores, marker='o', color='#2ca02c', linestyle='--')
plt.title('Validation Curve: Cross-Validation Accuracy vs. K')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Mean CV Accuracy')
plt.xticks(k_range)
plt.grid(axis='y', alpha=0.3)
plt.show()

# 7. Plotting the Confusion Matrix
cm = confusion_matrix(y_test, predictions)
labels = ['Setosa', 'Versicolor', 'Virginica']

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix: Predicted vs. Actual Species')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()