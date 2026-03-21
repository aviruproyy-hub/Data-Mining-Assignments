import tensorflow as tf
from sklearn import svm
import joblib # Used to save/load traditional ML models

# 1. Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess: Flatten images (28x28 -> 784) and Normalize
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# 3. Build and Train SVM
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(x_train[:10000], y_train[:10000]) 

# 4. Save the model
joblib.dump(clf, "mnist_svm.pkl")