# Importing the libraries
from keras._tf_keras.keras.models import load_model
import numpy as np
import get_training_test_set as gts
from sklearn.metrics import classification_report

model = load_model("face_recognition_model_V5.keras")

X_train, X_test, y_train, y_test = gts.get_data()
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred.round()))
