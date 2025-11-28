# Importing the libraries
from sklearn.model_selection import train_test_split
import os
import cv2
from imblearn.over_sampling import SMOTE
import numpy as np
import albumentations as A


# Defining a function to apply SMOTE
def smote(X, y):
    # Convert X to numpy array so it can be reshaped
    X = np.array(X)

    # Save the original shape
    original_shape = X.shape

    # Reshape X for SMOTE
    X = X.reshape(X.shape[0], -1)

    # Create an object of the SMOTE class
    smote = SMOTE()

    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Reshape X to what it was before
    X_resampled = X_resampled.reshape(-1, original_shape[1], original_shape[2], original_shape[3])

    return X_resampled, y_resampled


# Defining a function to randomly augment the training set
def random_augmentation(X):
    # Create an augmentation pipeline that simulates different conditions
    transform = A.Compose([
        # Simulate different brightness levels
        A.RandomBrightnessContrast(p=0.5),  # p=0.5 means apply this 50% of the time

        # Simulate different color temperatures (morning vs evening light)
        A.ColorJitter(p=0.5),

        # Add some noise to simulate different camera sensors
        A.GaussNoise(p=0.5),

        # Simulate different lighting conditions
        A.RandomGamma(p=0.5)
    ])

    # Add randomly processed images to training set
    for i in range(len(X)):
        X[i] = transform(image=X[i])['image']

    return X


# Defining a function to import the data
def get_data():
    # Initialise an array for the dataset
    dataset = []

    # Get the positive samples
    for img in os.listdir('face_data/positive_samples'):
        image = cv2.imread(f"face_data/positive_samples/{img}")
        image = cv2.resize(image, (224, 224))
        dataset.append([image, 0])

    # Get the negative samples
    for img in os.listdir('face_data/negative_samples'):
        image = cv2.imread(f"face_data/negative_samples/{img}")
        image = cv2.resize(image, (224, 224))
        dataset.append([image, 1])

    # Initialise arrays for the independent and dependant variables
    X = []
    y = []

    # Populate the independent and dependant variable arrays
    for img, label in dataset:
        X.append(img)
        y.append(label)

    # Split dataset into training set and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Apply random augmentations
    X_train = random_augmentation(X_train)

    # Apply SMOTE
    X_train, y_train = smote(X_train, y_train)

    return X_train, X_val, y_train, y_val
