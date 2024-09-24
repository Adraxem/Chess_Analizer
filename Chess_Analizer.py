import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


def load_data(data_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    num_classes = len(class_names)

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.png'):
                img = cv2.imread(os.path.join(class_dir, filename))
                img = cv2.resize(img, (img_width, img_height))
                images.append(img)
                labels.append(class_index)
    return np.array(images), np.array(labels), num_classes, class_names


data_dir = 'chessdir'
img_width, img_height = 96, 96
images, labels, num_classes, class_names = load_data(data_dir)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


epochs = [10, 20, 30, 50, 80]
batch_sizes = [8, 32, 64, 128]

results = {'batch_size': {}, 'epochs': {}}

for batch_size in batch_sizes:
    print(f"Training with Batch Size: {batch_size}")
    model = create_model()
    history = model.fit(X_train, y_train, epochs=30, batch_size=batch_size, validation_data=(X_val, y_val))
    val_accuracy = history.history['val_accuracy'][-1]
    results['batch_size'][batch_size] = val_accuracy

for epoch in epochs:
    print(f"Training with Epochs: {epoch}")
    model = create_model()
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=64, validation_data=(X_val, y_val))
    val_accuracy = history.history['val_accuracy'][-1]
    results['epochs'][epoch] = val_accuracy

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(list(results['batch_size'].keys()), list(results['batch_size'].values()), marker='o')
plt.xlabel('Batch Size')
plt.ylabel('Validation Accuracy')
plt.title('Accuracy vs. Batch Size')

plt.subplot(1, 2, 2)
plt.plot(list(results['epochs'].keys()), list(results['epochs'].values()), marker='o')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Accuracy vs. Epochs')

plt.tight_layout()
plt.show()
