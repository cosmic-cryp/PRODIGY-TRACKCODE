
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            img = cv2.resize(img, (150, 150))  
            img = img.astype(np.float32) / 255.0  
            images.append(img)
            label = filename.split('.')[0]  
            labels.append(label)
    return np.array(images), np.array(labels)
train_folder ="C:\\Users\\Patoju Karthikeya\\OneDrive\\Desktop\\HandGesture\\images\\peace"
validation_folder ="C:\\Users\\Patoju Karthikeya\\OneDrive\\Desktop\\HandGesture\\images\\peace"

train_images, train_labels = load_images_from_folder(train_folder)
validation_images, validation_labels = load_images_from_folder(validation_folder)
validation_images, test_images, validation_labels, test_labels = train_test_split(
    validation_images, validation_labels, test_size=0.5, random_state=42)

print("Train images shape:", train_images.shape)
print("Validation images shape:", validation_images.shape)
print("Test images shape:", test_images.shape)
Train images shape: (526, 150, 150, 3)
Validation images shape: (263, 150, 150, 3)
Test images shape: (263, 150, 150, 3)
import cv2
import os
import numpy as np

def load_images_from_folder(folder_path):
    images = []
    image_names = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, (150, 150))  # Resize images to (150, 150)
            img = img.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)
            image_names.append(filename)
    return np.array(images), image_names

# Path to your validation folder
validation_folder = "C:\\Users\\Patoju Karthikeya\\OneDrive\\Desktop\\HandGesture\\images\\peace"

# Load validation images and their names
validation_images, image_names = load_images_from_folder(validation_folder)

print("Validation images shape:", validation_images.shape)
print("Number of validation images:", len(image_names))
Validation images shape: (526, 150, 150, 3)
Number of validation images: 526
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification, change for multi-class
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Change for multi-class
              metrics=['accuracy'])

model.summary()
c:\Users\Patoju Karthikeya\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\layers\convolutional\base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 148, 148, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 74, 74, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 72, 72, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 36, 36, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 34, 34, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 17, 17, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 15, 15, 128)    │       147,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 7, 7, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 6272)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 512)            │     3,211,776 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │           513 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 3,453,121 (13.17 MB)
 Trainable params: 3,453,121 (13.17 MB)
 Non-trainable params: 0 (0.00 B)
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from PIL import Image


dataset_path = "C:\\Users\\Patoju Karthikeya\\OneDrive\\Desktop\\HandGesture\\images\\peace"

class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()
        

    def forward(self, x):
        def train_model():
            def evaluate_model():
    
                if __name__ == "__main__":
                    train_model()
                    evaluate_model()
