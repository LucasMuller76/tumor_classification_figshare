import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix

# Definir caminho das imagens
base_dir = "images"

# Criar datasets de treinamento e teste
batch_size = 32
img_size = (180, 180)

train_dataset = image_dataset_from_directory(base_dir,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             image_size=img_size,
                                             validation_split=0.2,
                                             subset="training",
                                             seed=123)

val_dataset = image_dataset_from_directory(base_dir,
                                           shuffle=True,
                                           batch_size=batch_size,
                                           image_size=img_size,
                                           validation_split=0.2,
                                           subset="validation",
                                           seed=123)

# Verificar classes
class_names = train_dataset.class_names
print(f"Classes detectadas: {class_names}")

# Criando o modelo CNN
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
epochs = 10
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(val_dataset)
print(f"Test accuracy: {test_acc:.4f}")


model.save("modelo_treinado.h5")  # Salvar o modelo após o treinamento
