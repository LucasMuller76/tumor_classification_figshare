import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing import image_dataset_from_directory
from keras.models import load_model
import os

# Carregar modelo treinado
model = load_model("modelo_treinado.h5")  
# Carregar dataset de teste
test_dir = "dataset/test"  
batch_size = 32
img_size = (180, 180)
test_dataset = image_dataset_from_directory(test_dir, image_size=img_size, batch_size=batch_size)

# Obter classes
class_names = test_dataset.class_names

# Fazer previsões
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

# Matriz de confusão
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Mostrar algumas imagens com as previsões
plt.figure(figsize=(10, 10))
for images, labels in test_dataset.take(1):  # Pega um batch de imagens
    preds = model.predict(images)
    pred_labels = np.argmax(preds, axis=1)
    
    for i in range(9):  # Mostrar 9 imagens
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_label = class_names[labels[i]]
        pred_label = class_names[pred_labels[i]]
        plt.title(f"Real: {true_label}\nPred: {pred_label}", fontsize=10)
        plt.axis("off")

plt.tight_layout()
plt.show()
