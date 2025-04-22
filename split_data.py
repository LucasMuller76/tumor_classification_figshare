import os
import shutil
import random

# Caminho das imagens organizadas por classe
input_folder = "images"

# Criar pastas para treino e teste
train_folder = "dataset/train"
test_folder = "dataset/test"

# Definir proporção de divisão (80% treino, 20% teste)
split_ratio = 0.8

# Criar as pastas se não existirem
for folder in [train_folder, test_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Separar os dados
for class_name in os.listdir(input_folder):
    class_path = os.path.join(input_folder, class_name)
    if not os.path.isdir(class_path):
        continue
    
    images = os.listdir(class_path)
    random.shuffle(images)

    split_point = int(len(images) * split_ratio)
    train_images = images[:split_point]
    test_images = images[split_point:]

    for dataset_type, image_list in zip(["train", "test"], [train_images, test_images]):
        output_class_folder = os.path.join("dataset", dataset_type, class_name)
        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)
        
        for image in image_list:
            src_path = os.path.join(class_path, image)
            dest_path = os.path.join(output_class_folder, image)
            shutil.copy2(src_path, dest_path)

print("Divisão concluída! Imagens separadas em 'dataset/train' e 'dataset/test'.")
