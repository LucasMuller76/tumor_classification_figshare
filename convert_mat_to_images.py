import os
import scipy.io
import numpy as np
import cv2
import h5py

# Definir pastas
input_folder = "data"
output_folder = "images"

# Criar pasta de saída se não existir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Criar subpastas para cada classe
classes = {1: "meningioma", 2: "glioma", 3: "pituitary_tumor"}
for class_name in classes.values():
    class_path = os.path.join(output_folder, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

# Função para verificar formato do .mat
def is_mat_v7_3(file_path):
    with open(file_path, "rb") as f:
        header = f.read(4)
    return header == b'MATL'

# Percorrer todos os arquivos .mat nas subpastas
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue
    
    for file in os.listdir(subfolder_path):
        if file.endswith(".mat"):
            file_path = os.path.join(subfolder_path, file)

            try:
                # Verifica se o arquivo é MATLAB v7.3 e usa h5py se necessário
                if is_mat_v7_3(file_path):
                    with h5py.File(file_path, 'r') as f:
                        image = np.array(f["cjdata/image"])
                        label = int(f["cjdata/label"][()][0])
                else:
                    mat_data = scipy.io.loadmat(file_path)
                    if "cjdata" not in mat_data:
                        continue
                    cjdata = mat_data["cjdata"]
                    image = np.array(cjdata["image"], dtype=np.float32)
                    label = int(cjdata["label"][0][0])

                # Normalizar imagem para 0-255
                image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)

                # Determinar a classe
                class_name = classes.get(label, "unknown")
                if class_name == "unknown":
                    continue

                # Salvar imagem
                output_path = os.path.join(output_folder, class_name)
                output_filename = os.path.join(output_path, f"{file.replace('.mat', '.jpg')}")
                cv2.imwrite(output_filename, image)

                print(f"Salvo: {output_filename}")

            except Exception as e:
                print(f"Erro ao processar {file_path}: {e}")

print("Conversão concluída! Imagens salvas na pasta:", output_folder)
