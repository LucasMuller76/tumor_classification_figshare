import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_path = 'C:/Users/mulle/Documents/brain_tumor_figshare/images'

# Definindo o caminho das imagens (uma de cada)
img_paths = {
    'Meningioma': os.path.join(base_path, 'meningioma', os.listdir(os.path.join(base_path, 'meningioma'))[0]),
    'Glioma': os.path.join(base_path, 'glioma', os.listdir(os.path.join(base_path, 'glioma'))[0]),
    'Pituitary': os.path.join(base_path, 'pituitary_tumor', os.listdir(os.path.join(base_path, 'pituitary_tumor'))[0]),
}

# Plotar as imagens
plt.figure(figsize=(12, 4))
for i, (label, path) in enumerate(img_paths.items()):
    img = mpimg.imread(path)
    plt.subplot(1, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(label)
    plt.axis('off')

plt.tight_layout()
plt.savefig('amostras_figshare.png')
plt.show()
