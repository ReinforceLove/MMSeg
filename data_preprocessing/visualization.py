import numpy as np
import matplotlib.pyplot as plt


def visualize_image(image_path):
    image = np.load(image_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray', vmin=-1, vmax=1)
    plt.title('Image')
    plt.axis('off')
    plt.show()



def visualize_label_with_black_background(label_path):
    label = np.load(label_path)
    background = np.zeros(label.shape, dtype=np.uint8)
    label_with_alpha = np.zeros((*label.shape, 4), dtype=np.uint8)
    cmap = plt.get_cmap('jet')
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            color = cmap(label[i, j] / label.max())
            label_with_alpha[i, j, :3] = (np.array(color[:3]) * 255).astype(np.uint8)
            label_with_alpha[i, j, 3] = int(255 * (label[i, j] > 0))
    plt.figure(figsize=(8, 8))
    plt.imshow(background, cmap='gray', vmin=0, vmax=1)
    plt.imshow(label_with_alpha)
    plt.title('Label')
    plt.axis('off')
    plt.show()

