import cv2
import numpy as np
import matplotlib.pyplot as plt


# Реализация функции k-means clustering без использования cv2.kmeans
def custom_kmeans(data, k, max_iters=100):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iters):
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


# Функция для применения кластеризации с использованием custom_kmeans
def apply_custom_kmeans(image, k):
    # Convert BGR to RGB (if needed for display)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels (flatten)
    pixels = image.reshape((-1, 3))

    # Apply custom K-means clustering
    centroids, labels = custom_kmeans(pixels, k)

    # Convert centroids to uint8 (colors)
    centers = np.uint8(centroids)

    # Map the labels to the centers to get segmented image
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image shape
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image


# Загрузка изображения
image = cv2.imread('input_image.jpg')

# Отображение оригинального изображения
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Применение кластеризации с разными значениями k
ks = [3, 5, 10]

for i, k in enumerate(ks):
    segmented_image = apply_custom_kmeans(image, k)

    plt.subplot(2, 2, i + 2)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title('K = {}'.format(k))
    plt.axis('off')

plt.tight_layout()
plt.show()

