import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

(mnist_x, mnist_y), _ = tf.keras.datasets.mnist.load_data()
print(mnist_x.shape, mnist_y.shape)

(cifar_x, cifar_y), _ = tf.keras.datasets.cifar10.load_data()
print(cifar_x.shape, cifar_y.shape)

import matplotlib.pyplot as plt
plt.imshow(mnist_x[0], cmap='gray')

print(mnist_y[0:10])

plt.imshow(mnist_x[4], cmap='gray')

print(cifar_y[0:10])

plt.imshow(cifar_x[0], cmap='gray')

plt.imshow(cifar_x[1])


import numpy as np

d1 = np.array([1, 2, 3, 4, 5])
print(d1.shape)

d2 = np.array([d1, d1, d1, d1])
print(d2.shape)

d3 = np.array([d2, d2, d2])
print(d3.shape)

d4 = np.array([d3, d3])
print(d4.shape)


print(mnist_y.shape)
print(cifar_y.shape)

x1 = np.array([1, 2, 3, 4, 5])
print(x1.shape)
print(mnist_y[0:5])
print(mnist_y[0:5].shape)

x2 = np.array([[1, 2, 3, 4, 5], [1,2,3,4,5], [1,2,3,4,5]])
print(x2.shape)

x3 = np.array([[[1], [2], [3], [4], [5]]])
print(x3.shape)
print(cifar_y[0:5])
print(cifar_y[0:5].shape)