import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


datas = glob.glob('/home/longstone123/garbage_data/*/*/*.jpg')
datas = np.random.permutation(datas)
class_name = ["can", "glass", "paper", "pet", "plastic", "styrofoam", "vinyl"]
dic = {"can":0, "glass":1, "paper":2, "pet":3, "plastic":4, "styrofoam":5, "vinyl":6}

X = []
Y = []
for imagename in datas:
    image = Image.open(imagename)
    image = image.resize((128,128))
    image = np.array(image)
    X.append(image)
    label = imagename.split('/')[4]
    label = dic[label]
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

labels = Y[..., tf.newaxis]

labels = tf.keras.utils.to_categorical(labels)

images = X / 255.0
labels = labels / 255.0
    
X = tf.keras.layers.Input(shape=[128, 128, 3])

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(7, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)
history = model.fit(images, labels, epochs=50, batch_size = 64, callbacks=[early_stopping])

model.save('/home/longstone123/lenet5.h5')