import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tqdm import tqdm


datas = glob.glob('D:\\code\\data\\ai_hub/*/*/*.jpg')
datas = np.random.permutation(datas)
class_name = ["can", "glass", "paper", "pet", "plastic", "styrofoam", "vinyl"]
dic = {"can":0, "glass":1, "paper":2, "pet":3, "plastic":4, "styrofoam":5, "vinyl":6}

images = []
labels = []
for imagename in tqdm(datas):
    image = Image.open(imagename)
    image = image.resize((64,64))
    image = np.array(image)
    images.append(image)
    label = imagename.split('\\')[4]
    label = dic[label]
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

labels = labels[..., tf.newaxis]

labels = tf.keras.utils.to_categorical(labels)

images = images / 255.0
labels = labels / 255.0
    
X = tf.keras.layers.Input(shape=[64, 64, 3])

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(7, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
tf.keras.optimizers.Adam(
    learning_rate = 0.0001
)

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics='accuracy')

# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)
# history = model.fit(images, labels, epochs=50, batch_size = 10, callbacks=[early_stopping])
history = model.fit(images, labels, epochs=50, batch_size = 10)

model.save('/home/longstone123/lenet5_test.h5')