import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from tqdm import tqdm

datas = glob.glob('D:\\code\\data\\Garbage classification/*/*.jpg')
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
    label = imagename.split('/')[4]
    label = dic[label]
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

labels = labels[..., tf.newaxis]

labels = tf.keras.utils.to_categorical(labels)

images = images / 255.0
labels = labels / 255.0

base_model = tf.keras.applications.resnet.ResNet50(include_top=True, weights=None, input_shape=(64, 64, 3), pooling=max, classes=3)
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(7, activation='softmax'))

tf.keras.optimizers.Adam(
    learning_rate=0.000001
)
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics='accuracy')

# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)
# history = model.fit(images, labels, epochs=50, batch_size = 10, callbacks=[early_stopping])
history = model.fit(images, labels,validation_split=0.2, epochs=50, batch_size = 1)

model.save('/home/longstone123/lenet5_test.h5')