import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from tqdm import tqdm

datas = glob.glob('/home/longstone123/Garbage_classification/*/*.jpg')
datas = np.random.permutation(datas)
class_name = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
dic = {"cardboard":0, "glass":1, "metal":2, "paper":3, "plastic":4, "trash":5}

a1 = tqdm(np.array([plt.imread(datas[i]) for i in range(len(datas))]))
b1 = np.array([datas[i].split('/')[4] for i in range(len(datas))])

a1 = a1.reshape(2527, 384, 512, 3)
# 독립 = tf.image.resize(독립, (120, 120))
b1 = pd.get_dummies(b1)

X = tf.keras.layers.Input(shape=[384, 512, 3])

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(6, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)
history = model.fit(a1, b1, validation_split=0.2,  epochs=100, batch_size = 64, callbacks=[early_stopping])