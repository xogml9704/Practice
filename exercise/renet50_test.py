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

base_model = tf.keras.applications.resnet.ResNet50(include_top=True, weights=None, input_shape=(384, 512, 3), pooling=max, classes=3)
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics='accuracy')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)
history = model.fit(a1, b1, validation_split=0.2,  epochs=50, batch_size = 1, callbacks=[early_stopping])

test_datas = glob.glob('D:\\code\\data\\test_data/*.jpg')
test_datas = np.random.permutation(test_datas)

test_a1 = np.array([plt.imread(test_datas[i]) for i in range(len(test_datas))])

print(test_a1.shape)

loss, accuracy = model.evaluate(a1, b1)
print("Test 데이터 정확도 : ", accuracy)
predictions = model.predict(test_a1)