#%%
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
from PIL import Image

datas = glob.glob('D:\\code\\data\\ai_hub_data2/*/*/*.jpg')
class_name = ["plasting bag", "styrofoam", "cans"]
dic = {"plasting bag":0, "styrofoam":1, "cans":2}

X = []
Y = []
for imagename in datas:
    image = Image.open(imagename)
    image = image.resize((128, 128))
    image = np.array(image)
    X.append(image)
    label = imagename.split('\\')[4]
    label = dic[label]
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

train_images, test_images, train_labels, test_labels = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=44)

train_labels = train_labels[..., tf.newaxis]
# test_labels = test_labels[..., tf.newaxis]

print(train_images.shape)
print(train_labels.shape)
# %%
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
# %%
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(15, 9))
# for i in range(15):
#     img_idx = np.random.randint(0, 225)
#     plt.subplot(3,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[img_idx])
#     plt.xlabel(class_name[train_labels[img_idx][0]])

# %%
train_labels = tf.keras.utils.to_categorical(train_labels)
# %%
    
X = tf.keras.layers.Input(shape=[128, 128, 3])

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(3, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')
# %%
model.summary()
# %%

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)
history = model.fit(train_images, train_labels, validation_split=0.2,  epochs=50, batch_size = 64, callbacks=[early_stopping])
#%%

test_datas = glob.glob('D:\\code\\data\\test_data/*.jpg')

X2 = []
Y2 = []
for imagename in test_datas:
    image = Image.open(imagename)
    image = image.resize((128, 128))
    image = np.array(image)
    X2.append(image)
    label = np.array(label)
    Y2.append(label)

X2 = np.array(X2)
Y2 = np.array(Y2)
print(X2.shape)
print(Y2.shape)
#%%

# model.save('D:\\code\\model\\lenet5_1_model.h5')

#%%

loss, accuracy = model.evaluate(train_images, train_labels)
print("Test 데이터 정확도 : ", accuracy)
predictions = model.predict(X2)

# %%

img = test_images[4]
print(img.shape)
# %%
img = (np.expand_dims(img,0))

print(img.shape)

# %%
predictions_single = model.predict(img)

print(predictions_single)
# %%
plt.imshow(test_images[4])
# %%
np.argmax(predictions_single[0])

# %%
