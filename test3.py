#%%
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import resnet

datas = glob.glob('D:\\code\\data\\archive\\Garbage classification/*/*.jpg')
datas = np.random.permutation(datas)

독립 = np.array([plt.imread(datas[i]) for i in range(len(datas))])
종속 = np.array([datas[i].split('\\')[5] for i in range(len(datas))])

print(독립.shape, 종속.shape)

print(종속[0:10])
# %%
독립 = 독립.reshape(2527, 384, 512, 3)
종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape)
# %%

model = resnet.ResNet50(include_top=True, weights=None, input_shape=(384, 512, 3), pooling=max, classes=3)
model.compile(loss='categorical_crossentropy', metrics='accuracy')
# %%
model.summary()
# %%
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)
history = model.fit(독립, 종속, validation_split=0.2,  epochs=50, batch_size = 64, callbacks=[early_stopping])
#%%

test_datas = glob.glob('D:\\code\\data\\test_data/*.jpg')
test_datas = np.random.permutation(test_datas)

test_독립 = np.array([plt.imread(test_datas[i]) for i in range(len(test_datas))])

print(test_독립.shape)

#%%

loss, accuracy = model.evaluate(독립, 종속)
print("Test 데이터 정확도 : ", accuracy)
predictions = model.predict(test_독립)
#%%

# 첫 번째 데이터 예측 결과 값
predictions[0]
#%%
# 표현방식 변경(소수점 두자리)
np.set_printoptions(formatter={'float_kind':lambda x: "{0:0.2f}".format(x)})

#%%
# 첫 번쨰 데이터 예측 결과 값
predictions[0]
#%%
# 첫 번째 데이터 에측 결과 값 그래프로 표시
plt.figure(figsize=(10,2))
plt.plot(predictions[0],'o')
#%%
# 첫 번째 데이터 값
plt.imshow(test_독립[0])
# %%
plt.figure(figsize=(10,2))
plt.plot(predictions[4],'o')
# %%
# 첫 번째 데이터 값
plt.imshow(test_독립[4])
# %%

# %%