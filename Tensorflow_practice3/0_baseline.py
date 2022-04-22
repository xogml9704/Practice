#%%
import tensorflow as tf

#mnist 데이터 셋
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test) = mnist.load_data()

# 모델
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

# 훈련(학습)
history = model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=64)
#%%
# 평가(정확도)
loss, accuracy = model.evaluate(x_test, y_test)
print("Test 데이터 정확도 : ", accuracy)
#%%
# 예측
predictions = model.predict(x_test)
#%%
# 첫 번째 데이터 예측 결과 값
predictions[0]
#%%
# 표현방식 변경(소수점 두자리)
import numpy as np
np.set_printoptions(formatter={'float_kind':lambda x: "{0:0.2f}".format(x)})
#%%
# 첫 번쨰 데이터 예측 결과 값
predictions[0]
#%%
# 첫 번째 데이터 에측 결과 값 그래프로 표시
import matplotlib.pyplot as plt
plt.figure(figsize=(12,2))
plt.plot(predictions[0],'o')
#%%
# 첫 번째 데이터 손 글씨
plt.imshow(x_test[0])
#%%