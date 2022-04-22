#%%
import tensorflow as tf

#%%
# 모델 만들기 (다중 분류)
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
model.fit(x_train, y_train)
        #  f       label
#%%
mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test, y_test) = mnist.load_data()
#%%
# 손 글씨 출력
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
print("lable",y_train[0])

# %%
plt.imshow(x_train[0], cmap='gray')

# %%

import numpy as np
np.set_printoptions(linewidth=120)

print(x_train[0])
plt.imshow(x_train[0])
# %%

# epochs : 전체 데이터 셋을 학습 완료한 상태
# batch_size : 한번의 배치마다 전체 데이터에서 일부를 불러오는 사이즈
# iteration : 한 epoch당 필요한 배치 사이즈 개수, 파라미터 업데이트 개수 (전체 데이터 셋 크기 / 베치 사이즈)

# 훈련(학습)
model.fit(x_train, y_train)

# epochs
model.fit(x_train, y_train, epochs=3)

# batch_size
model.fit(x_train, y_train, epochs=3, batch_size=64)

# verbose = 0, silent모드(출력 x)
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)

# verbose = 1, 진행 표시
model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=1)

# verbose = 2, 진행 표시 없음
history = model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=2)

# 그림의로 그려보기 위해 history를 추가
plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
