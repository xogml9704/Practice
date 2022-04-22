# 1. 과거의 데이터를 준비합니다.
#  레모네이드 = pd.read_csv('lemonade.csv')
#  독립 = 레모네이드[['온도']]
#  종속 = 레모네이드[['판매량']]

#  print(독립.shape, 종속.shape)

# 2. 모델의 구조를 만듭니다.
#  X = tf.keras.layers.Input(shape=[1])
#  Y = tf.keras.layers.Dense(1)(X)
#  model = tf.keras.models.Model(X, Y)
#  model.compile(loss='mse')

# 3. 데이터로 모델을 학습(FIT)합니다.
#  model.fit(독립, 종속, epochs=1000)

# 4. 모델을 이용합니다.
# print("Predictions : ", model.predict(([15])))

import pandas as pd
import tensorflow as tf

레모네이드 = pd.read_csv('D:\\code\\vs\\Tensorflow_practice2\\lemonade.csv')
독립 = 레모네이드[['온도']]
종속 = 레모네이드[['판매량']]

print(독립.shape, 종속.shape)

x = tf.keras.layers.Input(shape=[1])
y = tf.keras.layers.Dense(1)(x)
model = tf.keras.models.Model(x, y)
model.compile(loss='mse')

model.fit(독립, 종속, epochs=1000)

print("Predictions : ", model.predict(([15])))