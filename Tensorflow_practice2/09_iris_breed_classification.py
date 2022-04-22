import tensorflow as tf
import pandas as pd

# 1. 과거의 데이터를 준비합니다.

파일경로 = 'D:\\code\\vs\\Tensorflow_practice2\\iris.csv'
아이리스 = pd.read_csv(파일경로)
print(아이리스.head())

# 원핫인코딩
인코딩 = pd.get_dummies(아이리스)
print(인코딩.head())

# 독립변수, 종속변수
print(인코딩.columns)
독립 = 인코딩[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속 = 인코딩[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(독립.shape, 종속.shape)

# 2. 모델의 구조를 만듭니다.
X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 3. 데이터로 모델을 학습(FIT) 합니다.
model.fit(독립, 종속, epochs=400)

# 4. 모델을 이용합니다.
print(model.predict(독립[0:5]))

print(종속[0:5])

print(model.predict(독립[-5:]))

print(종속[-5:])

# 학습한 가중치
print(model.get_weights())