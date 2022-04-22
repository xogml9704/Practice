# 라이브러리를 사용합니다.
import tensorflow as tf
import pandas as pd

# 1. 과거의 데이터를 준비합니다.

파일경로 = 'D:\\code\\vs\\Tensorflow_practice2\\boston.csv'
보스턴 = pd.read_csv(파일경로)

# 종속변수, 독립변수

독립 = 보스턴[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
종속 = 보스턴[['medv']]
print(독립.shape, 종속.shape)

# 2. 모델의 구조를 만듭니다.
X = tf.keras.layers.Input(shape=[13])
H = tf.keras.layers.Dense(10, activation='swish')(X)
Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

print(model.summary())

# 3. 데이터로 모델을 학습(FIT) 합니다.
model.fit(독립, 종속, epochs=100, verbose=0)
model.fit(독립, 종속, epochs=10)


# 4. 모델을 이용합니다.
print(model.predict(독립[:5]))
print(종속[:5])

# 1. 과거의 데이터를 준비합니다.
파일경로2 = 'D:\\code\\vs\\Tensorflow_practice2\\iris.csv'
아이리스 = pd.read_csv(파일경로2)

# 원핫인코딩
아이리스 = pd.get_dummies(아이리스)

# 종속변수, 독립변수
print(아이리스.columns)
독립2 = 아이리스[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속2 = 아이리스[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(독립2.shape, 종속2.shape)

# 2. 모델의 구조를 만듭니다.
X = tf.keras.layers.Input(shape=[4])
H = tf.keras.layers.Dense(8, activation='swish')(X)
H = tf.keras.layers.Dense(8, activation='swish')(H)
H = tf.keras.layers.Dense(8, activation='swish')(H)
Y = tf.keras.layers.Dense(3, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

print(model.summary())

# 3. 데이터로 모델을 학습(FTI) 시킵니다.
model.fit(독립2, 종속2, epochs=90)

# 4. 모델을 이용합니다.
print(model.predict(독립2[0:5]))
print(종속2[0:5])