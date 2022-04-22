# 레이어(layer)
#  하나의 출력(텐서)
#  레이어(tf.keras.layers)
#  하나의 입력(텐서)

# Sequential 모델
#  하나의 입력과 하나의 출력이 있는 레이어을 쌓을 떄 적합함
#  다중 입력 또는 다중 출력이 있는 경우 적합하지 않음 (함수형 모델에서는 가능)

import tensorflow as tf

# 시퀸셜 모델 만드는 방법1 (콤마 주의)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# 시퀀셜 모델 만드는 방법 2 (add 메소드 활용)
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(units=128, activation='relu'))
# model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# 입력
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Flatten layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name="A"),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.summary()