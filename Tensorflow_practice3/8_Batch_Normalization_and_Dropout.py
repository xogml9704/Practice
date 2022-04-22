# 배치 정규화(Batch Normalization)
#  신경망에 데이터를 입력으로 넣을 때는 스케일러를 활용해 모든 데이터를 공통 범위로 배치한다.
#  매우 다른 크기의 데이터는 다른 크기의 활성화를 생성하는 경향이 있어 훈련을 불안정하게 한다.

# standarScaler 평균과 표준편차 활용 (평균을 제거하고 데이터를 분산으로 조정)

# MinMaxScaler 최대/최소값이 각각 1과 0이 되도록 함

# RobustScaler 이상치 영향을 최소화함, IQR 활용
import tensorflow as tf

# 모델
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 드롭아웃(Dropout)

# 모델
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])