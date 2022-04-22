# 컴파일이란?
#  컴파일러(compiler, 순화용어 : 해석기, 번역기)는 특정 프로그래밍 언어로 쓰여
#  있는 문서를 다른 프로그래밍 언어로 옮기는 언어 번역 프로그램을 말한다.

# 라이브러리 불러오기
import tensorflow as tf

# 모델 만들기(다중 분류)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일 방법 1
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

# Loss(손실 함수) -> Optimizer(최적화) -> Metrics(지표)
#  훈련(학습) 하는 동안 오차를 측정함, 이 함수를 최소화 해야함

# 훈련하는 동안 최소화 시켜야 함(Optimizer)
#  Ex) SDG, Adagrad, RMSProp, Adam

# 다중분류
model.compile(optimizer='adam',
             loss='categorical_crossentropy', 
             metrics='accuracy')

# 이진분류
model.compile(optimizer='adam',
             loss='binary_crossentropy', 
             metrics='accuracy')

# 회귀
model.compile(optimizer='adam',
             loss='mse', 
             metrics='mae')

#  sparse_categorical_crossentropy
#   Label : 1, 2, 3
 
#  categorical_crossentropy
#   Label : [1,0,0], [0,1,0], [0,0,1]

# Optimizer(최적화 알고리즘)
#  손실함수를 바탕으로 모델의 업데이트 방법을 결정함

# metrics(지표)
#  훈련과 테스트 단계 모니터링
#  분류 : accuracy, 회귀 : MAE, MSE 등
#   MAE : 실제값과 예측값 차이를 절대값으로 변환해 평균화 한 것
#   MSE : 실제값과 예측값 차이를 제곱해 평균한 것