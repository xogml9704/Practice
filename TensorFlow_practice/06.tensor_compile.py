from tensorflow.python.keras.layers import Dense, Input, Flatten
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.layers import Concatenate
# 모델 컴파일(compile)
#  모델을 구성한 후, 사용한 손실 함수(loss function), 옵티마이저(optimizer)를 지정

# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])

#  손실 함수(Loss Function)
#   학습이 진행되면서 해당 과정이 얼마나 잘 되고 있는지 나타내는 지표
#  모델이 훈련되는 동안 최소화될 값으로 주어진 문제에 대한 성공 지표
#  손실 함수에 따른 결과를 통해 학습 피라미터를 조정
#  최적화 이론에서 최소화 하고자 하는 함수
#  미분 가능한 함수 사용
#  Keras에서 주요 손실 함수 제공
#   sparse_categorical_crossentropy : 클래스가 배타적 방식으로 구분.
#    즉(0,1,2,...,9)와 같은 방식으로 구분되어 있을 때 사용
#   categorical_cross_entropy : 클래스가 원-핫 인코딩 방식으로 되어 있을 때 사용
#   binary_crossentropy : 이진 분류를 수행할 때 사용

# 평균절대오차(Mean Absolute Error, MAE)
#  오차가 커져도 손실함수가 일정하게 증가
#  이삼치(Outiler)에 강건함(Robust)
#   데이터에서 [입력-정답] 관계가 적절하지 않은 것이 있을 경우에, 좋은 추정을 하더라도
#   오차가 발생하는 경우가 발생
#   해당 이상치에 해당하는 지점에서 손실 함수의 최소값으로 가는 정도의 영향력이 크지 않음
#  회귀(Regression)에 많이 사용

# 원-핫 인코딩(One-Hot Encoding)
#  범주형 변수를 표현할 떄 사용
#  가변수(Dummy Variable)이라고도 함
#  정답인 레이블을 제외하고 0으로 처리

# 교차 엔트로피 오차(Cross Entropy Error, CEE)
#  이진 분류(Binary Classification), 다중 클래스 분류(Multi Class Classification)
#  소프트 맥스(softmax)와 원-핫 인코딩(one-hot encoding) 사이의 출력 간 거리를 비교
#  정답인 클래스에 대해서만 오차를 계산
#  정답을 맞추면 오차가 0, 틀리면 그 차이가 클수록 오차가 무한히 커짐
#  y = log(x)
#   x 가 1에 가까울수록 0에 가까워짐
#   x 가 0에 가까울수록 y값은 무한히 커짐
