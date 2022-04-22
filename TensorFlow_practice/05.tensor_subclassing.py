# 서브 클래싱(Subclassing)
#  커스터마이징에 최적화된 방법
#  Model 클래스를 상속받아 Model이 포함하는 기능을 사용할 수 있음
#   fit(), evaluate(), predict()
#   save(), load()
#  주로 call() 메소드안에서 원하는 계산 가능
#   for, if, 저수준 연산 등
#  권장되는 방법은 아니지만 어떤 모델의 구현 코드를 참고할 때, 해석할 수 있어야 함
from tensorflow.python.keras.layers import Dense, Input, Flatten
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.layers import Concatenate

class MyModel(Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super(MyModel. self).__init__(**kwargs)
        self.dense_layer1 = Dense(300, activation=activation)
        self.dense_layer2 = Dense(100, activation=activation)
        self.dense_layer3 = Dense(units, activation=activation)
        self.output_layer = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        x = self.dense_layer3(x)
        x = self.output_layer(x)
        return x

inputs = Input(shape=(28, 28, 1))
x = Flatten(input_shape=(28, 28, 1))(inputs)
x = Dense(300, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()

print(model.layers)

hidden_2 = model.layers[2]
print(hidden_2.name)
print(model.get_layer('dense') is hidden_2)

weights, biases = hidden_2.get_weights()
print(weights.shape)
print(biases.shape)

print(weights)
print(biases)