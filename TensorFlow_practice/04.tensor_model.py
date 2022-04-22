# 모델(Model)
#  Sequential()
#  서브클래싱(Subclassing)
#  함수형 API

# Sequential()
#  모델이 순차적인 구조로 진행할 때 사용
#  간단한 방법
#   Sequential 객체 생성 후, add()를 이용한 방법
#   Sequential 인자에 한번에 추가 방법
#  다중 입력 및 출력이 존재하는 등의 복잡한 모델을 구성할 수 없음

from tensorflow.python.keras.layers import Dense, Input, Flatten
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.utils.vis_utils import plot_model

model = Sequential()
model.add(Input(shape=(28, 28)))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model2 = Sequential([Input(shape=(28, 28), name='Input'),
                   Dense(300, activation='relu', name='Dense1'),
                   Dense(100, activation='relu', name='Dense2'),
                   Dense(10, activation='softmax', name='Output')])

model2.summary()

# 함수형 API
#  가장 권장되는 방법
#  모델을 복잡하고, 유연하게 구성 가능
#  다중 입출력을 다룰 수 있음

inputs = Input(shape=(28, 28, 1))
x = Flatten(input_shape=(28,28,1))(inputs)
x = Dense(300, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()

from tensorflow.python.keras.layers import Concatenate

input_layer = Input(shape=(28, 28))
hidden1 = Dense(100, activation='relu')(input_layer)
hidden2 = Dense(30, activation='relu')(hidden1)
concat = Concatenate()([input_layer, hidden2])
output = Dense(1)(concat)

model = Model(inputs=[input_layer], outputs=[output])
model.summary()

input_1 = Input(shape=(10, 10), name='input_1')
input_2 = Input(shape=(10, 28), name='input_2')

hidden1 = Dense(100, activation='relu')(input_2)
hidden2 = Dense(10, activation='relu')(hidden1)
concat = Concatenate()([input_1, hidden2])
output = Dense(1, activation='softmax', name='output')(concat)

model = Model(inputs=[input_1, input_2], outputs=[output])
model.summary()

input_ = Input(shape=(10, 10), name='input_')
hidden1 = Dense(100, activation='relu')(input_)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid', name='main_output')(hidden2)
sub_out = Dense(1, name='sum_output')(hidden2)

model = Model(inputs=[input_], outputs=[output, sub_out])
model.summary()

input_1 = Input(shape=(10, 10), name='input_1')
input_2 = Input(shape=(10, 28), name='input_2')
hidden1 = Dense(100, activation='relu')(input_2)
hidden2 = Dense(10, activation='relu')(hidden1)
concat = Concatenate()([input_1, hidden2])
output = Dense(1, activation='sigmoid', name='main_output')(concat)
sub_out = Dense(1, name='sum_output')(hidden2)

model = Model(inputs=[input_1, input_2], outputs=[output, sub_out])
model.summary()