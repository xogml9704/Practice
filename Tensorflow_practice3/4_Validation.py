import tensorflow as tf

#Mnist 데이터 셋
mnist = tf.keras.datasets.mnist
(x, y),(x_test, y_test) = mnist.load_data()

# 방법 1 : validation_split 활용
model.fkt(x, y, validation_split=0.2, epochs=5)

# 방법 2 : train_test_split 활용
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.2)

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5)

history = model.fit(x, y, validation_split=0.2, epochs=5, batch_size=64)
