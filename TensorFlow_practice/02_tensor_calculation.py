import numpy as np
import tensorboard
import tensorflow as tf

print(tf.constant(2) + tf.constant(2))
print(tf.constant(2) - tf.constant(2))
print(tf.add(tf.constant(2), tf.constant(2)))
print(tf.subtract(tf.constant(2), tf.constant(2)))

print(tf.constant(2) * tf.constant(2))
print(tf.constant(2) / tf.constant(2))
print(tf.multiply(tf.constant(2), tf.constant(2)))
print(tf.divide(tf.constant(2), tf.constant(2)))

# print(tf.constant(2) + tf.constant(2.2)) # 같은 데이터 타입이 아니면 연산이 되지 않음

print(tf.cast(tf.constant(2), tf.float32) + tf.constant(2.2))