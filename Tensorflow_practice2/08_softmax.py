
X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy')

# y = f(w1x1 + x2x2 + w3x3 + w4x4 = b)
# f - 회귀모델 : Identity (y = x)
#   - 분류모델 : softmax

# 분류에 쓰는 loss = categorical_crossentropy
# 회귀에 쓰는 loss = mse

# 정확도를 볼 수 있는 코딩
model.compile(loss = 'categorical_crossentropy', metrics='accuracy')