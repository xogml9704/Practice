# 1. 과거의 데이터를 준비합니다.
(독립, 종속), _ = tf.keras.datasets.mnist.load_data()

종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape)
# (60000, 28, 28), (60000, 1)

# 2. 모델의 구조를 만듭니다.
X = tf.keras.layers.Input(shape=[28, 28])
H = tf.keras.layers.Flatten()(X)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='scategorical_crossentropy', metrics='accuracy')

# 3. 데이터로 모델을 학습(FTI) 시킵니다.
model.fit(독립, 종속, epochs=10)

# 4. 모델을 이용합니다.
print("Predictions : ", model.predict(독립[0:5]))
123