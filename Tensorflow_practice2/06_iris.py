# 양적 -> 회귀(regression)
# 범주형 -> 분류(classification)

# 1. 과거의 데이터를 준비합니다.
아이리스 = pd.read_csv('iris.csv')
아이리스 = pd.get_dummies(아이리스)

독립 = 아이리스[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속 = 아이리스[['품종']]
print(독립.shape, 종속.shape)

#2. 모델의 구조를 만듭니다.
X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy')

#3. 데이터로 모델을 학습(FIT) 합니다.
model.fit(독립, 종속, epochs=1000)

#4. 모델을 이용합니다.
print("Predictions : ", model.predict(독립[0:5]))