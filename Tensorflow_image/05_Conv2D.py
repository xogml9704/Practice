특정한 패턴의 특징이 어디서 나타나는지를 확인하는 도구
"Convolution"

특징 맵 = feature map

# 1. 과거의 데이터를 준비합니다.
(독립, 종속), _ = tf.keras.datasets.mnist.load_data()
독립 = 독립.reshape(60000, 28, 28, 1)
종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape)

# 2. 모델의 구조를 만듭니다.
X = tf.keras.layers.Input(shape=[28, 28, 1])
H = tf.keras.layers.Conv2D(3, kernel_size=5, activation='swish')(X) # 3개의 특징맵 = 3채널의 특징맵
H = tf.keras.layers.Conv2D(6, kernal_size=5, activation='swish')(H) # 6개의 특징맵 = 6채널의 특징맵
H = tf.keras.layers.Flatten()(H) # 표로 만든다
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss = 'scategorical_crossentropy', metrics='accuracy')