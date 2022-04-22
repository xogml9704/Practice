# 모델 저장
#  1. 가중치 값만 저장

#  2. 모델 전체 저장

# 모델 가중치 저장
import tensorflow as tf

cp_path = 'model_save/cp.ckpt'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                verbose=1)


# 모델(함수)
def load_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics='accuracy')
    
    model = load_model()

    model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=64, callbacks=[checkpoint])

# 저장된 모델 가중치 불러오기

# 새 모델
model = load_model()
model.evaluate(x_test, y_test)

# 저장된 모델 가중치 불러오기
model.load_weights(cp_path)
model.evaluate(x_test, y_test)

# 모델 저장(HDF5)

# 방법 1
import tensorflow as tf
model = load_model()
checkpoint = tf.keras.callbacks.ModelCheckpoint('model_save.h5') # save_weights_only=False
model.fit(x_train, y_train, epochs=3, callbacks=[checkpoint])

# 방법 2
model = load_model()
model.fit(x_train, y_train, epochs=3)
model.save('model_save2.h5')

# 모델 불러오기(HDF5)
model = load_model('model_save.h5')
model.evaluate(x_test, y_test)