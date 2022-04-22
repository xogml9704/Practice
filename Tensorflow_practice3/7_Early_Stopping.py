import tensorflow as tf

# 조기종료

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
# monitor : val_accuracy 관찰, patience : 에폭 3번까지 개선되지 않을 시 종료

history = model.fit(X, Y, validation_split=0.2, epochs=30, batch_size=64, callbacks=[early_stopping])

# callbacks(콜백)은 학습 실행 시 손실값 등을 모니터링 하고 손실에 따라 일부 작업을 수행할 수 있음

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# monitor : val_loss 관찰, patience : 에폭 3번까지 개선되지 않을 시 종료

history = model.fit(x, y, validation_split=0.2, epochs=30, batch_size=64, callbacks=[early_stopping])

# min_delta 기준 미달시 조기 종료
# 3epoch 동안 손실이 0.05 이상 개선되지 않을 경우 훈련 중지
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.05)
history = model.fit(x, y, validation_split=0.2, epochs=30, batch_size=64, callbacks=[early_stopping])

# 정리
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.05)

monitor : 모니터 하고자 하는 값(val_loss, val_accuracy 등)
patience : 모니터 하는 값의 개선이 없는 경우 몇 번의 epoch를 진행할지 값
min_delta : 개선이라고 말하고자 하는 값의 크기
