# 평가(정확도)
loss, accuracy = model.evaluate(x_test, y_test)
print("Test 데이터 정확도 : ", accuracy)

# 예측
predictions = model.predict(x_test)

# 첫 번째 데이터 예측 결과 값
predictions[0]

# 표현방식 변경(소수점 두자리)
import numpy as np
np.set_printoptions(formatter={'float_kind':lambda x : "{0.0.2f}".format(x)})

# 첫 번쨰 데이터 예측 결과 값
predictions[0]

# 첫 번째 데이터 에측 결과 값 그래프로 표시
import matplotlib.pyplot as plt
plt.figure(figsize=(12,2))
plt.plot(predictions[0],'o')

# 첫 번째 데이터 손 글씨
plt.imshow(x_test[0])