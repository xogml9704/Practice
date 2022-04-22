표의 열 vs 포함관계

x1 = np.array([5.1, 3.5, 1.4, 0.2])
print(x1.ndim, x1.shape)
# 1차원 형태, (4,)

x2 = np.array([4.9, 3.0, 1.4, 0.2])
x3 = np.array([4.7, 3.2, 1.3, 0.2])

아이리스 = np.array([x1, x2, x3])
print(아이리스.ndim, 아이리스.shape)
# 2차원 형태, (3,4)

img1 = np.array([
    [ 0, 255],
    [255, 0]
])
print(img1.ndim, img1.shape)
# 2차원 형태, (2,2)

img2 = np.array([255, 255], [255, 255])
img3 = np.array([0, 0], [0, 0])

이미지셋 = np.array([img1, img2, img3])
print(이미지셋.ndim, 이미지셋.shape)
# 3차원형태, (3, 2, 2)

4차원 공간에 표현되는 관측치

d1 = np.array([1,2,3])
print(d1.ndim, d1.shape)
# 1차원 형태, (3,)

d2 = np.array([d1, d1, d1, d1, d1])
print(d2.ndim, d2.shape)
# 2차원 형태 (5, 3)

d3 = np.array([d2, d2, d2, d2])
print(d3.ndim, d3.shape)
# 3차원 형태 (4, 5, 3)

d4 = np.array([d3, d3])
print(d4.ndim, d4.shape)
# 4차원 형태, (2, 4, 5, 3)

배열의  깊이 = 차원 수 

# MNIST
(독립, 종속),_=tf.keras.datasets.mnist.load_data()
print(독립.shape, 종속.shape)
# (60000, 28, 28) (60000,)

# CIFAR10
(독립, 종속),_=tf.keras.datasets.cifar10.load_data()
print(독립.shape, 종속.shape)
# (50000, 32, 32, 3) (50000, 1)