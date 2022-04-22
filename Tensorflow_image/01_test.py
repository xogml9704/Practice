#%%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import tensorflow as tf
import keras
#%%
print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

from keras.models import load_model
model = load_model('MNIST_CNN_model.h5')

model.summary()
#%%
import matplotlib.pyplot as plt

test1 = plt.imread('D:\\code\\vs\\Tensorflow_image\\0.PNG')
plt.imshow(test1)
#%%
test_num = plt.imread('./my_num/re_1.jpg')
test_num = test_num[:,:,0]
test_num = (test_num > 125) * test_num
test_num = test_num.astype('float32') / 255.

plt.imshow(test_num, cmap='Greys', interpolation='nearest');

test_num = test_num.reshape((1, 28, 28, 1))

print('The Answer is ', model.predict_classes(test_num))