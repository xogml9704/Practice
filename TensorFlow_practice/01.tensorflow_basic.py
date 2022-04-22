# 텐서(Tensor)
# Rank : 축의 개수
# Shape : 형상(각 축에 따른 개수)
# Type : 데이터 타입

import numpy as np
import tensorboard
import tensorflow as tf

# 0D Tensor(Scalar)
#  하나의 숫자를 담고 있는 텐서(Tensor)
#  축과 형상이 없음

t0 = tf.constant(1)
print(t0)
print(tf.rank(t0))

# 1D Tensor(Vector)
#  값들을 저장힌 리스트와 유사한 텐서
#  하나의 축이 존재

t1 = tf.constant([1,2,3])
print(t1)
print(tf.rank(t1))

# 2D Tensor(Matrix)
#  행렬과 같은 모양으로 두 개의 축이 존재
#  일반적인 수치, 통계 데이터 셋이 해당
#  주로 샘플(samples)과 특성(features)을 가진 구조로 사용

t2 = tf.constant([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
print(t2)
print(tf.rank(t2))

# 3D tensor
#  큐브(cube)와 같은 모양으로 세개의 축이 존재
#  데이터가 연속된 시퀸스 데이터나 시간 축이 포함된 시계열 데이터에 해당
#  주식 가격 데이터셋, 시간에 따른 질병 발병 데이터 등이 존재
#  주로 샘플(samples), 타임스탭(Timesteps), 특셩(features)을 가진 구조로 사용

t3 = tf.constant([[[1,2,3],
                  [4,5,6],
                  [7,8,9]],
                  [[1,2,3],
                  [4,5,6],
                  [7,8,9]],
                  [[1,2,3],
                  [4,5,6],
                  [7,8,9]]])
print(t3)
print(tf.rank(t3))

# 4D Tensor
#  4개의 축
#  컬러 이미지 데이터가 대표적인 사례(흑백 이미지 데이터는 3D Tensor로 가능)
#  주로 샘플(samples), 높이(height), 너비(width), 컬러 채널(channel)을 가진 구조로 사용

# 5D Tensor
#  5개의 축
#  비디오 데이터가 대표적인 사례
#  주로 샘플(samples), 프레임(frames), 높이(height), 너비(width), 컬러(channel)을 가진 구주로 사용

# 텐서 데이터 타입
#     텐서의 기본 dtype
#         정수형 텐서 : int
# 32
#         실수형 텐서 : float32
#         문자열 텐서 : string
#     int32, float32, string 타입 외에도 float16, int8 타입 등이 존재
#     연산시 텐서의 타입 일치 필요
#     타입변환에는 tf.cast() 사용

i = tf.constant(2)
print(i)

f = tf.constant(2.)
print(f)

s = tf.constant('Suan')
print(s)

f16 = tf.constant(2.,dtype=tf.float16)
print(f16)

i8 = tf.constant(2, dtype=tf.int8)
print(i8)

f32 = tf.cast(f16, tf.float32)
print(f32)

i32 = tf.cast(i8, tf.int32)
print(i32)