# 데이터 타입 조정
#  변수(칼럼) 타입 확인 : 데이터.dtypes
#  변수를 범주형으로 변경 :
#   데이터['칼럼명'].astype('category')
 
#  변수를 수치형으로 변경 :
#   데이터['칼럼명'].astype('int')
#   데이터['칼러명'].astype('float')
 
#  NA 값의 처리
#   NA 갯수 체크 : 데이터.isna().sum()
#   NA 값 채우기 : 데이터['칼럼명'].fillna(특정숫자)

import pandas as pd

파일경로 = 'D:\\code\\vs\\Tensorflow_practice2\\iris2.csv'
아이리스 = pd.read_csv(파일경로)
print(아이리스.head())

# 원 핫 인코딩
인코딩 = pd.get_dummies(아이리스)
print(인코딩.head())

print(아이리스.dtypes)

# 품종 타입을 범주형으로 바꾸어 준다.
아이리스['품종'] = 아이리스['품종'].astype('category')
print(아이리스.dtypes)

# 원 핫 인코딩
인코딩 = pd.get_dummies(아이리스)
print(인코딩.head())

# NA 값을 체크해 봅시다.
print(아이리스.isna().sum())

print(아이리스.tail())

# NA 값에 꽃잎폭 평균값을 넣어주는 방법
mean = 아이리스['꽃잎폭'].mean()
아이리스['꽃잎폭'] = 아이리스['꽃잎폭'].fillna(mean)
print(아이리스.tail())