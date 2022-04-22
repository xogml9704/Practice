# 데이터 불러오기

# 파일경로 = 'lemonade.csv'
# 데이터 = pd.read_csv(파일경로)

# import pandas as pd

# #독린변수와, 종속변수의 분리
# 독립 = 데이터[['온도']]
# 종속 = 데이터[['판매량']]

# #데이터 모양 확인
# print(독립.shape, 종속.shape)

# 데이터 준비하기 1
#  실습을 통해 배울 도구들
#   파일 읽어오기 : pd.read_csv('/경로/파일명.csv')
#   모양 확인하기 : print(데이터.shape)
#   칼럼 선택하기 : 데이터[['칼럼명1', '칼럼명2', '칼럼명3']]
#   칼럼 이름 출력하기 : print(데이터.columns)
#   맨 위 5개 관측치 출력하기 : 데이터.head()

import pandas as pd

# 파일들로부터 데이터 읽어오기
파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
레모네이드 = pd.read_csv(파일경로)

파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
보스턴 = pd.read_csv(파일경로)

파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
아이리스 = pd.read_csv(파일경로)

# 데이터 모양으로 확인하기
print(레모네이드.shape)
print(보스턴.shape)
print(아이리스.shape)

# 칼럼이름 출력
print(레모네이드.columns)
print(보스턴.columns)
print(아이리스.columns)

독립 = 레모네이드[['온도']]
종속 = 레모네이드[['판매량']]
print(독립.shape, 종속.shape)

독립1 = 보스턴[['crim','zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
종속1 = 보스턴[['medv']]
print(독립1.shape, 종속1.shape)

독립2 = 아이리스[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속2 = 아이리스[['품종']]
print(독립2.shape, 종속2.shape)

print(레모네이드.head())
print(보스턴.head())
print(아이리스.head())