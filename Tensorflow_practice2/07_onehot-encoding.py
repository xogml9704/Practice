# 원핫인코딩
# 범주를 칼럼으로 만들어 줘야 한다.

아이리스 = pd.read_csv('iris.csv')

# 원핫인코딩 
아이리스 = pd.get_dummies(아이리스)