#%%
import pandas as pd
import numpy as np
# 데이터 불러오기
# 경로
path = 'D:\\code\\data\\challenge\\'

# 데이터 불러오기
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
data = pd.read_csv(path + "icml_face_data.csv")

#%%
print(train.shape, test.shape, data.shape)
#%%
data.head()
#%%
# 데이터 종류
data['emotion'].value_counts()


# %%

# 0 : 'Angry' 1 : 'Disgust', 2 : 'Fear', 3 : 'Hapeey', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'
data['emotion'].value_counts().plot(kind='barh', figsize=(8,4))

# %%

data['Usage'].value_counts()

# %%
data['Pixle'].value_counts()
# %%
