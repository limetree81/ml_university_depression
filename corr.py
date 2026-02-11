#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import streamlit as st

#%%
df = pd.read_csv("student_lifestyle_100k.csv")
df

#%%
df.columns

#%%
cols = ['Student_ID', 'Age', 'Gender', 'Department', 'CGPA', 'Sleep_Duration',
       'Study_Hours', 'Social_Media_Hours', 'Physical_Activity',
       'Stress_Level', 'Depression']

num_cols = ['Age', 'CGPA', 'Sleep_Duration',
       'Study_Hours', 'Social_Media_Hours', 'Physical_Activity',
       'Stress_Level'
       ]

bin_cols = ['Gender', 'Depression']
cat_cols = ['Department']
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
#xt = scaler.fit_transform(df[num_cols], yt)

##%%
from sklearn.preprocessing import OneHotEncoder

bin_encoder = OneHotEncoder(sparse_output=False, drop='first')
bin_encoder.fit_transform(df[bin_cols])

cat_encoder = OneHotEncoder(sparse_output=False)
cat_encoder.fit_transform(df[cat_cols])

xt2 = np.concatenate([
    scaler.fit_transform(df[num_cols]),
    bin_encoder.fit_transform(df[bin_cols]), 
    #cat_encoder.fit_transform(df[cat_cols])
    ], 
    axis=1)

processed_df = pd.DataFrame(xt2, columns=num_cols + bin_cols)
#%%


# 1. 상관관계 행렬 계산
corr_matrix = processed_df.corr()

# 2. 상삼각행렬(Upper Triangle)만 추출하여 중복 및 자기 자신(1.0) 제거
# np.triu는 대각선 위쪽만 남깁니다. k=1을 주어 대각선(자기 자신)까지 제거합니다.
sol = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 3. Series 형태로 풀기(stack) 및 인덱스 초기화
corr_series = sol.stack().reset_index()
corr_series.columns = ['Feature 1', 'Feature 2', 'Correlation']

# 4. 'Abs_Correlation'(절댓값) 컬럼 생성 및 정렬
corr_series['Abs_Correlation'] = corr_series['Correlation'].abs()
sorted_corr = corr_series.sort_values(by='Abs_Correlation', ascending=False)

# 결과 출력
print(sorted_corr.head(10))

#%%
corr_mat = processed_df.corr()
np.fill_diagonal(corr_mat.values, 0)
sns.heatmap(corr_mat, cmap='coolwarm', center=0, vmin=-1.0, vmax=1.0)
plt.tight_layout()
plt.show()
