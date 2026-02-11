#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from minepy import MINE

#%%
df = pd.read_csv("student_lifestyle_100k.csv")
df

#%%
df.columns

#%%
df.mean(numeric_only=True)


#%%
df['Department'].value_counts()
df['Gender'].value_counts()

#%%
test_size = int(0.2 * len(df))
index = np.random.permutation(len(df))
test_index = index[:test_size]
train_index = index[test_size:]
df_test = df.iloc[test_index]
df_train = df.iloc[train_index]

xt = df_train.drop('Depression', axis=1)
yt = df_train['Depression']
xv = df_test.drop('Depression', axis=1)
yv = df_test['Depression']
#%%
cols = ['Student_ID', 'Age', 'Gender', 'Department', 'CGPA', 'Sleep_Duration',
       'Study_Hours', 'Social_Media_Hours', 'Physical_Activity',
       'Stress_Level', 'Depression']

num_cols = ['Age', 'CGPA', 'Sleep_Duration',
       'Study_Hours', 'Social_Media_Hours', 'Physical_Activity',
       #'Stress_Level'
       ]

bin_cols = ['Gender']
cat_cols = ['Department']
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1. Define the preprocessing for different column types
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('bin', OneHotEncoder(sparse_output=False, drop='first'), bin_cols),
    ('mul', OneHotEncoder(sparse_output=False), cat_cols)
])

# 2. Create the Pipeline
# The pipeline now has two clear steps: 'pre' (the ColumnTransformer) and 'classifier'
model = Pipeline([
    ('pre', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced'))
    #('classifier', LogisticRegression())
])

# 3. Fit on the ORIGINAL training data (not xt2)
# The Pipeline handles the transformation, so you feed it the raw DataFrame
model.fit(df_train.drop('Depression', axis=1), yt)


#%%

(model.predict(xt) == yt).mean() #np.float64(0.999975)
(model.predict(xv) == yv).mean() #np.float64(0.8977)

#%%
# 1. 일반적인 predict 대신, 클래스별 확률값을 가져옵니다.
# 결과는 [False일 확률, True일 확률] 형태의 배열입니다.
y_probs = model.predict_proba(xv)[:, 1] 

# 2. 임계값(Threshold)을 설정합니다. (예: 0.1)
# 0.5 대신 0.1로 낮추면 Recall(재현율)이 비약적으로 상승합니다.
threshold = 0.1
y_new_pred = (y_probs >= threshold).astype(int)

# 3. 새로운 결과로 성능 리포트 출력
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print(f"--- 임계값 {threshold} 적용 결과 ---")
print(classification_report(yv, y_new_pred))

# 4. 혼동 행렬 시각화
cm_new = confusion_matrix(yv, y_new_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_new)
disp.plot()
plt.show()

#%%
from minepy import MINE
import pandas as pd

# MINE 객체 생성
mine = MINE()

mic_results = []

# 수치형 변수들에 대해 MIC 계산
for col in num_cols:
    mine.compute_score(df_train[col], yt)
    mic_results.append({
        'Feature': col,
        'MIC': mine.mic(),
        'Pearson': df_train[col].corr(yt) # 비교를 위해 기존 상관계수도 포함
    })

mic_df = pd.DataFrame(mic_results).sort_values(by='MIC', ascending=False)
print(mic_df)

# #%%
# import numpy as np
# from sklearn.metrics import classification_report

# def verify_report(tn, fp, fn, tp):
#     # 1. 주신 수치를 바탕으로 가상의 y_true와 y_pred 생성
#     y_true = [0] * (tn + fp) + [1] * (fn + tp)
#     y_pred = [0] * tn + [1] * fp + [0] * fn + [1] * tp
    
#     # 2. sklearn의 실제 리포트 출력
#     print(f"--- Input: TN={tn}, FP={fp}, FN={fn}, TP={tp} ---")
#     print(classification_report(y_true, y_pred, target_names=['False', 'True'], digits=4))

# # --- Case 1 검증 ---
# verify_report(17989, 44, 1933, 34)

# # --- Case 2 검증 ---
# verify_report(18033, 0, 1967, 0)
# %%
