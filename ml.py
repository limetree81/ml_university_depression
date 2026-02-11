#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

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
       'Stress_Level'
       ]

bin_cols = ['Gender']
cat_cols = ['Department']
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
#xt = scaler.fit_transform(df[num_cols], yt)

##%%
from sklearn.preprocessing import OneHotEncoder

bin_encoder = OneHotEncoder(sparse_output=False, drop='first')
bin_encoder.fit_transform(df_train[bin_cols])

cat_encoder = OneHotEncoder(sparse_output=False)
cat_encoder.fit_transform(df_train[cat_cols])

xt2 = np.concatenate([
    scaler.fit_transform(df_train[num_cols]),
    bin_encoder.fit_transform(df_train[bin_cols]), 
    cat_encoder.fit_transform(df_train[cat_cols])], 
    axis=1)

xt2.shape

#%%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
#rf = RandomForestClassifier(class_weight='balanced')
rf.fit(xt2, yt)

#%%
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
    #('classifier', RandomForestClassifier())
    #('classifier', RandomForestClassifier(class_weight='balanced'))
    ('classifier', LogisticRegression())
])

# 3. Fit on the ORIGINAL training data (not xt2)
# The Pipeline handles the transformation, so you feed it the raw DataFrame
model.fit(df_train.drop('Depression', axis=1), yt)


#%%

#(model.predict(xt) == yt).mean() #np.float64(0.999975)
(model.predict(xv) == yv).mean() #np.float64(0.8977)

#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(xv)
cm = confusion_matrix(yv, y_pred)
ConfusionMatrixDisplay(cm).plot()
cm

#%%
from sklearn.metrics import classification_report
print(classification_report(yv, model.predict(xv)))

#%%
import pickle
pickle.dump(rf, open('rf.pickle','wb'))
