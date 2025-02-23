import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('Dataset .csv')
df = df.drop(columns=['Restaurant ID','Locality','Switch to order menu'],axis=1)
df['Cuisines']=df['Cuisines'].fillna(df['Cuisines'].mode()[0])
rating_mapping = {
    'Excellent': 5,
    'Very Good': 4,
    'Good': 3,
    'Average': 2,
    'Poor': 1,
    'Not rated': 0
}
df['Rating'] = df['Rating text'].map(rating_mapping)
df.drop(columns=['Rating text'],axis=1,inplace=True)
df['Name_Length'] = df['Restaurant Name'].apply(len)
df['Address_Length'] = df['Address'].apply(len)
status_mapping = {
    'No': 0,
    'Yes': 1
}
df['Has Table booking']=df['Has Table booking'].map(status_mapping)
df['Has Online delivery']=df['Has Online delivery'].map(status_mapping)
df['Is delivering now']=df['Is delivering now'].map(status_mapping)
df.drop(columns=['Restaurant Name','Rating color'],axis=1,inplace=True)
categorical_col=['Currency','Cuisines','Locality Verbose','Address','City']
for col in categorical_col:
   frequency_encoding = df[col].value_counts()
   df[col] = df[col].map(frequency_encoding)
x=df.drop(columns=['Aggregate rating'])
y=df['Aggregate rating']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
import xgboost as xgb
simple_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    seed=123,
    learning_rate=0.1,
)
simple_reg.fit(x_train,y_train)

y_pred1 = simple_reg.predict(x_test)
import pickle
with open('aggragaterating.pkl','wb') as model_file:
  pickle.dump(simple_reg,model_file)