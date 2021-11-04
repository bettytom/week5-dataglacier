import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
marketing=pd.read_csv('DirectMarketing.csv')
from sklearn.preprocessing import LabelEncoder
encod=LabelEncoder()
marketing['Age']=encod.fit_transform(marketing['Age'])
marketing['Gender']=encod.fit_transform(marketing['Gender'])
marketing['OwnHome']=encod.fit_transform(marketing['OwnHome'])
marketing['Married']=encod.fit_transform(marketing['Married'])
marketing['Location']=encod.fit_transform(marketing['Location'])
marketing['History']=encod.fit_transform(marketing['History'])

marketing['History'].fillna(method='ffill',limit=4,inplace=True)
marketing['History'].isna().sum()

X=marketing[['Salary','OwnHome','Gender','Children','History','Catalogs','Location','Married','Age']]
Y=marketing['AmountSpent']

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,Y,train_size=0.7)
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression_fit=regression.fit(train_x,train_y)
predict_regression=regression.predict(test_x)
import joblib
joblib.dump(predict_regression,'linear.pkl')
linearmodel=joblib.load(open('linear.pkl','rb'))

