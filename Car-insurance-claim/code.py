# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df=pd.read_csv(path)
print(df.head())
print(df.info())
column_names=['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
df[column_names]=df[column_names].replace({"\$":"",",":""},regex=True)
print(df.head())
X=df.drop('CLAIM_FLAG',1)
y=df['CLAIM_FLAG']
count=y.value_counts()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=6)



# Code ends here


# --------------
# Code starts here
X_train[columns]=X_train[columns].astype(float)
X_test[columns]=X_test[columns].astype(float)
print(X_train.isnull().sum())
print(X_test.isnull().sum())


# Code ends here


# --------------
X_train.dropna(subset=['YOJ', 'OCCUPATION'], inplace=True)
X_test.dropna(subset=['YOJ', 'OCCUPATION'], inplace=True)
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

column_mean = ['AGE', 'CAR_AGE', 'INCOME', 'HOME_VAL']

for column in column_mean:
    mean_train = X_train[column].mean()
    mean_test  = X_test[column].mean()
    X_train[column].fillna(value=mean_train, inplace=True)
    X_test[column].fillna(value=mean_test, inplace=True)


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le = LabelEncoder()

for column in columns:
    X_train[column] = le.fit_transform(X_train[column].astype('str'))
    X_test[column] = le.transform(X_test[column].astype('str'))




# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model=LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)


# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote=SMOTE(random_state=9)
X_train,y_train=smote.fit_sample(X_train,y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




# Code ends here


# --------------
# Code Starts here
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)

# Code ends here


