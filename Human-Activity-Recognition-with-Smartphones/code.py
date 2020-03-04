# --------------
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import math
import seaborn as sns 
from sklearn.svm import SVC

# Loading Data
data =pd.read_csv(path)
print(data.isnull().sum())
print(data.describe())


# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

# Store the label values 
label=data.iloc[:,-1]


# plot the countplot
g=sns.countplot(x=label)
g.set_xticklabels(label,rotation=90)



# --------------
# make the copy of dataset
data_copy=data.copy()



# Create an empty column 
data_copy['duration']= " "


# Calculate the duration
duration_df = data_copy.groupby([label[(label=='WALKING_UPSTAIRS') | (label=='WALKING_DOWNSTAIRS')],'subject'])['duration'].count()*1.28
print(type(duration_df))
duration_df=pd.DataFrame(duration_df)
plot_data=duration_df.sort_values(by='duration').reset_index()
# plot_data=duration_df.sort_values(by='duration').reset_index()
print(plot_data)
sns.barplot(data=plot_data,x='subject',y='duration',hue='Activity')



# Sort the values of duration





# --------------
#exclude the Activity column and the subject column
# feature_cols=data_copy.select_dtypes(exclude='object').columns
feature_cols=data.drop(['Activity','subject'],1).columns

#Calculate the correlation values
correlated_values=data[feature_cols].corr().stack().reset_index()    



#stack the data and convert to a dataframe
correlated_values=pd.DataFrame(data=correlated_values)
correlated_values.rename(columns={'level_0':'Feature_1','level_1':"Feature_2", 0: 'Correlation_score'},inplace=True)



#create an abs_correlation column
correlated_values['abs_correlation']=correlated_values['Correlation_score'].abs()


#Picking most correlated features without having self correlated pairs
s_corr_list=correlated_values.sort_values(by='abs_correlation',ascending=False)
top_corr_fields=s_corr_list[s_corr_list['abs_correlation']>0.8]
top_corr_fields=top_corr_fields[(top_corr_fields['Feature_1'])!=(top_corr_fields['Feature_2'])].reset_index(drop=True)



# --------------
# importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import confusion_matrix, accuracy_score

# Encoding the target variable
le=LabelEncoder()
data['Activity']=le.fit_transform(data['Activity'])

X=data.drop('Activity',1)
y=data['Activity']



# split the dataset into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=40)



# Baseline model 
classifier=SVC(random_state=0)
clf=classifier.fit(X_train,y_train)
y_pred=clf.predict(X_test)
precision,accuracy,f_score,_=error_metric(y_test,y_pred,average = 'weighted')
print(precision)
print(accuracy)
print(f_score)
model1_score=accuracy_score(y_test,y_pred)
print(model1_score)



# --------------
# importing libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel



# Feature selection using Linear SVC
lsvc = LinearSVC(C=0.01, penalty='l1',dual=False, random_state=42)
lsvc.fit(X_train,y_train)



# model building on reduced set of features
model_2 = SelectFromModel(lsvc, prefit=True)
new_train_features = model_2.transform(X_train)
new_test_features = model_2.transform(X_test)

classifier_2 = SVC()
clf_2 = classifier_2.fit(new_train_features,y_train)
y_pred_new = clf_2.predict(new_test_features)

model2_score=accuracy_score(y_test,y_pred_new)
precision,recall,f_score,_=error_metric(y_test,y_pred_new,average = 'weighted')
print(precision)
print(recall)
print(f_score)
print(model2_score)



# --------------
# Importing Libraries
from sklearn.model_selection import GridSearchCV

# Set the hyperparmeters
selector=GridSearchCV(estimator=SVC(),param_grid={'kernel': ['linear', 'rbf'],'C': [100, 20, 1, 0.1]},scoring='accuracy')
selector.fit(new_train_features,y_train)

print(selector.best_params_)
print(selector.cv_results_)

means=selector.cv_results_['mean_test_score']
stds = selector.cv_results_['std_test_score']
parameters=selector.cv_results_['params']
print(means)
print(stds)
print(parameters)

print(selector.best_params_)


# Usage of grid search to select the best hyperparmeters
classifier_3 = SVC(kernel='rbf', C=100)
   

# Model building after Hyperparameter tuning

clf_3=classifier_3.fit(new_train_features,y_train)
y_pred_final = clf_3.predict(new_test_features)
model3_score=accuracy_score(y_test,y_pred_final)
precision,recall,f_score,_ = error_metric(y_test,y_pred,average='weighted')
print(precision)
print(recall)
print(f_score)
print(model3_score)




