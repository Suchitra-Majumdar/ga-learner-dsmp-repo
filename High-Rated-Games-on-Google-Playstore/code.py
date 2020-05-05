# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
data =data[data['Rating'] <=5]
print(data.head())
print(data.shape)
plt.hist(data['Rating'])

#Code ends here


# --------------
# code starts here



total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())
print(type(total_null))
print(type(percent_null))
missing_data = pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'])
print(missing_data)
data = data.dropna()
total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())
missing_data_1 = pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'])
print(missing_data_1)



# code ends here


# --------------

#Code starts here


sns.catplot(x='Category',y='Rating',data=data,kind='box',height=10)
plt.xticks(rotation=90)
plt.title('Rating vs Category [BoxPlot]')

#Code ends here


# --------------
# #Importing header files
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# import re
# #Code starts here
# print(data['Installs'].value_counts())
# # data['Installs'].apply(re.sub(r'[,\+]', " ",data['Installs']))


# # data['Installs'] = data['Installs'].replace(",", "") 
# # data['Installs'] = data['Installs'].str.apply(lambda x: re.sub(r"\+"," ",(x))
# # data['Installs'] = data['Installs'].str.apply(lambda x: re.sub(r",","",(x))
# data['Installs'] = data['Installs'].str.replace(',','')
# data['Installs'] = data['Installs'].str.replace('+','')
# print(data["Installs"].head())
# le = LabelEncoder()
# data['Installs'] = data['Installs'].apply(int)
# sns.regplot(x='Installs',y='Rating',data=data)
# plt.title('Rating vs Installs [RegPlot]')



#Code ends here
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'].head())

data['Installs'] = data['Installs'].str.replace(',','')
data['Installs'] = data['Installs'].str.replace('+','')
data['Installs'] = data['Installs'].apply(int)
print(data['Installs'].head())

le = LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])


graph = sns.regplot(x="Installs", y="Rating" , data=data)
graph.set_title('Rating vs Installs [RegPlot]')
#Code ends here



# --------------
#Code starts here
import re
import seaborn as sns 
print(data['Price'].value_counts())
# data['Price'] = data['Price'].str.replace("$","")

data['Price'] = data['Price'].apply(lambda x: re.sub(r'[$]',"",str(x)))
print(data['Price'].head())
data['Price'] = data['Price'].apply(float)
sns.regplot(x="Price", y="Rating", data=data)
plt.title('Rating vs Price [RegPlot]')

#Code ends here


# --------------

#Code starts here
print(data['Genres'].unique())
data['Genres'] = data['Genres'].apply(lambda x: x.split(";")[0])
gr_mean = data.groupby('Genres',as_index=False)['Rating'].mean()
print(gr_mean.describe())
gr_mean = gr_mean.sort_values(by='Rating')
print(gr_mean)
# print(gr_mean)



#Code ends here


# --------------

#Code starts here
import pandas as pd 
import seaborn as sns
data['Last Updated'] =pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()
data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days
sns.regplot(x="Last Updated Days", y="Rating", data=data)
plt.title('Rating vs Last Updated [RegPlot]')



#Code ends here


