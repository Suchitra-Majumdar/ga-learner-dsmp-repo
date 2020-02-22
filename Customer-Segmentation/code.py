# --------------
import numpy as np 
import pandas as pd 
offers=pd.read_excel(path,sheet_name=0)
transactions=pd.read_excel(path,sheet_name=1)
transactions['n']=1
df=offers+transactions
# df=pd.concat([offers,transactions],axis=1,join='inner')
print(offers.shape)
print(transactions.shape)
print(df.shape)
df.head()


# --------------
# Code starts here

# create pivot table
matrix=df.pivot_table(index='Customer Last Name', columns='Offer #',values='n')

# replace missing values with 0
matrix.fillna(0, inplace=True)

# reindex pivot table
matrix.reset_index(inplace=True)

# display first 5 rows
matrix.head()

# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object
cluster=KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10,random_state=0)

# create 'cluster' column
matrix['cluster']=cluster.fit_predict(matrix[matrix.columns[1:]])
matrix.head()
# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca=PCA(n_components=2, random_state=0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
matrix['x']=pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y']=pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
# dataframe to visualize clusters by customer names
clusters=matrix.iloc[:,[0, 33, 34, 35]]

# visualize clusters
plt.scatter(x=clusters['x'], y=clusters['y'], c=clusters['cluster'])


# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'
print(clusters.head(2))
print(transactions.head(2))
data=pd.merge(clusters,transactions)

# merge `data` and `offers`
data=pd.merge(offers,data)
# initialzie empty dictionary
champagne={}

# iterate over every cluster
for i in range(0,5):
    # observation falls in that cluster
    new_df=data[data['cluster']==i]
    # sort cluster according to type of 'Varietal'
    counts=new_df['Varietal'].value_counts(ascending=False)
    # check if 'Champagne' is ordered mostly
    if counts.index[0]=='Champagne':
        # add it to 'champagne'
        champagne.update({i:counts[0]})

# get cluster with maximum orders of 'Champagne' 
cluster_champagne=max(champagne,key=champagne.get)

# print out cluster number
print(cluster_champagne)



# --------------
# Code starts here

# empty dictionary
discount={}

# iterate over cluster numbers
for i in range(0,5):
    # dataframe for every cluster
    new_df=data[data['cluster']==i]
    # average discount for cluster
    counts=sum(new_df['Discount (%)'])/len(new_df)
    # adding cluster number as key and average discount as value 
    discount.update({i:counts})

# cluster with maximum average discount
cluster_discount=max(discount,key=discount.get)
print(cluster_discount)

# Code ends here


