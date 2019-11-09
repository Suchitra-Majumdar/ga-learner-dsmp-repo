# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df=pd.read_csv(path)
p_a=((df[df['fico']>700].count())/len(df)).loc['fico']
print(p_a)
p_b=((df[df['purpose']=='debt_consolidation'].count())/len(df)).loc['purpose']
print(p_b)
df1=df[df['purpose']=='debt_consolidation']
df1.head()
p_a_b=((df[(df['purpose']=='debt_consolidation') & (df['fico']>700)]).count()/(df[df['fico']>700].count())).loc['purpose']
print(p_a_b)
result=p_a==p_a_b
print(result)
print('Hence independency is ', result)


# code ends here


# --------------
# code starts here
prob_lp=((df[df['paid.back.loan']=='Yes'].count())/len(df)).loc['paid.back.loan']
print(prob_lp)
prob_cs=((df[df['credit.policy']=='Yes'].count())/len(df)).loc['credit.policy']
print(prob_cs)
new_df=df[df['paid.back.loan']=='Yes']
new_df.head()
prob_pd_cs=(((df[(df['paid.back.loan']=='Yes') & (df['credit.policy']=='Yes')]).count())/len(new_df)).loc['paid.back.loan']
print(prob_pd_cs)
bayes=(prob_pd_cs*prob_lp)/prob_cs
print(bayes)




# code ends here


# --------------
# code starts here

df1=df[df['paid.back.loan']=='No']
print(df1.head())
pl_count=df1['purpose'].value_counts()
print(pl_count)
pl_count.plot(kind='bar')

# code ends here


# --------------
# code starts here
inst_median=df['installment'].median()
print(inst_median)
inst_mean=df['installment'].mean()
print(inst_mean)
df.hist(column='installment',bins=10)
df.hist(column='log.annual.inc', bins=10)



# code ends here


