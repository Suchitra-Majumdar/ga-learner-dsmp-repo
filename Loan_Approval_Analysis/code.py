# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 

# code starts here
bank=pd.read_csv(path)
categorical_var=bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var = bank.select_dtypes(include ='number')
print(numerical_var)






# code ends here


# --------------
# code starts here
banks=bank.drop('Loan_ID',axis=1)
print(banks.isnull().sum())
bank_mode=banks.mode()
banks=banks.fillna(banks.mode())
values={'Gender': 'Male','Married':'Yes','Dependents':0,'Education':'Graduate','Self_Employed':'No','ApplicantIncome':2500,'CoapplicantIncome':0.0,'LoanAmount':120.0,'Loan_Amount_Term':360.0,'Credit_History':1.0,'Property_Area':'Semiurban','Loan_Status':'Y'}

banks = banks.fillna(value=values)
banks.isnull().sum()


#code ends here


# --------------
# Code starts here
import pandas as pd 
import numpy as np 
avg_loan_amount=banks.pivot_table(index=['Gender','Married','Self_Employed'],values='LoanAmount',aggfunc=np.mean)
print(avg_loan_amount)


# code ends here



# --------------
# code starts here
import pandas as pd 
import numpy as np 



loan_approved_se = banks[(banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')].count()
Loan_Status=614
percentage_se =(loan_approved_se['Self_Employed']/Loan_Status)*100
print(percentage_se)
loan_approved_nse=banks[(banks['Self_Employed']=='No')& (banks['Loan_Status']=='Y')].count()
loan_approved_nse=loan_approved_nse['Self_Employed']
loan_approved_nse
percentage_nse=(loan_approved_nse/Loan_Status)*100
print(percentage_nse)
# code ends here


# --------------
# code starts here
loan_term=banks['Loan_Amount_Term'].apply(lambda x:x/12)
print(loan_term)
big_loan_term=loan_term[loan_term>=25].count()
print(big_loan_term)


# code ends here


# --------------
# code starts here
loan_groupby= banks.groupby('Loan_Status')
loan_groupby=loan_groupby['ApplicantIncome','Credit_History']
mean_values=loan_groupby.mean()


# code ends here


