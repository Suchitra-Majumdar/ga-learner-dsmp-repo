# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
data_file = path
data = np.genfromtxt(data_file, delimiter =",", skip_header = 1)
census = np.concatenate([new_record,data])
print("\nCensus: \n\n", census)



# --------------
#Code starts here
import numpy as np
age = np.array(census[:,0])
print(age)
max_age = age.max()
min_age = age.min()
age_mean = age.mean()
age_std = age.std()
print(max_age)
print(min_age)
print(age_mean)




# --------------
#Code starts here
import numpy as np
race = np.array(census[:,2])
race
race_0 = census[census[:,2]== 0]
race_1 = census[census[:,2]== 1]
race_2 = census[census[:,2]== 2]
race_3 = census[census[:,2]== 3]
race_4 = census[census[:,2]== 4]
print(race_0)
len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)
if len_0 < len_1 < len_2< len_3<len_4:
    minority_race = 0
elif len_1 < len_0 < len_2< len_3<len_4:
    minority_race = 1
elif len_2<len_0<len_3<len_4:
    minority_race = 2
elif len_3<len_0<len_1<len_2<len_4:
    minority_race=3
else:
    minority_race=4

print(minority_race)




# --------------
#Code starts here
import numpy as np 
senior_citizens = census[age>60]
#print(senior_citizens)
working_hours_sum=sum(senior_citizens[:,6])
print(working_hours_sum)
senior_citizens_len = len(senior_citizens)
print(senior_citizens_len)
avg_working_hours = working_hours_sum/senior_citizens_len
print(avg_working_hours)



# --------------
#Code starts here
high = census[census[:,1]>10]
low = census[census[:,1]<=10]
print(high)
avg_pay_high = high[:,7].mean()
avg_pay_low = low[:,7].mean()
print(avg_pay_high)
print(avg_pay_low)
if avg_pay_high > avg_pay_low:
    print("There is truth in better education leads to better pay")
else:
    print("There is no truth in better education leads to better pay")




