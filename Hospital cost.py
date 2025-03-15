#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Project: Healthcare Cost Prediction

# ## Project objective:
# ### Analyze factors influencing hospital billing and predict healthcare costs for patients.
# ### Help patients and insurers optimize expenses

# In[350]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[351]:


df=pd.read_csv(r'C:\Users\USER\Documents\untouced project\medical cost personal - Copy\insurance.csv')


# ## Column Descriptions
# ### age:The age of the patient in years.
# ### The gender of the patient (male or female).
# ### Body Mass Index (BMI): a measure of body fat based on height and weight.
# ### children:The number of dependents the patient has.
# ### smoke:Indicates whether the patient smokes (yes or no).
# ### region:The geographical area where the patient lives (e.g., northeast, northwest, southeast, southwest).
# ### charges:The medical insurance cost (billing amount) for the patient.

# In[352]:


df


# # Data Preprocessing

# In[353]:


df.isnull().sum()


# In[354]:


df.duplicated().sum()


# In[355]:


## drop duplicates
dup=df.drop_duplicates(inplace=True)


# In[356]:


df.info()


# In[357]:


df.shape


# # Factors influencing hospital bills

# In[ ]:





# ## 1. Does age influence hospital cost

# ## H0:Age influence hospiatl hospital cost
# ## H1:Age doesnt influence hopital cost

# In[358]:


from scipy .stats import pearsonr


# In[359]:


p,stats=pearsonr(df['age'],df['charges'])

if p >0.05:
    print('yes age influence charges')
else:
    print('no')

print(p,stats)
    


# ### Using the hypothesis technique, age of a person increases the hospital cost

# In[360]:


plt.figure(figsize=(10,10))
sns.barplot(x=df['age'],y=df['charges'],ci=None)
plt.xticks(rotation='vertical')
plt.title('age vs charges')
plt.savefig('age vs charge.jpg',dpi=300, bbox_inches="tight")


# ## from this chart, the hight the age of a person the more he or she pays for hospital bills

# In[ ]:





# ## 2.How does sex of a person impact medical costs?

# In[361]:


from scipy.stats import ttest_ind


# In[362]:


female=df[df['sex']=='female']['charges']
male=df[df['sex']=='male']['charges']


# In[363]:


p,stats=ttest_ind(female,male)
if p>0.05:
    print('yes')
else:
    print('no')

print(p,stats)


# ### statistically, the gender of a person doesnt affect hospital cost

# In[364]:


# Plot histogram of medical charges, colored by sex
plt.figure(figsize=(8,6))
sns.histplot(df, x='charges', hue='sex', kde=True, bins=30)
plt.title("Distribution of Medical Charges by Sex")
plt.xlabel("Medical Charges")
plt.ylabel("Count")
plt.savefig('sex vs charges.jpg',dpi=300,bbox_inches='tight')


# ## from this chart , you can see that charges are not affeccted by Gender

# In[ ]:





# ## 3.Is BMI correlated with medical expenses?
# ## BMI is a numerical value calculated from a person's weight and height. It is used to categorize individuals into different weight classes and assess potential health risks.

# In[365]:


from scipy.stats import spearmanr


# In[366]:


p,stats=spearmanr(df['bmi'],df['charges'])
if p>0.05:
    print('yes')
else:
    print('no')

print(p,stats)


# ### using hypothesis technique the weight of a person increases hospital cost

# In[367]:


plt.figure(figsize=(10,10))
sns.lineplot(x=df['bmi'],y=df['charges'])
plt.xticks(rotation='vertical')
plt.title('BMI and charges')
plt.savefig('BMI vs charge.jpg',dpi=300, bbox_inches="tight")


# ### from this chart above, you can see that  the lower the bmi, the lower the cost, the higher the bmi, the higher the cost reason i think is bmi i used to categorize obesity, so the hight your bmi the more chance of being obesed and also has high health risk

# In[ ]:





# ### 4. the number of children a person has influences cost?

# In[368]:


p,stats=ttest_ind(df['children'],df['charges'])
if p>0.05:
    print('yes')
else:
    print('no')
    print(p,stats)


# ## Findings: the number of children you have doesnt affect insurance cost

# In[369]:


children=df.groupby('children')['charges'].sum()


# In[397]:


sns.barplot(children,palette='magma')
plt.savefig('children vs charge.jpg',dpi=300, bbox_inches="tight")


# ## Also from the chart, it doesnt show children affecting hospital cost

# ## 5.How does smoking impact medical costs?

# In[371]:


yes_smoke=df[df['smoker']=='yes']['charges']
no_smoke=df[df['smoker']=='no']['charges']


# In[372]:


p,stats=ttest_ind(yes_smoke,no_smoke)
if p>0.05:
    print('yes')
else:
    print('no')
print(p,stats)


# ## finding:  smoking habit  affect the hospital or insurance cost, i think reason is because the person has a high health risk

# In[373]:


sns.histplot(df,x='charges',hue='smoker')
plt.savefig('smoker vs charge.jpg',dpi=300, bbox_inches="tight")


# ## same with the chart

# In[ ]:





# ## 6.Which region has the highest medical costs?

# In[374]:


region=df.groupby('region')['charges'].sum()


# In[375]:


sns.barplot(region,palette='dark')
plt.title('regions')
plt.savefig('region vs charge.jpg',dpi=300, bbox_inches="tight")


# ## findings: the southeast regions pays more hospital or incurance cost from the chart above

# In[ ]:





# In[376]:


df2=df.copy()
# duplicate columns


# In[377]:


from sklearn .preprocessing import LabelEncoder


# In[378]:


le=LabelEncoder()


# In[379]:


df2[['sex','smoker','region']]=df2[['sex','smoker','region']].apply(le.fit_transform)


# # building a cost predictive model

# In[380]:


df_corr=df2.corr()


# In[381]:


plt.figure(figsize=(10,10))
sns.heatmap(df_corr,annot=True)
plt.title('correlations among columns')


# # feature selection

# In[382]:


x=df2[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y=df2['charges']


# In[383]:


from sklearn .preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[384]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[385]:


scaler=StandardScaler()


# In[386]:


x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[387]:


ln=RandomForestRegressor(n_estimators=300)
ln.fit(x_train,y_train)


# In[388]:


ln.score(x_train,y_train)


# In[389]:


ypred=ln.predict(x_test)


# In[390]:


from sklearn.metrics import r2_score,mean_squared_error


# In[391]:


r2_score(y_test,ypred)


# In[392]:


print(mean_squared_error(y_test,ypred))


# In[393]:


tt=pd.DataFrame({'actual':y_test,
                 'predicted':ypred})


# In[394]:


tt


# # feature importances

# In[395]:


feature=ln.feature_importances_
values=['age', 'sex', 'bmi', 'children', 'smoker', 'region']


# In[398]:


sns.barplot(x=values,y=feature,palette='coolwarm')
plt.savefig('feature.jpg',dpi=300,bbox_inches='tight')


# In[399]:


import joblib


# In[400]:


model=joblib.dump(ln,'hospital_cost_model.joblib')


# In[ ]:




