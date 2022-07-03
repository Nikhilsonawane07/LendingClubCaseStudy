#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load the libraries
import pandas as pd #To work with dataset
import numpy as np #Math library
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt #to plot some parameters in seaborn
import warnings #To avoid any warnings
warnings.filterwarnings(action="ignore")
import datetime as dt # To work with Time date data set


# # Importing Data

# In[2]:


df = pd.read_csv('loan.csv')


# # 1) Data Understanding

# In[3]:


#Looking the data
df.head()


# In[4]:


# Size of the Data
df.shape


# ### Deleting columns having null values greater than 80%

# In[5]:


# Checking columns having more than 80% null values
df.columns[100*df.isnull().mean() > 80]


# In[6]:


# Data imputation doesn't make any sense in this column as they have very high null value count 
# So its better to remove this columns
Null_cols = ['mths_since_last_record', 'next_pymnt_d', 'mths_since_last_major_derog',
       'annual_inc_joint', 'dti_joint', 'verification_status_joint',
       'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_6m',
       'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
       'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
       'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
       'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
       'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
       'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc',
       'mths_since_recent_bc_dlq', 'mths_since_recent_inq',
       'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd',
       'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl',
       'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0',
       'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
       'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
       'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
       'total_il_high_credit_limit']
df = df.drop(Null_cols, axis = 1)
df.shape


# ### The following customer behavior variables are not available at the time of loan application, thus they cannot be used as predictors for credit approval.
# ### This variables are listed below and they can be removed
# 

# In[7]:


cust_behav_cols = ['delinq_2yrs','earliest_cr_line','inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc',
                  'out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee',
                  'recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','last_credit_pull_d','application_type']
df = df.drop(cust_behav_cols, axis = 1)
df.shape


# In[8]:


# Looking at the data 
df.info()


# In[9]:


# Checking value count for column : Pymnt_plan
df['pymnt_plan'].value_counts()


# In[10]:


# Here, all values are 'n', hence we can drop this column
df = df.drop('pymnt_plan', axis = 1)
df.shape


# In[11]:


# URL column can also be dropped, we already have memebr ID, so we can refer particular user with the help of ID, so need for url column
df = df.drop('url', axis = 1)
df.shape


# # Data Cleaning and Imputation

# In[12]:


# Chcking data set
df.info()


# In[13]:


# Checking Null value count
100*df.isnull().mean()


# ### Converting 'term' column

# In[48]:


# checking value count 
df['term'].value_counts()


# In[49]:


# only two values, no need for treatment


# ### int_rate column treatment

# In[50]:


# checking value count
df['int_rate'].value_counts()


# In[53]:


# removing percentage symbol and converting it into float datatype
df['int_rate'] = df['int_rate'].apply(lambda x: float(x[:-1]))

df['int_rate'].value_counts()


# In[55]:


df['int_rate'].describe()


# In[ ]:





# ### int_rate values are float numbers, so it is better to bin them into categories, here categories made are 5-10, 10-15, 15-20, 20-25
# 
# 

# In[60]:


bins = [0,5,10,15,20,25]
df['bin_int_rate'] = pd.cut(df['int_rate'], bins)
df['bin_int_rate'].value_counts()


# In[14]:


# In emp_title column, many users havent filled any information regarding there employment title, so it can be filled with title "Other"
df['emp_title'] = df['emp_title'].fillna('Other')

# Checking null value count
df['emp_title'].isnull().mean()


# ### Installment column

# In[62]:


# checking installment column
df['installment'].describe()


# ### verification_status column

# In[63]:


df['verification_status'].value_counts()


# ### issue_d column

# In[64]:


df['issue_d'].value_counts()


# In[74]:


# creating new column with year from issue date
df['issue_date2'] = pd.to_datetime(df['issue_d'],  errors = 'coerce')
df.head()


# ### loan_status column

# In[75]:


df['loan_status'].value_counts()


# #### The ones marked 'current' are neither fully paid not defaulted. Hence theis rows are removed

# In[78]:


df = df[~(df['loan_status']=='Current')]
df.shape


# In[80]:


df['desc'].nunique


# ### purpose column

# In[81]:


df['purpose'].value_counts()


# In[82]:


df['purpose'].info()


# ### Zip code

# In[83]:


df['zip_code'].value_counts()


# In[ ]:


# cant infer much info from zip code, can be dropped
# will be decided later


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Treatment for emp_length column
# #### Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 

# In[27]:


# Checking null value count in 'emp_length'
df['emp_length'].isnull().sum()


# In[28]:


# Replacing null values with mode value
df['emp_length'] = df['emp_length'].fillna(df['emp_length'].mode()[0])

# checking null value count
df['emp_length'].isnull().mean()


# In[29]:


# emp_length dataset value count
df['emp_length'].value_counts()


# In[30]:


# checking data type 
df['emp_length'].dtype


# In[31]:


# replacing less than 1 year employment length by 0 and more than 10 years by 10
df['emp_length'] = df['emp_length'].apply(lambda x: 0 if x=='< 1 year' else(10 if x=='10+ years' else int(x[0])))
df['emp_length'].value_counts()


# ### 'desc' column Treatment

# In[32]:


# checking value count in 'desc' column
df['desc'].value_counts()


# In[34]:


df['title'].value_counts()


# #### Loan description provided by the borrower is random, this column would not be used for data analysis so missing values present can be replaced with 'other'

# In[33]:


df['desc'] = df['desc'].fillna('other')

# checking null values
df['desc'].isnull().mean()


# ### 'title' column treatment

# In[35]:


# checking value count
df['title'].value_counts()


# In[37]:


# missing values are filled with 'other'

df['title'] = df['title'].fillna('other')

# checking null values
df['title'].isnull().mean()


# In[38]:


# Here we can see that, many duplicate values are there with different lowercase and uppercase, so we will convert all in lower case
df['title'] = df['title'].apply(lambda x: x.lower())

# checking value count again
df['title'].value_counts()


# ### mnths_since_last_deliq column treatment

# In[ ]:





# In[ ]:





# ### no of collection in 12 months column

# In[ ]:





# ### chargeoff_within_12_mths

# In[ ]:





# ### pub_rec_bankruptcies Column Treatment

# In[40]:


# checking value count
df['pub_rec_bankruptcies'].value_counts()


# In[44]:


# filling missing values with '0' bankruptcies
df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(0)

df['pub_rec_bankruptcies'].isnull().mean()


# In[45]:


df['pub_rec_bankruptcies'].value_counts()


# ### tax_liens column Treatment

# In[46]:


df['tax_liens'].value_counts()


# In[47]:


# here most of the values are '0',hence this column wont be utilise in analysis, hence it is dropped

df = df.drop('tax_liens', axis = 1)

df.shape


# In[84]:


df.columns


# In[ ]:




