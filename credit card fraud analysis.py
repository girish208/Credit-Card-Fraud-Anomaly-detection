
# coding: utf-8

# In[1]:


import os


# In[2]:


os.chdir('C:/Users/ASUS/Desktop/Data Science/credit card fraud')


# In[3]:


import pandas as pd
import numpy as np
data=pd.read_csv('creditcard.csv')


# In[4]:


data.head()


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.shape


# In[8]:


percentage_fraud=data['Class'].value_counts()/len(data)*100


# In[9]:


percentage_fraud


# In[10]:


print('Transactions without fraud in dataset are {}%'.format(round(percentage_fraud[0],2)))
print('Transactions with fraud detected in dataset are {}%'.format(round(percentage_fraud[1],2)))


# In[11]:


import matplotlib.pyplot as plt
import matplotlib as mlt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"]=(16,9)
plt.style.use('fivethirtyeight')


# In[12]:



sns.countplot('Class',data=data,palette='RdBu')
plt.xticks([0,1],('No Fraud','Fraud'))
plt.show()


# In[13]:


Fraud=data[data['Class']==1]
Normal=data[data['Class']==0]


# In[14]:


Fraud.shape


# In[15]:


Normal.shape


# In[16]:


# Analyzing Time and Amount features
fig,(ax1,ax2)=plt.subplots(2,1,sharex=True)
amount_val=data['Amount'].values
Time_val=data['Time'].values

bins=50

ax1.hist(Fraud.Time,bins=bins)
ax1.set_title('Time Analysis of Fraud Transactions')

ax2.hist(Normal.Time,bins=bins)
ax2.set_title('Time Analysis of Normal Distribution')

plt.xlabel('Time(in seconds)')
plt.ylabel('Number of transaction')

plt.show()


# In[17]:


# Analyse fraud and normal transacions distribution through summary statistics
print('Fraud time analysis','\n',Fraud.Time.describe(),'\n','Normal Time Analysis','/n',Normal.Time.describe())


# It could be analysed that number of transactions of fraud is comparitively low as compared to that of normal transactions, however anomaly is that these fraud transactions are more evenly distributed than normal transactions. Moreover, these transactions are recoreded in period of 48 hours and it can be observed that maximum of fraudelent transactions are happening in Off-Peak hours as compared to normal transactions.

# Now, Let's analyse amount distbursed during fraudelent transactions in comparison to normal transaction

# In[18]:


print('Amount info of fraud transaction','\n',Fraud.Amount.describe())
print('Amount infor of normal transaction \n',Normal.Amount.describe())


# Maximum amount of money transacted during fraudalant transaction is $ 2125.87 as compared to $ 25691.16 while minimum is 0 which means that , fraudsters failed to transact money during this time-this may be because of multiple reasons. 75% of fraudelent transactions has amount withdrawan less than $ 105.

# In[19]:


fig,(ax1,ax2)=plt.subplots(2,1,sharex=True)

bins=10
ax1.hist(Fraud.Amount,bins=bins)
ax1.set_title('Amount transacted in Fraudalent transactions')

ax2.hist(Normal.Amount,bins=bins)
ax2.set_title('Amount transacted in Normal transaction')

plt.xlabel('Amount')
plt.ylabel('Transactions')
plt.yscale('log')

plt.show()


# Another major observation is that , Amount and Time data are unevenly distributed and should be scaled. Secondly, in order to make a better model, we should trim down the data so that we have 1:1 of Fraud and Normal transactions data.

# Since, data is unevenly distributed, we will scale the data and take subsample to avoid overfitting and wrong correlation.

# In[20]:


# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))
df=data.copy()

data.drop(['Time','Amount'], axis=1, inplace=True)


# In[21]:


data.head()


# # Original Split

# In[22]:


from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,StratifiedKFold,KFold

print('No Frauds', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')
print('Frauds', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')


# In[23]:


y=data['Class']
X=data.drop('Class',axis=1)
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]


# In[24]:


# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)
print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))


# ## Random Undersampling
# This we are doing to make 50:50 split for fraud and non-fraud transactions.

# In[25]:


#Shuffle the data first to random selection of samples.

data=data.sample(frac=1)

fraud=data[data['Class']==1]
normal=data[data['Class']==0][:492]

normal_distributed_df = pd.concat([fraud, normal])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()


# # Exploring the shuffled and re-framed dataset
# 
# Now, since we have redistributed dataset to ensure that we have equal number of fraud and non_fraud data. We will now analyze this dataset to better understanding fraud transactions. This will help in building models to detect anomalies.

# In[26]:


print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))



sns.countplot('Class', data=new_df, palette='RdBu')
plt.title('Equally Distributed Classes', fontsize=14)
plt.xticks([0,1],('Non_fraud','Fraud'))
plt.show()


# In[27]:


fig,(ax1,ax2)=plt.subplots(2,1,figsize=(24,36))
corr_df=df.corr()
sns.heatmap(corr_df,cmap='hot',ax=ax1)
ax1.set_title('Imbalanced dataset correlation')

corr_mat=new_df.corr()

sns.heatmap(corr_mat,cmap='hot',ax=ax2)
ax2.set_title('New truncated dataset correlation')
fig.tight_layout()
plt.show()


# # Summary and Explanation:
# * Negative Correlations: V17, V14, V12 and V10 are negatively correlated. Notice how the lower these values are, the more likely the end result will be a fraud transaction.
# * Positive Correlations: V2, V4, V11, and V19 are positively correlated. Notice how the higher these values are, the more likely the end result will be a fraud transaction.

# In[28]:


print(new_df.corr()['Class'].drop('Class').sort_values(ascending=False).head())
print(new_df.corr()['Class'].drop('Class').sort_values(ascending=False).tail())


# In[29]:


f, axes = plt.subplots(ncols=4, figsize=(20,4),sharex=True)

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Class", y="V17", data=new_df, palette='YlGn', ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=new_df, palette='YlGn', ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V12", data=new_df, palette='YlGn', ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V10", data=new_df, palette='YlGn', ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

plt.xticks([0,1],('Non Fraud','Fraud'))
plt.show()


# In[30]:


f, axes = plt.subplots(ncols=4, figsize=(20,4),sharex=True)

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="Class", y="V11", data=new_df, palette='RdBu', ax=axes[0])
axes[0].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=new_df, palette='RdBu', ax=axes[1])
axes[1].set_title('V4 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V2", data=new_df, palette='RdBu', ax=axes[2])
axes[2].set_title('V2 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V19", data=new_df, palette='RdBu', ax=axes[3])
axes[3].set_title('V19 vs Class Positive Correlation')

plt.xticks([0,1],('Non Fraud','Fraud'))
plt.show()


# # Anomaly Algorithm
# Credit card frauds are anomaly in normal credit card transactions. Its behaviour and is different from other transctions and can be assessed by variables measurement in above plots. These anomalies should by system and could be predicted . We will now work to build up a model to detect such anomalous behavior.

# ## Outlier detection
# Statistics normal principle of removing outlier is to consider data within +/- 1.5*InterQuartile range. We will now make it happen  for variables in our data.

# In[32]:


from scipy.stats import norm


# In[33]:


f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)


v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()


# The above distribution plot clearly indicates that V14 is more close to Gaussian distribution than V12 and V10

# In[40]:


f, (ax1, ax2, ax3,ax4) = plt.subplots(1,4, figsize=(20, 6))

v11_fraud_dist = new_df['V11'].loc[new_df['Class'] == 1].values
sns.distplot(v11_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V11 Distribution \n (Fraud Transactions)', fontsize=14)

v4_fraud_dist = new_df['V4'].loc[new_df['Class'] == 1].values
sns.distplot(v4_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V4 Distribution \n (Fraud Transactions)', fontsize=14)


v2_fraud_dist = new_df['V2'].loc[new_df['Class'] == 1].values
sns.distplot(v2_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V2 Distribution \n (Fraud Transactions)', fontsize=14)

v9_fraud_dist = new_df['V9'].loc[new_df['Class'] == 1].values
sns.distplot(v9_fraud_dist,ax=ax4, fit=norm, color='#C5B3F9')
ax4.set_title('V9 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()


# V11.V4 is more close to Gaussian distribution
# 
# Thus, V11, V4 and V14 are more close to Gaussian distribution. In other variables, there are outliers available and thus needed to be corrected to make distribution statistically significant.

# In[41]:


q25,q75=25,75

# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))

v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V14 outliers:{}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
print('----' * 44)

# -----> V12 removing outliers from fraud transactions
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))
outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print('V12 outliers: {}'.format(outliers))
print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))
new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))
print('----' * 44)


# Removing outliers V10 Feature
v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))
outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print('V10 outliers: {}'.format(outliers))
print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))
new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))


# In[42]:


f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))

colors = ['#B3F9C5', '#f9c5b3']
# Boxplots with outliers removed
# Feature V14
sns.boxplot(x="Class", y="V14", data=new_df,ax=ax1, palette=colors)
ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)
ax1.annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

# Feature 12
sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2, palette=colors)
ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)
ax2.annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

# Feature V10
sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3, palette=colors)
ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)
ax3.annotate('Fewer extreme \n outliers', xy=(0.95, -16.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)


# ## Clustering-Dimensionality reduction
# 
