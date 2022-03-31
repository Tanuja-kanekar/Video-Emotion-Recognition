#!/usr/bin/env python
# coding: utf-8

# ## Speech Emotion Recognition

# ### Importing the libraries

# In[1]:


#loading the necessary libraries
import librosa 
import pandas as pd
import numpy as np
import glob
import os
import sys


# In[2]:


#to get the path of current working directory
path = os.getcwd()
print(path)


# In[3]:


#saving the location of our data folder
Data1="/home/thanuja/Downloads/output/train/"
Data2="/home/thanuja/Downloads/output/test/"
Data3="/home/thanuja/Downloads/output/val/"


# In[4]:



dir_train=os.listdir(Data1)
dir_test=os.listdir(Data2)

print(dir_test)


# ### Train Data 

# In[5]:



emotion=[]
path=[]

for sub_dir1 in dir_train:
    filename=os.listdir(Data1+sub_dir1)
    for files in filename:
        part=files.split('.')[0].split('-')
        emotion.append(int(part[2]))
        path.append(Data1+sub_dir1+'/'+files)
                
df=pd.DataFrame(emotion,columns=['Emotions'])
pathdf=pd.DataFrame(path,columns=['Path'])
train_data=pd.concat([pathdf,df],axis=1)


# In[6]:


train_data.head()


# In[7]:


train_data.shape


# In[8]:


train_data.nunique()


# In[9]:


train_data=train_data.replace({2:'calm',3:'happy',4:'sad',5:'angry'
                ,7:'disgust',8:'surprise'})


# In[10]:


train_data


# In[11]:


train_data['Emotions'].value_counts()


# ### Test Data

# In[12]:


emotion1=[]
path1=[]

for sub_dir2 in dir_test:
    filename=os.listdir(Data2+sub_dir2)
    for files in filename:
        part=files.split('.')[0].split('-')
        emotion1.append(int(part[2]))
        path1.append(Data2+sub_dir2+'/'+files)
                


# In[13]:


df1=pd.DataFrame(emotion1,columns=['Emotions'])
pathdf1=pd.DataFrame(path1,columns=['Path'])
test_data=pd.concat([pathdf1,df1],axis=1)


# In[14]:


test_data.head()


# In[15]:


test_data=test_data.replace({2:'calm',3:'happy',4:'sad',5:'angry'
                ,7:'disgust',8:'surprise'})


# In[16]:


test_data.nunique()


# ### Train and Test data

# In[17]:


Data=pd.concat([train_data,test_data],ignore_index=True)
Data[120:125]


# ### Shuffling the data

# In[18]:


Data = Data.reindex(np.random.permutation(Data.index))
Data


# ### Saving the shuffled data

# In[19]:


Data.to_csv("SER1.csv",index=False)


# In[20]:


speech_data=pd.read_csv("SER1.csv")
speech_data


# ### Feature Extraction

# In[21]:


Feature_data = pd.DataFrame(columns=['Features'])

counter = 0
for index, path in enumerate(speech_data.Path):
    X, sample_rate = librosa.load(path,res_type='kaiser_fast')
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=40).T, axis=0)
    Feature_data.loc[counter]=[mfccs]
    counter=counter+1


# In[22]:


Feature_data.head()


# In[23]:


Feature_data=pd.concat([pd.DataFrame(Feature_data['Features'].values.tolist()),speech_data.Emotions],axis=1)
Feature_data


# ### Input data

# In[24]:


X_data=Feature_data.drop(['Emotions'],axis=1)
X_data


# ### Target Data

# In[25]:


Y_data=Feature_data.Emotions
Y_data.head()


# ### Train and Test split

# In[26]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_data,Y_data,test_size=0.2,random_state=40)


# In[27]:


print((x_train.shape,x_test.shape,y_train.shape,y_test.shape))


# ## Model Building

# #### Support Vector Machine for non standardized data

# #### SVC rbf

# In[64]:


from sklearn.svm import SVC
svc_model=SVC(C=1,gamma=0.0001,kernel='rbf').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[65]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# #### SVC Linear

# In[62]:


from sklearn.svm import SVC
svc_model=SVC(C=10,gamma=0.001,kernel='linear').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[63]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# #### Random Forest Classifier

# In[32]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators= 300,max_depth=5).fit(x_train,y_train)


# In[33]:


print('accuracy: {}'.format(model.score(x_test,y_test)))


# In[34]:


train_acc = float(model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# ### XGBoost Classifier

# In[35]:


from xgboost import XGBClassifier


# In[36]:


xg_model = XGBClassifier(learning_rate=3.5,n_estimators=300,max_depth=4).fit(x_train, y_train)


# In[37]:


print('accuracy: {}'.format(xg_model.score(x_test,y_test)))


# In[38]:


train_acc = float(xg_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(xg_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# In[66]:


final_model = svc_model=SVC(C=1,gamma=0.0001,kernel='rbf').fit(X_data,Y_data)


# In[67]:


import pickle

pickle.dump(final_model,open('final1.pkl','wb'))


# In[68]:


model = pickle.load(open('final1.pkl','rb'))


# In[69]:


file='/home/thanuja/Downloads/audio/val/happy/03-01-03-01-01-01-04.wav'


# In[70]:


Feature_data_valid1 = pd.DataFrame(columns=['Features_valid'])

X1, sample_rate1= librosa.load(file,res_type='kaiser_fast')
mfccs=np.mean(librosa.feature.mfcc(y=X1, sr=sample_rate1,n_mfcc=40).T, axis=0)
Feature_data_valid1.loc[0]=[mfccs]


# In[71]:


Feature_data_valid1


# In[72]:


Feature_data1=pd.DataFrame(Feature_data_valid1['Features_valid'].values.tolist())
Feature_data1.head()


# In[73]:


ser = model.predict(Feature_data1)


# In[74]:


print(ser)


# In[ ]:





# In[ ]:




