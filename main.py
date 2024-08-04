#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
df = pd.read_csv('predictive_maintenance.csv')


# In[28]:


df.head()


# In[29]:


columns_to_drop = ['UDI' , 'Product ID']
df.drop(columns=columns_to_drop , axis = 1 , inplace=True)


# In[30]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
# columnns_to_encode = ['Type', 'Failure Type']
df['Type'] = encoder.fit_transform(df['Type'])

le = LabelEncoder()
df['Failure Type'] = le.fit_transform(df['Failure Type'])
import joblib
joblib.dump(encoder , 'type_encoder.pkl')
joblib.dump(le , 'failure_type.pkl')


# In[31]:


df.head()


# In[32]:


columns_to_normalize = ['Air temperature [K]','Process temperature [K]' , 'Rotational speed [rpm]','Torque [Nm]' , 'Tool wear [min]']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])


# In[33]:


df.head()


# In[36]:


from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestClassifier
x = df.drop('Failure Type' , axis = 1)
y = df['Failure Type']
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

param_grid =  {
    'n_estimators':[10 , 20 , 50 , 100 , 200]
    }
model = RandomForestClassifier(random_state=42)
grid_param = GridSearchCV(param_grid=param_grid , verbose=5 , estimator= model  , cv = 5 , n_jobs=1)

grid_param.fit(x_train , y_train)
print("Best parameters" , grid_param.best_params_)
print("Best Score" , grid_param.best_score_)

best_model = grid_param.best_estimator_
test_score = best_model.score(x_test , y_test)

print("Test Score" , grid_param.best_score_)


# In[39]:


best_model.fit(x_train , y_train)
pred = best_model.predict(x_test)
from sklearn.metrics import accuracy_score , r2_score , f1_score , classification_report , recall_score
print("accuracy",accuracy_score(y_test , pred))
print("r2 score",r2_score(y_test , pred))
print("f1 score",f1_score(y_test , pred , average='macro'))
print("classification",classification_report(y_test , pred))
print("recall ",recall_score(y_test , pred , average='macro'))


# In[ ]:


joblib.load(best_model , 'rfc_best_model.pkl')

