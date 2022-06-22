# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:23:45 2022

@author: Acer
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping

from module_campaign import EDA,ModelCreation,ModelEvaluation

#%% STATIC
CSV_PATH = os.path.join(os.getcwd(),'dataset','Train.csv')
JOB_ENCODER_PATH = os.path.join(os.getcwd(),'job_encoder.pkl')
MARITAL_ENCODER_PATH = os.path.join(os.getcwd(),'marital_encoder.pkl')
EDUCATION_ENCODER_PATH = os.path.join(os.getcwd(),'education_encoder.pkl')
DEFAULT_ENCODER_PATH = os.path.join(os.getcwd(),'default_encoder.pkl')
HOUSING_ENCODER_PATH = os.path.join(os.getcwd(),'housing_loan_encoder.pkl')
PERSONAL_ENCODER_PATH = os.path.join(os.getcwd(),'personal_loan_encoder.pkl')
COMMUNICATION_ENCODER_PATH = os.path.join(os.getcwd(),'communication_type_encoder.pkl')
MONTH_ENCODER_PATH = os.path.join(os.getcwd(),'month_encoder.pkl')
PREV_CAMPAIGN_ENCODER_PATH = os.path.join(os.getcwd(),'prev_outcome_encoder.pkl')
TERM_DEPO_ENCODER_PATH = os.path.join(os.getcwd(),'term_depo_subs_encoder.pkl')
SS_PATH = os.path.join(os.getcwd(),'std_campaign.pkl')
OHE_PATH = os.path.join(os.getcwd(),'ohe_campaign.pkl')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')

#%% DATA LOADING
df = pd.read_csv(CSV_PATH)

#%% DATA INSPECTION
df.info()
temp = df.describe().T
df.isna().sum() # lot of NaNs especially days_since_prev_campaign(need to be dropped)
df.duplicated().sum() # no duplicate data

# need to drop ID,days_since_prev_campaign
# target(y) = term_deposit_subscribed
df = df.drop(labels=['id','days_since_prev_campaign_contact'],axis=1)

df.boxplot() 
'''
Column balance has a lot of outliers. The min value is up to -8020 and the 
max value is 102128. But since both of this value are considered normal value,
they will not be removed.
'''

# to display the continuous data
con_data = ['customer_age','balance','day_of_month','last_contact_duration',
            'num_contacts_in_campaign','num_contacts_prev_campaign']
eda  = EDA()
eda.plot_con(df,con_data)

# to display categorical data
cat_data = df.columns.difference(['customer_age','balance','day_of_month',
                                  'last_contact_duration',
                                  'num_contacts_in_campaign',
                                  'num_contacts_prev_campaign'],
                                 sort=False).tolist()
eda.plot_cat(df,cat_data)

df.groupby(['job_type','term_deposit_subscribed']
           ).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['education','term_deposit_subscribed']
           ).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
'''
Majority of the people regardless their job type and education 
are not subscribing the term_deposit.
'''
#%% DATA CLEANING
# column id and days_since_prev_campaign have been dropped
# need to impute without removing the NaNs
# need to save the label encoder

df_dummy = df.copy()
le = LabelEncoder()

paths = [JOB_ENCODER_PATH,MARITAL_ENCODER_PATH,EDUCATION_ENCODER_PATH,
         DEFAULT_ENCODER_PATH,HOUSING_ENCODER_PATH,PERSONAL_ENCODER_PATH,
         COMMUNICATION_ENCODER_PATH,MONTH_ENCODER_PATH,
         PREV_CAMPAIGN_ENCODER_PATH,TERM_DEPO_ENCODER_PATH]

for index,cat in enumerate(cat_data):
    temp = df_dummy[cat]
    temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()])
    df_dummy[cat] = pd.to_numeric(temp,errors='coerce')
    with open(paths[index],'wb') as file:
        pickle.dump(le,file)

# Impute the NaNs
knn = KNNImputer()
df_dummy = knn.fit_transform(df_dummy)
df_dummy = pd.DataFrame(df_dummy)
df_dummy.columns = df.columns

# to make sure there is no decimal numbers in categorical column
for cat in cat_data:
    df_dummy[cat] = np.floor(df_dummy[cat])

#%% FEATURES SELECTION
# term_deposit_subscribed as the target

# categorical (features) vs categorical target(term_deposit_subscribed) using cramer's V
for cat in cat_data:
    print(cat)
    confussion_mat = pd.crosstab(df_dummy[cat],
                                 df_dummy['term_deposit_subscribed']).to_numpy()
    print(eda.cramers_corrected_stat(confussion_mat))
'''
The highest percentage is prev_campaign_outcome with 0.341.
Eventho it is the highest percentage, it shows no correlation to the target.
So no categorical column will be chosen as features
'''
# continuous(features) vs categorical target(term_deposit_subscribed) using LogReg
for con in con_data:
    logreg = LogisticRegression()
    logreg.fit(np.expand_dims(df_dummy[con],axis=-1),
               df_dummy['term_deposit_subscribed'])
    print(con)
    print(logreg.score(np.expand_dims(df_dummy[con],axis=-1),
                       df_dummy['term_deposit_subscribed'])) # accuracy
'''
Since all continuous columns has a high percentage(>0.88) which trained by
Logistic Regression, so all of the continuous data will be chosen as features
'''

#%% PREPROCESSING
X = df_dummy.loc[:,con_data]
y = df_dummy['term_deposit_subscribed']

# Features scaling
# Standard Scaler
std = StandardScaler()
X = std.fit_transform(X)

# save the standard scaler
with open(SS_PATH,'wb') as file:
    pickle.dump(std,file)

# One Hot Encoder
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=-1))

# save ohe model
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)
    
# split the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=123)

#%% Model development
nb_features = np.shape(X)[1:]
nb_class = len(np.unique(df['term_deposit_subscribed']))

mc = ModelCreation()
model = mc.simple_two_layer_model(nb_features=nb_features,nb_class=nb_class)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

plot_model(model,show_shapes=True,show_layer_names=(True))

# callbacks
tensorboard_callbacks = TensorBoard(log_dir=LOG_FOLDER_PATH)

early_stopping_callbacks = EarlyStopping(monitor='loss',patience=5)

# model training
hist = model.fit(x=X_train,y=y_train,batch_size=128,epochs=100,
                 validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callbacks,early_stopping_callbacks])

#%% 
hist.history.keys()

me = ModelEvaluation()
me.plot_model_evaluation(hist)

#%% MODEL EVALUATION
results = model.evaluate(X_test,y_test)
print(results)

y_pred = np.argmax(model.predict(X_test),axis=1)
y_true = np.argmax(y_test,axis=1)

cm = confusion_matrix(y_true, y_pred)
cr = classification_report(y_true, y_pred)
acc_score = accuracy_score(y_true, y_pred)

labels = ['Not Subscribe', 'Subscribe']
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(cm)
print(cr)
print("Accuracy score: " + str(acc_score))

#%% MODEL SAVING
model.save(MODEL_SAVE_PATH)

#%% DISCUSSION
'''
The deep learning model that only comprises of Dense, Dropout, and 
Batch Normalization layers successfully achieved the accuracy of 90.1%.

Recall and f1-score reported 0.99 and 0.95 respectively.

Tensorboard was used to display the loss and accuracy graph of the model.

Early stopping callback was used to prevent overfitting of the deep learning 
model, and during the model training, the epochs stop at 32.
'''

