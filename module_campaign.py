# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:36:04 2022

@author: Acer
"""

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,BatchNormalization
from tensorflow.keras import Input
import numpy as np
import seaborn as sns
import scipy.stats as ss

class EDA():
    def __init__(self):
        pass
    
    def plot_cat(self,df,cat_data):
        '''
        This function is to generate plots for categorical columns

        Parameters
        ----------
        df : DataFrame
            DESCRIPTION.
        cat_data : LIST
            categorical column inside the dataframe.

        Returns
        -------
        None.

        '''
        cat_data = df.columns.difference(['customer_age','balance',
                                          'day_of_month',
                                          'last_contact_duration',
                                          'num_contacts_in_campaign',
                                          'num_contacts_prev_campaign'],
                                         sort=False).tolist()
        for cat in cat_data:
            plt.figure()
            sns.countplot(df[cat])
            plt.show()

    def plot_con(self,df,con_data):
        '''
        This function is to generate plots for continuous columns

        Parameters
        ----------
        df : DataFrame
            DESCRIPTION.
        continuous_col : LIST
            continuous column inside the dataframe.

        Returns
        -------
        None.

        '''
        con_data = ['customer_age','balance','day_of_month',
                    'last_contact_duration','num_contacts_in_campaign',
                    'num_contacts_prev_campaign']
        for con in con_data:
            plt.figure()
            sns.distplot(df[con])
            plt.show()
            
    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher,         
            Journal of the Korean Statistical Society 42 (2013): 323-328    
        """    
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n    
        r,k = confusion_matrix.shape    
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    
    
class ModelCreation():
    def __init__(self):
        pass
    
    def simple_two_layer_model(self,nb_features,nb_class,node_num=64,
                               drop_rate=0.2):
        model = Sequential()
        model.add(Input(shape=(nb_features))) #input length
        model.add(Dense(node_num,activation='linear', name='Hidden_layer1'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(node_num,activation='linear', name='Hidden_layer2'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(nb_class,activation='softmax',name='Output_layer')) # output 
        model.summary()
        
        return model

class ModelEvaluation():
    def __init__(self):
        pass
    
    def plot_model_evaluation(self,hist):
        hist_keys = [i for i in hist.history.keys()]
        plt.figure()
        plt.plot(hist.history[hist_keys[0]])
        plt.plot(hist.history[hist_keys[2]])
        plt.legend(['train_loss','val_loss'])
        plt.title('Loss')
        plt.show()

        plt.figure()
        plt.plot(hist.history[hist_keys[1]])
        plt.plot(hist.history[hist_keys[3]])
        plt.legend(['train_acc','val_acc'])
        plt.title('Accuracy')
        plt.show()
        
