# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:01:36 2022

@author: ReneGadeOne
"""
import mediapipe as mp
import time
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#data loading part
data = pd.read_csv("dataset.csv")

#based on 6 model
keys_data = data.drop(['b6', 'b82','id'], axis='columns')
X,Y = keys_data,data['b6']


#based on 82 models
#data_for_6 = data.loc[data['b6'] == 3] # number can be 0 to 5 
#keys_data = data_for_6.drop(['b6','id' , 'b82'], axis='columns')
#X,Y = keys_data,data_for_6['b82']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#model training function
def yoga_train_func (x_train ,y_train,x_test,y_test,model_name):
    model = SVC(C=10.0, kernel='poly')
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    # save the model to disk
    filename = model_name +'.pkl'
    pickle.dump(model, open(filename, 'wb')) 
    return print('score of '+ model_name +' is '+ str(score))
    
#yoga_train_func
yoga_train_func(x_train ,y_train,x_test,y_test,'cat_0')
