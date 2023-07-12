# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme

veriler = pd.read_csv('eksikveriler.csv')
#pd.read_csv("veriler.csv")
print(veriler)



from sklearn import preprocessing
le=preprocessing.LabelEncoder()#veriyi sayısallaştırmak için 


ulke=veriler.iloc[:,0:1].values#ülkeleri ayır
print(ulke)

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])#ülke verisini sayısallaştır
print(ulke)


ohe=preprocessing.OneHotEncoder()#kategorik verilerin binarizasyonunu gerçekleştirmemizi sağlar
ulke=ohe.fit_transform(ulke).toarray() #mevcut değere 1 diğerlerine 0 ver
print(ulke)