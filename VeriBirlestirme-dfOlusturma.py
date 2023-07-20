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
print('\nverinin ilk hali:\n\n')
print(veriler)

#veri on isleme


#eksik veriler
#sci - kit learn

from sklearn.impute import SimpleImputer
#eksik verilerin ortalamasını yaz
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 1 ile 4 arasındaki stunları ayır
Yas = veriler.iloc[:,1:4].values

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])# eksik olan verilere yas ortalamasını gir


ulke = veriler.iloc[:,0:1].values # ulke stununu ayır


from sklearn import preprocessing

le = preprocessing.LabelEncoder()#ulkeler yerine sayı atamak için kullanırız

ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) #sayısal veriye dönüştürme


ohe = preprocessing.OneHotEncoder()# değerini yaz gerisine 0 atama işlemi yapar
ulke = ohe.fit_transform(ulke).toarray()




#print(list(range(22)))
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])# ülke kolonlarını oluşturma ve sonuca eşitleme


sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas']) # boy kilo yaş kolonlarını sonuca eşitleme



cinsiyet = veriler.iloc[:,-1].values#cinsiyeti verilerden ayırma

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])#cinsiyeti kolona ayırma


s=pd.concat([sonuc,sonuc2], axis=1)# sonuc ile sonuc 2 yi birleştirme 


s2=pd.concat([s,sonuc3], axis=1)# s ile sonuc3 ü birleştirme 

print('\n\n Verinin Son Hali: \n\n')
print(s2)

from sklearn.model_selection import train_test_split 
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0) #33 e 73 train e rasgele seç






