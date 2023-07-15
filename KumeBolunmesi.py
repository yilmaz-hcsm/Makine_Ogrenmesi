# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:53:51 2023

@author: yilma
"""

#kutuphaneler bolumu
#ders1 kutuphanelerin yuklenmesi
import pandas as pd


#verileri yükleme
veriler= pd.read_csv('veriler.csv')

print(veriler)

from sklearn.model_section import train_test_split
