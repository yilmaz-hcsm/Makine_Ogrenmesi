# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler=pd.read_csv('maaslar.csv')

#dataFrame Olarak kolonları alıyoruz.(dilimleme)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#numpy arrayler dönüşümü.
X=x.values
Y=y.values

#linear reggression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

#görselleştirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()


#polynomal regression 2. dereceden
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)
print(x_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#görselleştirme
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

#polynomal regression 4. dereceden daha iyi tahmin ediyor.
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#görselleştirme
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.8]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.8]])))