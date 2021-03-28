# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:21:12 2021

@author: saake
"""
'''
1. Import libraries
2. Import Data
3. Process data into different models and splits
4. Do linear regression
5. Mess around with parameters for highest accuracy
6. Do MLP
7. Mess around for highest accuracy
8. Pick the better regression for each feature
9. Plot and visualize
10. Start with Flask
11. Get the basic UI up
12. Get the database/processing up
13. Get visualization to show after upload of dataset
14. Test the website by uploading the dataset to it
15. Upload/presentation
'''


# Import libraries
import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
import pickle

def getBestReg(X_train, X_test, y_train, y_test):
    #Do Linear Regression
    regL = LinearRegression(normalize=False).fit(X_train, y_train)
    
    avg = 0
    for i in range(1,50):
        reg = LinearRegression(normalize=False).fit(X_train, y_train)
        avg+=reg.score(X_test, y_test)
    
    lr = avg/20
    
    #Do MLP
    
    mlp = MLPRegressor(max_iter=800, alpha=1e-4,
                        solver='lbfgs', verbose=0, tol=1e-8, activation='identity')
    regM = mlp.fit(X_train, y_train.values.ravel())
    
    avg = 0
    for i in range(1,50):
        regr = mlp.fit(X_train, y_train.values.ravel())
        avg+=regr.score(X_test, y_test)
    
    ml = avg/20
    print(ml)
    
    if lr > ml:
        return regL
    else:
        return regM

#Import Data
data = read_csv('./Restaurant-Dataset.csv', skiprows=[53,54,55,56,57,58,59,60,61,62,63,64,65])

#Process data
season = {'Winter': 1,'Spring': 2, 'Summer': 3, 'Fall': 4}
data.Season = [season[item] for item in data.Season]

beefDataX = data[['Beef bought', 'Beef thrown', 'Beef deficit', 'Number of sales', 'Season']].copy()
beefDataY = data[['Beef Y']].copy()

beefX_train, beefX_test, beefY_train, beefY_test = train_test_split(beefDataX, beefDataY, test_size=0.2)

chickenDataX = data[['Chicken bought', 'Chicken thrown', 'Chicken deficit', 'Number of sales', 'Season']].copy()
chickenDataY = data[['Chicken Y']].copy()

chickenX_train, chickenX_test, chickenY_train, chickenY_test = train_test_split(chickenDataX, chickenDataY, test_size=0.2)

porkDataX = data[['Pork bought', 'Pork thrown', 'Pork deficit', 'Number of sales', 'Season']].copy()
porkDataY = data[['Pork Y']].copy()

porkX_train, porkX_test, porkY_train, porkY_test = train_test_split(porkDataX, porkDataY, test_size=0.2)

turkeyDataX = data[['Turkey bought', 'Turkey thrown', 'Turkey deficit', 'Number of sales', 'Season']].copy()
turkeyDataY = data[['Turkey Y']].copy()

turkeyX_train, turkeyX_test, turkeyY_train, turkeyY_test = train_test_split(turkeyDataX, turkeyDataY, test_size=0.2)

Other_meatDataX = data[['Other meat bought', 'Other meat thrown', 'Other meat deficit', 'Number of sales', 'Season']].copy()
Other_meatDataY = data[['Other meat Y']].copy()

Other_meatX_train, Other_meatX_test, Other_meatY_train, Other_meatY_test = train_test_split(Other_meatDataX, Other_meatDataY, test_size=0.2)

eggsDataX = data[['Eggs bought', 'Eggs thrown', 'Eggs deficit', 'Number of sales', 'Season']].copy()
eggsDataY = data[['Eggs Y']].copy()

eggsX_train, eggsX_test, eggsY_train, eggsY_test = train_test_split(eggsDataX, eggsDataY, test_size=0.2)

root_vegetablesDataX = data[['Root vegetables bought', 'Root vegetables thrown', 'Root vegetables deficit', 'Number of sales', 'Season']].copy()
root_vegetablesDataY = data[['Root vegetables Y']].copy()

root_vegetablesX_train, root_vegetablesX_test, root_vegetablesY_train, root_vegetablesY_test = train_test_split(root_vegetablesDataX, root_vegetablesDataY, test_size=0.2)

gourdsDataX = data[['Gourds bought', 'Gourds thrown', 'Gourds deficit', 'Number of sales', 'Season']].copy()
gourdsDataY = data[['Gourds Y']].copy()

gourdsX_train, gourdsX_test, gourdsY_train, gourdsY_test = train_test_split(gourdsDataX, gourdsDataY, test_size=0.2)

leafy_vegetablesDataX = data[['Leafy vegetables bought', 'Leafy vegetables thrown', 'Leafy vegetables deficit', 'Number of sales', 'Season']].copy()
leafy_vegetablesDataY = data[['Leafy vegetables Y']].copy()

leafy_vegetablesX_train, leafy_vegetablesX_test, leafy_vegetablesY_train, leafy_vegetablesY_test = train_test_split(leafy_vegetablesDataX, leafy_vegetablesDataY, test_size=0.2)

bulky_vegetablesDataX = data[['Bulky vegetables bought', 'Bulky vegetables thrown', 'Bulky vegetables deficit', 'Number of sales', 'Season']].copy()
bulky_vegetablesDataY = data[['Bulky vegetables Y']].copy()

bulky_vegetablesX_train, bulky_vegetablesX_test, bulky_vegetablesY_train, bulky_vegetablesY_test = train_test_split(bulky_vegetablesDataX, bulky_vegetablesDataY, test_size=0.2)

other_vegetablesDataX = data[['Other vegetables bought', 'Other vegetables thrown', 'Other vegetables deficit', 'Number of sales', 'Season']].copy()
other_vegetablesDataY = data[['Other vegetables Y']].copy()

other_vegetablesX_train, other_vegetablesX_test, other_vegetablesY_train, other_vegetablesY_test = train_test_split(other_vegetablesDataX, other_vegetablesDataY, test_size=0.2)

breadsDataX = data[['Breads bought', 'Breads thrown', 'Breads deficit', 'Number of sales', 'Season']].copy()
breadsDataY = data[['Breads Y']].copy()

breadsX_train, breadsX_test, breadsY_train, breadsY_test = train_test_split(breadsDataX, breadsDataY, test_size=0.2)

grainsDataX = data[['Grains bought', 'Grains thrown', 'Grains deficit', 'Number of sales', 'Season']].copy()
grainsDataY = data[['Grains Y']].copy()

grainsX_train, grainsX_test, grainsY_train, grainsY_test = train_test_split(grainsDataX, grainsDataY, test_size=0.2)

dairyDataX = data[['Dairy bought', 'Dairy thrown', 'Dairy deficit', 'Number of sales', 'Season']].copy()
dairyDataY = data[['Dairy Y']].copy()

dairyX_train, dairyX_test, dairyY_train, dairyY_test = train_test_split(dairyDataX, dairyDataY, test_size=0.2)

x = dairyDataX.iloc[1]

models = {}
models.setdefault('beef', getBestReg(beefX_train, beefX_test, beefY_train, beefY_test))
print(models.get('beef').predict([np.array([80, 6, 0, 300, 1])]))

models.setdefault('chicken', getBestReg(chickenX_train, chickenX_test, chickenY_train, chickenY_test))
models.setdefault('pork', getBestReg(porkX_train, porkX_test, porkY_train, porkY_test))
models.setdefault('turkey', getBestReg(turkeyX_train, turkeyX_test, turkeyY_train, turkeyY_test))
models.setdefault('Other meat', getBestReg(Other_meatX_train, Other_meatX_test, Other_meatY_train, Other_meatY_test))
models.setdefault('eggs', getBestReg(eggsX_train, eggsX_test, eggsY_train, eggsY_test))
models.setdefault('root vegetables', getBestReg(root_vegetablesX_train, root_vegetablesX_test, root_vegetablesY_train, root_vegetablesY_test))
models.setdefault('gourds', getBestReg(gourdsX_train, gourdsX_test, gourdsY_train, gourdsY_test))
models.setdefault('bulky vegetables', getBestReg(bulky_vegetablesX_train, bulky_vegetablesX_test, bulky_vegetablesY_train, bulky_vegetablesY_test))
models.setdefault('other vegetables', getBestReg(other_vegetablesX_train, other_vegetablesX_test, other_vegetablesY_train, other_vegetablesY_test))
models.setdefault('breads', getBestReg(breadsX_train, breadsX_test, breadsY_train, breadsY_test))
models.setdefault('grains', getBestReg(grainsX_train, grainsX_test, grainsY_train, grainsY_test))
models.setdefault('dairy', getBestReg(dairyX_train, dairyX_test, dairyY_train, dairyY_test))

# ax = plt.gca()
# plt.plot([1,2,3,4,5,6,7,8,9,10,11], beefY_test, color='blue')
# plt.plot([1,2,3,4,5,6,7,8,9,10,11], beefY_pred, color='red')


# pickling
pickle.dump(models, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[4, 300, 500]]))





