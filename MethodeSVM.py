# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 23:48:46 2019

@author: Amir
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, train_test_split,cross_validate
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
mydataset = pd.read_csv('D:\Etudes\M2\S1 Tronc commun\Machine Learning\Projet\phpYLeydd.csv')
db = mydataset.loc[mydataset.Phase.isin(["'D'", "'S'"])]

print(db.shape)
#On supprime les colonnes corrélées
X = db.drop(['Phase','X8','X11','X27','X28'],axis=1)
y = db['Phase']
#On divise notre jeu de données en train et en test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=2)
#Initialisation du classifieur
svclassifier =  SVC()
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
acc = svclassifier.score(X_test,y_test)
print("Score pour les parametres par défaut : ",acc)
#Recherche des meilleurs parametres avec GridSearch
parametres = [{'C':[0.1,1,10,20,100],'gamma':[10,'scale'],'kernel':['rbf','linear','poly']}]
grid = model_selection.GridSearchCV(estimator=svclassifier,param_grid=parametres,scoring='accuracy')
grille = grid.fit(X_train,y_train)
print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,["params","mean_test_score"]])
y_pred = grid.predict(X_test)
print("Les Meilleurs parametres sont : ",grille.best_params_)
print("Le meilleur score obtenu est : ",grille.best_score_)
#Affichage matrice de confusion et de classification
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))