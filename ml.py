#!/usr/bin/env python
# coding: utf-8

# # Semplice gioco per lavorare con controllo del flusso, funzioni, importazione di librerie e visualizzazione

# ## TASK 1
# Implementare una funzione per giocare a 'indovina il numero'.
# - Generare un numero intero casuale entro un range definito come input della funzione
# - La funzione valuta un secondo input che è il tentativo del giocatore umano
# - Confrontando il tentativo con il valore corretto, la funzione restituisce un feedback sulla base del fatto che il numero da indovinare sia uguale al tentativo, superiore o inferiore

# In[203]:


import numpy as np


# ## TASK 2
# Implementare un giocatore artificiale che competa con la funzione per indovinare il numero. Memorizzare l'esito dei testativi.

# ## TASK 3
# Implementare più giocatori che applicano più strategie diverse, riutilizzando il codice comune ai diversi giocatori.

# In[204]:


import pandas as pd


# In[205]:


import pandas as pd
df = pd.read_csv('dataset hc.csv')


# #### print (df)

# ## TASK 4
# Eseguire più partite fra giocatori che utilizzano strategie diverse e raccogliere i risultati ottenuti.

# ## TASK 5
# Visualizzare gli esiti dei diversi giocatori.

# In[206]:


# Data management
import pandas as pd

# Data preprocessing and trasformation (ETL)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, FunctionTransformer, Binarizer, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_iris, make_moons, make_classification


# Math and Stat modules
import numpy as np
from scipy.stats import sem
from random import choice

# Supervised Learning
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold, RepeatedKFold, ShuffleSplit, StratifiedShuffleSplit, learning_curve, validation_curve
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# Unsupervised Learning

# Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[207]:


print (df)


# In[208]:


get_ipython().system('head -n 10 df')


# In[209]:


get_ipython().system('Powershell -Command "Get-Content df -Head 10"')


# In[210]:


df.head(4)


# In[211]:


df.info()


# In[212]:


df.describe()


# In[213]:


df.hist(figsize=(22,9))


# In[214]:


matrice_correlazione = df.corr()


# In[215]:


matrice_correlazione['age'].sort_values(ascending=False)


# In[216]:


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# In[217]:


colonne_interesse = ['avg_glucose_level', 'age', 'hypertension', 'heart_disease']


# In[218]:


scatter_matrix(df[colonne_interesse], figsize=(22,9))


# In[219]:


def unknown_imputer(X):
  missing_value = 'Unknown'

  X = X.values
  X = np.array([[str(x[0])] for x in X])
  unique_values, count = np.unique(X,return_counts=True)
  num_nan = count[unique_values == missing_value]
  counting = count[unique_values != missing_value]
  values = unique_values[unique_values != missing_value]
  X_new = X.copy()
  freq = counting / np.sum(counting)
  X_new[X_new == missing_value] = np.random.choice(values,size=num_nan,p=freq)
  return X_new


# In[220]:


def NaN_imputer(X):
  missing_value = 'nan'

  X = X.values
  X = np.array([[str(x[0])] for x in X])
  unique_values, count = np.unique(X,return_counts=True)
  num_nan = count[unique_values == missing_value]
  counting = count[unique_values != missing_value]
  values = unique_values[unique_values != missing_value]
  X_new = X.copy()
  freq = counting / np.sum(counting)
  X_new[X_new == missing_value] = np.random.choice(values,size=num_nan,p=freq)
  return X_new


# In[221]:


print (df)


# In[222]:


df['gender'].value_counts()


# In[223]:


df = df[df.gender != 'Other']


# In[224]:


df['age'].value_counts()


# In[225]:


df['hypertension'].value_counts()


# In[226]:


df['avg_glucose_level'].value_counts()


# In[227]:


df['heart_disease'].value_counts()


# In[228]:


df['ever_married'].value_counts()


# In[229]:


df['work_type'].value_counts()


# In[230]:


df['Residence_type'].value_counts()


# In[231]:


df['bmi'].value_counts()


# In[232]:


df['smoking_status'].value_counts()


# In[233]:


one_hot_PP = Pipeline([
  ('one_hot', OneHotEncoder())
]) 


# In[234]:


robust_scaler_PP = Pipeline([ 
   ('imputer', FunctionTransformer(NaN_imputer)),
   ('scaler', RobustScaler())
 ])


# In[235]:


inputer_multiCategory_PP = Pipeline([
  ('imputer', FunctionTransformer(unknown_imputer)),
  ('one_hot', OneHotEncoder())
])


# In[236]:


inputer_number_PP = Pipeline([
   ('imputer', FunctionTransformer(NaN_imputer)),
   ('scaler', RobustScaler())
 ])


# In[237]:


ordinal_encoder_PP = Pipeline([
  ('ordinal_enc', OrdinalEncoder(categories=[['Yes', 'No']]))
])


# In[238]:


ordinal_encoder_PP2 = Pipeline([
  ('ordinal_enc', OrdinalEncoder(categories=[['Rural', 'Urban']]))
])


# In[239]:


data_preprocessing = ColumnTransformer(
  [
    ('age', robust_scaler_PP, ['age']),      
    ('gender', one_hot_PP, ['gender']),      
    ('ever_married', ordinal_encoder_PP, ['ever_married']),      
    ('avg_glucose_level', robust_scaler_PP, ['avg_glucose_level']),      
    ('Residence_type', ordinal_encoder_PP2, ['Residence_type']),      
    ('bmi', robust_scaler_PP, ['bmi']),      
    ('smoking_status', one_hot_PP, ['smoking_status']),
    ('work_type', one_hot_PP, ['work_type']) ],
    remainder = 'passthrough'
)
 


# In[240]:


v_target = df['stroke'].tolist()
df.drop(columns=['id', 'stroke'], inplace=True)


# In[241]:


Matrix = data_preprocessing.fit_transform (df)


# In[242]:


len(v_target)


# In[243]:


X_train, X_test, y_train, y_test = train_test_split(Matrix, v_target, test_size = 0.2, random_state = 42)


# In[244]:


len(X_train)


# In[267]:


import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Data management
import pandas as pd
import pickle

# Data preprocessing and trasformation (ETL)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, FunctionTransformer, Binarizer, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_iris, make_moons, make_classification


# Math and Stat modules
import numpy as np
from scipy.stats import sem, randint
from random import choice

# Supervised Learning
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold, RepeatedKFold, ShuffleSplit, StratifiedShuffleSplit, learning_curve, validation_curve, cross_validate
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# Hyperparameter Optimization
#from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.fixes import loguniform

# Unsupervised Learning
# Clustering algorithms and evaluation metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestNeighbors


# Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

import seaborn as sns
from sklearn.tree import export_graphviz



def compute_model_stats(y_real, y_predicted, label):
  from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve

  accuracy = round(accuracy_score(y_real, y_predicted),2) # accuracy = round(np.sum(y_predicted == y_real)/len(y_real),2)
  precision = round(precision_score(y_real, y_predicted),2)
  recall = round(recall_score(y_real, y_predicted),2)
  F1_score = round(f1_score(y_real, y_predicted),2)
  print('  ' + label + '  |  accuracy:', accuracy, '     precision:', precision, '    recall:', recall, '    f1_score:', F1_score)
  return accuracy, precision, recall, F1_score



class c_Perceptron:
  def __init__(self, X_train, X_test, y_train, y_test):
    perceptron = Perceptron()
    self.X_train, self.X_test = X_train, X_test
    self.y_train, self.y_test = y_train, y_test

    y_train_predicted = cross_val_predict(perceptron, X_train, y_train, cv = 10)
    self.train_accuracy, self.train_precision, self.train_recall, self.train_F1_score = compute_model_stats(y_train, y_train_predicted, 'Perceptron (Train)')

    y_test_predicted = cross_val_predict(perceptron, X_test, y_test, cv = 10)
    self.test_accuracy, self.test_precision, self.test_recall, self.test_F1_score = compute_model_stats(y_test, y_test_predicted, 'Perceptron (Test) ')


  def plot_learning_curves(self):
    Cs = [0.01, 0.1, 1, 10] # definire un insieme di valori di C tenendo in considerazione le precedenti osservazioni sul suo effetto 
    fig = plt.figure(figsize=(18,6))
    
    for i, c in enumerate(Cs):
      print('Training n°', i)
      MODEL = Perceptron()

      train_sizes, train_scores, test_scores = learning_curve(MODEL, X = self.X_test, y = self.y_test, train_sizes=np.linspace(0.1,1,10), cv = 5, n_jobs=-1, shuffle = True)

      train_mean = np.mean(train_scores, axis=1)
      train_std = np.std(train_scores, axis=1)
      test_mean = np.mean(test_scores, axis=1)
      test_std = np.std(test_scores, axis=1)

      ax = fig.add_subplot(150+(i+1))
      ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
      ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.2,1))
      ax.set_xlabel('Dimensione del training set')
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
    plt.show()




class c_Logistic_Regression:
  def __init__(self, X_train, X_test, y_train, y_test, treshold=0.9, max_iter=1000):
    self.X_train, self.X_test = X_train, X_test
    self.y_train, self.y_test = y_train, y_test
    
    y_train_predicted = self.predict_Y(X_train, y_train, treshold)
    self.y_train_predicted = y_train_predicted
    self.train_accuracy, self.train_precision, self.train_recall, self.train_F1_score = compute_model_stats(y_train, y_train_predicted, 'Log.Reg. (Train)')

    y_test_predicted = self.predict_Y(X_test, y_test, treshold)
    self.y_test_predicted = y_test_predicted
    self.test_accuracy, self.test_precision, self.test_recall, self.test_F1_score = compute_model_stats(y_test, y_test_predicted, 'Log.Reg. (Test) ')

  def predict_Y(self, X, Y, treshold):
    logit_cls = LogisticRegression(max_iter = 1000)
    y_predicted_score = cross_val_predict(logit_cls, X, Y, cv = 10, method='decision_function')
    prec, recall, soglia = precision_recall_curve(Y, y_predicted_score)
    try:
      soglia_prec = soglia[np.argmax(prec >= treshold)]
    except:
      soglia_prec = soglia[-1]
    y_predicted_score = y_predicted_score >= soglia_prec
    return y_predicted_score
  def plot_learning_curves(self):
    Cs = [0.01, 0.1, 1, 10] # definire un insieme di valori di C tenendo in considerazione le precedenti osservazioni sul suo effetto 
    fig = plt.figure(figsize=(18,6))
    
    for i, c in enumerate(Cs):
      print('Training n°', i)

      logit_cls = LogisticRegression(max_iter = 1000)
      train_sizes, train_scores, test_scores = learning_curve(logit_cls, X = self.X_test, y = self.y_test, train_sizes=np.linspace(0.1,1,10), cv = 5, n_jobs=-1, shuffle = True)

      train_mean = np.mean(train_scores, axis=1)
      train_std = np.std(train_scores, axis=1)
      test_mean = np.mean(test_scores, axis=1)
      test_std = np.std(test_scores, axis=1)

      ax = fig.add_subplot(150+(i+1))
      ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
      ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.2,1))
      ax.set_xlabel('Dimensione del training set')
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
    plt.show()



class c_linear_SVM:
  def __init__(self, X, X_test, Y, Y_test):
    self.X = X
    self.Y = Y
    svm_cls = LinearSVC(C=1, max_iter=50000)
    svm_cls.fit(X, Y)

    y_train_predicted = svm_cls.predict(X)
    self.train_accuracy, self.train_precision, self.train_recall, self.train_F1_score = compute_model_stats(Y, y_train_predicted, 'SVM linear (Train)')

    y_test_predicted = svm_cls.predict(X_test)
    self.test_accuracy, self.test_precision, self.test_recall, self.test_F1_score = compute_model_stats(Y_test, y_test_predicted, 'SVM linear (test) ')


  def plot_learning_curves(self):
    Cs = [0.01, 0.1, 1, 10] # definire un insieme di valori di C tenendo in considerazione le precedenti osservazioni sul suo effetto 
    fig = plt.figure(figsize=(18,6))
    for i, c in enumerate(Cs):
      print('Training SVM per C =', c)
      svm_cls = LinearSVC(C = c, max_iter=50000)

      train_sizes, train_scores, test_scores = learning_curve(svm_cls, X = self.X, y = self.Y, train_sizes=np.linspace(0.1,1,10), cv = 5, n_jobs=-1, shuffle = True)

      train_mean = np.mean(train_scores, axis=1)
      train_std = np.std(train_scores, axis=1)
      test_mean = np.mean(test_scores, axis=1)
      test_std = np.std(test_scores, axis=1)

      ax = fig.add_subplot(150+(i+1))
      ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
      ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.2,1))
      ax.set_xlabel('Dimensione del training set')
      ax.set_ylabel('Accuracy C = ' + str(c))
      ax.legend(loc='lower right')
    plt.show()





class c_non_linear_SVM:
  def __init__(self, X, X_test, Y, Y_test, kernel, D):
    assert kernel in ['poly', 'rbf'], "Wrong kernel modality"

    if kernel == 'poly':
      SVM = SVC(kernel="poly", degree=D['degree'], coef0=D['coef0'])
    elif kernel == 'rbf':
      SVM = SVC(kernel="rbf", gamma=D['gamma'], C=D['C'])
    '''param_grid = [
          {'kernel': ['poly'], 'degree': [1, 2, 3], 'coef0': [1, 10, 50]}
          {'kernel': ['rbf'], 'gamma': [.1, 5, 10], 'C': [0.1, 1, 1000]},
      ]'''

    self.SVM = SVM
    self.X = X
    self.Y = Y

    SVM.fit(X, Y)
    y_train_predicted = SVM.predict(X)
    self.train_accuracy, self.train_precision, self.train_recall, self.train_F1_score = compute_model_stats(Y, y_train_predicted, 'SVM non-linear('+kernel+') ' + str(D) +' (Train)')
    y_test_predicted = SVM.predict(X_test)
    self.test_accuracy, self.test_precision, self.test_recall, self.test_F1_score = compute_model_stats(Y_test, y_test_predicted, 'SVM non-linear('+kernel+')' + str(D) +' (test) ')


  def plot_learning_curves_RBF(self):
    gamma1, gamma2 = 0.1, 2
    C1, C2 = 0.01, 5
    hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for gamma, C in hyperparams:
      rbf_kernel_svm_clf = SVC(kernel="rbf", gamma = gamma, C = C)
      train_size, train_scores, test_scores = learning_curve(rbf_kernel_svm_clf,
                                                        X=self.X,
                                                        y=self.Y,
                                                        train_sizes=np.linspace(0.1,1.0,10),
                                                        cv=5,
                                                        n_jobs=-1)
      print('fatto {}, {}'.format(gamma,C))
      train_means.append(np.mean(train_scores, axis=1))
      train_stds.append(np.std(train_scores, axis=1))
      test_means.append(np.mean(test_scores, axis=1))
      test_stds.append(np.std(test_scores, axis=1))
      train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(4):
      ax = fig.add_subplot(221+i)
      ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.4,1))
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
      ax.set_title(r"$\gamma={}, C={}$".format(*hyperparams[i]), fontsize=18)
    plt.show()



  def plot_learning_curves_POLY(self):
    dg1, dg2 = 1, 5 # degree
    C1, C2 = 1, 10 # 0.01, 5 # coeff0
    hyperparams = (dg1, C1), (dg1, C2), (dg2, C1), (dg2, C2)

    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for Degree, Coef0 in hyperparams:
      kernel_svm_clf = SVC(kernel="poly", degree = Degree, coef0 = Coef0)
      train_size, train_scores, test_scores = learning_curve(kernel_svm_clf,
                                                        X=self.X,
                                                        y=self.Y,
                                                        train_sizes=np.linspace(0.1,1.0,10),
                                                        cv=5,
                                                        n_jobs=-1)
      print('fatto {}, {}'.format(Degree, Coef0))
      train_means.append(np.mean(train_scores, axis=1))
      train_stds.append(np.std(train_scores, axis=1))
      test_means.append(np.mean(test_scores, axis=1))
      test_stds.append(np.std(test_scores, axis=1))
      train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(4):
      ax = fig.add_subplot(221+i)
      ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.4,1))
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
      ax.set_title('Degree: ' + str(hyperparams[i][0]) + ' |  Coef0: ' + str(hyperparams[i][1]))
      #ax.set_title(r"$\Degree={}, coef0={}$".format(*hyperparams[i]), fontsize=18)
    plt.show()




class c_decision_tree:
  def __init__(self, X, X_test, Y, Y_test):
    self.X = X
    self.Y = Y
    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42) # min_samples_leaf
    tree_clf.fit(X, Y)
    self.tree_clf = tree_clf
    
    y_train_predicted = tree_clf.predict(X)
    self.train_accuracy, self.train_precision, self.train_recall, self.train_F1_score = compute_model_stats(Y, y_train_predicted, 'Tree (Train)')

    y_test_predicted = tree_clf.predict(X_test)
    self.test_accuracy, self.test_precision, self.test_recall, self.test_F1_score = compute_model_stats(Y_test, y_test_predicted, 'Tree (test) ')



  def plot_learning_curve(self):
    min_leaf = [5, 10, 100, 200, 350]

    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for mlf in min_leaf:
      dt_mlf = DecisionTreeClassifier(min_samples_leaf=mlf, random_state=42, max_depth=15)
      train_size, train_scores, test_scores = learning_curve(dt_mlf,
                                                          X=self.X,
                                                          y=self.Y,
                                                          train_sizes=np.linspace(0.1,1.0,10),
                                                          cv=10,
                                                          n_jobs=-1)
      print('fatto {}'.format(mlf))
      train_means.append(np.mean(train_scores, axis=1))
      train_stds.append(np.std(train_scores, axis=1))
      test_means.append(np.mean(test_scores, axis=1))
      test_stds.append(np.std(test_scores, axis=1))
      train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(5):
      ax = fig.add_subplot(231+i)
      ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.4,1))
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
      ax.set_title(r"min_sam_leaf: {}".format(min_leaf[i]), fontsize=12)
    plt.show()





class c_random_forest:
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, stratify=Y)

    # Random Forest
    self.RF_clf = RandomForestClassifier(n_estimators=250, max_leaf_nodes=64, max_depth=50, n_jobs=-1)
    self.RF_clf.fit(self.X_train, self.y_train)
    y_train_predicted = self.RF_clf.predict(X)
    compute_model_stats(Y, y_train_predicted, 'Random Forest (Train)')
    y_test_predicted = self.RF_clf.predict(self.X_test)
    compute_model_stats(self.y_test, y_test_predicted, 'Random Forest (test) ')

    # Extra Tree
    self.et_clf = self.et_clf = ExtraTreesClassifier(n_estimators=250, max_leaf_nodes=64, n_jobs=-1)
    self.et_clf.fit(self.X_train, self.y_train)
    y_train_predicted = self.et_clf.predict(X)
    compute_model_stats(Y, y_train_predicted, 'Extra Trees (Train)')
    y_test_predicted = self.et_clf.predict(self.X_test)
    compute_model_stats(self.y_test, y_test_predicted, 'Extra Trees (test) ')

    # Ada Boost
    self.ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=6), n_estimators=100, algorithm='SAMME.R', learning_rate=0.5)
    y_train_predicted = self.et_clf.predict(X)
    compute_model_stats(Y, y_train_predicted, 'ADA Boosting (Train)')
    y_test_predicted = self.et_clf.predict(self.X_test)
    compute_model_stats(self.y_test, y_test_predicted, 'ADA Boosting (test) ')



  def plot_learning_curve(self):
    OPT = [None, 2, 5, 10, 30]

    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for opt in OPT:
      dt_mlf = RandomForestClassifier(n_estimators=250, max_leaf_nodes=64, max_depth=opt, n_jobs=-1)  # n_estimators=250, max_leaf_nodes=64  max_features=10)   # min_samples_leaf=mlf, random_state=42)
      train_size, train_scores, test_scores = learning_curve(dt_mlf,
                                                          X=self.X,
                                                          y=self.Y,
                                                          train_sizes=np.linspace(0.1,1.0,10),
                                                          cv=10,
                                                          n_jobs=-1)
      print('fatto {}'.format(str(opt)))
      train_means.append(np.mean(train_scores, axis=1))
      train_stds.append(np.std(train_scores, axis=1))
      test_means.append(np.mean(test_scores, axis=1))
      test_stds.append(np.std(test_scores, axis=1))
      train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(5):
      ax = fig.add_subplot(231+i)
      ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.4,1))
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
      ax.set_title(r"max_depth: {}".format(OPT[i]), fontsize=12)
    plt.show()



# In[246]:


PERCEPTRON = c_Perceptron(X_train, X_test, y_train, y_test)
PERCEPTRON.plot_learning_curves()


# In[268]:


LOG_REGR = c_Logistic_Regression(X_train, X_test, y_train, y_test, treshold=0.5)
LOG_REGR.plot_learning_curves()


# In[252]:


TREE = c_decision_tree(X_train, X_test, y_train, y_test)
TREE.plot_learning_curve()


# In[253]:


RF = c_random_forest(X_train, y_train)
RF.plot_learning_curve()


# In[259]:


LINEAR_SVM = c_linear_SVM(X_train, X_test, y_train, y_test)
LINEAR_SVM.plot_learning_curves()


# In[261]:


NON_LINEAR_SVM = c_non_linear_SVM(X_train, X_test, y_train, y_test, 'poly', {'degree': 5, 'coef0': 10})
NON_LINEAR_SVM.plot_learning_curves_POLY()


# In[269]:


NON_LINEAR_SVM = c_non_linear_SVM(X_train, X_test, y_train, y_test, 'rbf', {'gamma': 1, 'C': 0.01})
NON_LINEAR_SVM.plot_learning_curves_RBF()


# In[ ]:




