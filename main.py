import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import utils as utils

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#%%
print(train.isnull().sum())
print(test.isnull().sum())
#%%
X_train = train.drop(['PassengerId', 'Survived'], axis = 1)
X_test = test.drop(['PassengerId'], axis = 1)

#%%
X_train = utils.normalize_feature(X_train)
X_test = utils.normalize_feature(X_test)

utils.create_feature(X_train)
utils.create_feature(X_test)

#%%
col = pd.Series(list(X_train.columns))
print(col)
#%% selecionar as features

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'mulher', 'porto', 'crianca']

X_train = X_train[features]
X_test =  X_test[features]

y_train = train['Survived']

#%% padronização das features (variáveis)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

#%% Logistic Regression
model_lr = LogisticRegression()
param_lr = [('liblinear','saga'),
            ('l1','l2'),
            (-0.5,3.5)]


#%% Naive Bayes para Classificação

from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()
param_nb = [(3.0)]


#%% KNN para Classificação
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5, p=2)
param_knn = [(2,10),
             (2,5)]


#%% SVM para Classificação
from sklearn.svm import SVC
model_svc = SVC( kernel='rbf', C=3, gamma=0.1, degree=2)
param_svc = [('linear', 'rbf', 'poly'),
             (1,10),
             (0.1,1.0),
             (2,6)]

#%% Arvore de decisão
from sklearn.tree import DecisionTreeClassifier
model_det = DecisionTreeClassifier()
param_det = [('entropy', 'gini'),
              (3, 20),
              (2,10),
              (1,10)]

#%% Random forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()

param_rf = [('entropy', 'gini'),
              (100, 1000),
              (3, 20),
              (2,10),
              (1,10)]

#%% Optimização dos hiperparametros
from skopt import gp_minimize
from functools import partial

func_objetivo = partial(utils.treinar_modelo, model_rf, X_train=X_train_sc, y_train=y_train)

otimos_rf = gp_minimize(
    func_objetivo,
    param_rf,
    random_state=0,
    verbose = 1,
    n_calls=30,
    n_random_starts=10,
    n_jobs= -1
)

#%%
print(otimos_rf.fun, otimos_rf.x)

#%%
model_rf = RandomForestClassifier(criterion=otimos_rf.x[0], n_estimators=otimos_rf.x[1], max_depth=otimos_rf.x[2], min_samples_split=otimos_rf.x[3], min_samples_leaf=otimos_rf.x[4],
                                  random_state=0)
#%%
model_rf.fit(X_train_sc, y_train)
y_pred = model_rf.predict(X_train_sc)
print(y_pred)
#%%
mc = confusion_matrix(y_train, y_pred)
print(mc)

#%%
score = model_rf.score(X_train_sc, y_train)
print(score)

#%%
y_pred = model_rf.predict(X_test_sc)
print(y_pred.size)
#%%
submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = y_pred

submission.to_csv('data/submission.csv', index = False)