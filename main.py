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
            (0.1,3.5)]


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

func_obj_rf = partial(utils.treinar_modelo, RandomForestClassifier(), X_train=X_train_sc, y_train=y_train)
func_obj_lr = partial(utils.treinar_modelo, LogisticRegression(), X_train=X_train_sc, y_train=y_train)
func_obj_dtc = partial(utils.treinar_modelo, DecisionTreeClassifier(), X_train=X_train_sc, y_train=y_train)
func_obj_svc = partial(utils.treinar_modelo, SVC(), X_train=X_train_sc, y_train=y_train)
func_obj_kneig = partial(utils.treinar_modelo, KNeighborsClassifier(), X_train=X_train_sc, y_train=y_train)
func_obj_gaussina = partial(utils.treinar_modelo, GaussianNB(), X_train=X_train_sc, y_train=y_train)

otimos_rf = gp_minimize(
    func_obj_rf,
    param_rf,
    random_state=0,
    verbose = 1,
    n_calls=30,
    n_random_starts=10,
    n_jobs= -1
)

otimos_lr = gp_minimize(
    func_obj_lr,
    param_lr,
    random_state=0,
    verbose = 1,
    n_calls=30,
    n_random_starts=10,
    n_jobs= -1
)

otimos_dtc = gp_minimize(
    func_obj_dtc,
    param_det,
    random_state=0,
    verbose = 1,
    n_calls=30,
    n_random_starts=10,
    n_jobs= -1
)

otimos_svc = gp_minimize(
    func_obj_svc,
    param_svc,
    random_state=0,
    verbose = 1,
    n_calls=30,
    n_random_starts=10,
    n_jobs= -1
)

otimos_kneig = gp_minimize(
    func_obj_kneig,
    param_knn,
    random_state=0,
    verbose = 1,
    n_calls=30,
    n_random_starts=10,
    n_jobs= -1
)

# otimos_gaussina = gp_minimize(
#     func_obj_gaussina,
#     param_,
#     random_state=0,
#     verbose=1,
#     n_calls=30,
#     n_random_starts=10,
#     n_jobs=-1
# )

#%%
print("otimos_rf:", otimos_rf.x)
print("otimos_lr:", otimos_lr.x)
print("otimos_dtc:", otimos_dtc.x)
print("otimos_svc:", otimos_svc.x)
print("otimos_kneig:", otimos_kneig.x)


#%%
model_rf = RandomForestClassifier(criterion=otimos_rf.x[0], n_estimators=otimos_rf.x[1], max_depth=otimos_rf.x[2], min_samples_split=otimos_rf.x[3], min_samples_leaf=otimos_rf.x[4],
                                  random_state=0)
model_lr = LogisticRegression(solver=otimos_lr.x[0],penalty=otimos_lr.x[1],C=otimos_lr.x[2], random_state=0)

model_svc = SVC(kernel=otimos_svc.x[0],
                C=otimos_svc.x[1],
                gamma=otimos_svc.x[2],
                degree=otimos_svc.x[3],
                random_state=0)

model_knn = KNeighborsClassifier(n_neighbors=otimos_kneig.x[0],
                                 p=otimos_kneig.x[1],
                                 n_jobs=-1)

model_det = DecisionTreeClassifier(criterion=otimos_dtc.x[0],
                                    max_depth=otimos_dtc.x[1],
                                    min_samples_split=otimos_dtc.x[2],
                                    min_samples_leaf=otimos_dtc.x[3],
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