import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#%%
print(train.isnull().sum())
print(test.isnull().sum())
#%%
X_train = train.drop(['PassengerId', 'Survived'], axis = 1)
X_test = test.drop(['PassengerId'], axis = 1)

#%%
def normalize_feature(X):
  X['Age'] = X['Age'].fillna(X['Age'].mean())
  X['Fare'] = X['Fare'].fillna(X['Fare'].mean())
  X['Embarked'] = X['Embarked'].fillna('S')
  return X

def create_feature(X):
  X['mulher'] = X['Sex'].map(lambda x: 1 if x == 'female' else 0)
  X['porto'] = X['Embarked'].map(lambda x: 1 if x == 'S' else (2 if x == 'C' else 3))
  X['crianca'] = X['Age'].map(lambda x: 1 if x < 12 else 0)
  return X
#%%
X_train = normalize_feature(X_train)
X_test = normalize_feature(X_test)

create_feature(X_train)
create_feature(X_test)
#%%
col = pd.Series(list(X_train.columns))
print(col)
#%%
#selecionar as features

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'mulher', 'porto', 'crianca']

X_train = X_train[features]
X_test =  X_test[features]

y_train = train['Survived']

#%%
import matplotlib.pyplot as plt

for i in X_train.columns:
  plt.hist(X_train[i], bins=50, label=i)
  plt.title('Histogram of ' + i)
  plt.show()

#%%
gp = train.groupby(['Survived']).count()
table = pd.pivot_table(train, index='Survived', columns='Pclass', values='PassengerId', aggfunc='count')
print(table)

#%%
# padronização das features (variáveis)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

#%% Logistic Regression
model_lr = LogisticRegression(max_iter=1000)

score = cross_val_score(model_lr, X_train_sc, y_train, cv = 10)
print(np.mean(score))

#%% Naive Bayes para Classificação

from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()

score = cross_val_score(model_nb, X_train_sc, y_train, cv = 10)
print(np.mean(score))

#%% KNN para Classificação
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5, p=2)

score = cross_val_score(model_knn, X_train_sc, y_train, cv = 10)
print(np.mean(score))

#%% SVM para Classificação
from sklearn.svm import SVC
model_svc = SVC( kernel='rbf', C=3, gamma=0.1, degree=2)
score = cross_val_score(model_svc, X_train_sc, y_train, cv = 10)
print(np.mean(score))

#%% Arvore de decisão
from sklearn.tree import DecisionTreeClassifier
model_det = DecisionTreeClassifier(criterion='entropy', max_depth= 3, min_samples_leaf=1, min_samples_split=2, random_state=0)
score = cross_val_score(model_det, X_train_sc, y_train, cv = 10)
print(np.mean(score))


#%% Random forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(criterion='entropy', max_depth= 10, min_samples_leaf=1, min_samples_split=2, random_state=0)
score = cross_val_score(model_rf, X_train_sc, y_train, cv = 10)
print(np.mean(score))

#%% Optimização dos hiperparametros
from skopt import gp_minimize

def treinar_modelo(parametros):
  model_rf = RandomForestClassifier(criterion=parametros[0], n_estimators=parametros[1], max_depth=parametros[2],  min_samples_split=parametros[3], min_samples_leaf=parametros[4],
                                    random_state=0)
  score = cross_val_score(model_rf, X_train_sc, y_train, cv=10)
  mean_score = np.mean(score)
  print(mean_score)
  return -mean_score

parametros = [('entropy', 'gini'),
              (100, 1000),
              (3, 20),
              (2,10),
              (1,10)
              ]

otimos_rf = gp_minimize(treinar_modelo, parametros, random_state=0, verbose = 1, n_calls=50, n_random_starts=10, n_jobs= -1)

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