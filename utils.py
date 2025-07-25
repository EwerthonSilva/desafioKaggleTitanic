import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

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

def treinar_modelo(modelo_tipo, parametros, X_train, y_train):
    try:
        if isinstance(modelo_tipo, RandomForestClassifier):
            model = RandomForestClassifier(
                criterion=parametros[0],
                n_estimators=parametros[1],
                max_depth=parametros[2],
                min_samples_split=parametros[3],
                min_samples_leaf=parametros[4],
                random_state=0
            )
        elif isinstance(modelo_tipo, GaussianNB):
            model = GaussianNB()

        elif isinstance(modelo_tipo, DecisionTreeClassifier):
            model = DecisionTreeClassifier(
                criterion=parametros[0],
                max_depth=parametros[1],
                min_samples_split=parametros[2],
                min_samples_leaf=parametros[3],
                random_state=0
            )

        elif isinstance(modelo_tipo, SVC):
            model = SVC(
                kernel=parametros[0],
                C=parametros[1],
                gamma=parametros[2],
                degree=parametros[3],
                random_state=0
            )

        elif isinstance(modelo_tipo, KNeighborsClassifier):
            model = KNeighborsClassifier(
                n_neighbors=parametros[0],
                p=parametros[1],
                n_jobs=-1
            )

        elif isinstance(modelo_tipo, LogisticRegression):
            model = LogisticRegression(
                solver=parametros[0],
                penalty=parametros[1],
                C=parametros[2],
                n_jobs=-1,
                random_state=0
            )
        else:
            raise ValueError("Modelo não reconhecido.")

        # Avaliação via cross-validation
        score = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
        mean_score = np.mean(score)

        return -mean_score  # Minimizar erro → maximizar acurácia

    except Exception as e:
        print(f"Erro ao treinar {modelo_tipo.__class__.__name__}: {e}")
        return 1.0  # Penaliza score alto em caso de erro
