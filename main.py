from imports import *

from utils import *
# import utils as utils

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#%%
print(train.isnull().sum())
print(test.isnull().sum())
#%%
X_train = train.drop(['PassengerId', 'Survived'], axis = 1)
X_test = test.drop(['PassengerId'], axis = 1)

#%%
X_train = normalize_feature(X_train)
X_test = normalize_feature(X_test)

create_feature(X_train)
create_feature(X_test)

#%%
col = pd.Series(list(X_train.columns))
print(col)
#%% selecionar as features

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'mulher', 'porto', 'crianca']

X_train = X_train[features]
X_test =  X_test[features]

y_train = train['Survived']

#%% padronização das features (variáveis)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

#%%
resultados = []

#%%
for nome_modelo, conteudo in modelos_parametros.items():
    modelo = conteudo['modelo']
    parametros = conteudo['parametros']

    print(f"\n⏳ Otimizando: {nome_modelo} ...")

    if parametros:  # Tem hiperparâmetros → usar gp_minimize
        func_obj = partial(treinar_modelo, modelo, X_train=X_train_sc, y_train=y_train)

        resultado = gp_minimize(
            func=func_obj,
            dimensions=parametros,
            random_state=0,
            verbose=0,
            n_calls=30,
            n_random_starts=10,
            n_jobs=-1
        )

        melhores_param = resultado.x
        melhor_score = -resultado.fun  # Acurácia média

        print(f"✅ {nome_modelo} - Melhor score: {melhor_score:.4f}")
        print(f"Parâmetros: {melhores_param}")

        resultados.append((nome_modelo, melhor_score, melhores_param))

    else:  # GaussianNB, sem otimização
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(modelo, X_train_sc, y_train, cv=10, scoring='accuracy')
        media_score = score.mean()

        print(f"✅ {nome_modelo} - Score fixo: {media_score:.4f}")
        resultados.append((nome_modelo, media_score, None))



#%%
# Ordenar por score decrescente
resultados.sort(key=lambda x: x[1], reverse=True)

print("\n🏆 Ranking dos modelos:")
for nome, score, param in resultados:
    print(f"{nome}: {score:.4f}")

melhor_modelo_nome, melhor_score, melhores_param = resultados[0]
print(f"\n🚀 Melhor modelo: {melhor_modelo_nome} com score {melhor_score:.4f}")



#%%
# model_rf.fit(X_train_sc, y_train)
# y_pred = model_rf.predict(X_train_sc)
# print(y_pred)
#%%
# mc = confusion_matrix(y_train, y_pred)
# print(mc)

#%%
# score = model_rf.score(X_train_sc, y_train)
# print(score)

#%%
# y_pred = model_rf.predict(X_test_sc)
# print(y_pred.size)
#%%
# submission = pd.DataFrame()
# submission['PassengerId'] = test['PassengerId']
# submission['Survived'] = y_pred

# submission.to_csv('data/submission.csv', index = False)