import pandas as pd

# Importação dos dados e separação entre previsor e classe
base = pd.read_csv("risco_crédito.csv")
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# Numerização dos dados categóricos
# OBS: O algoritmos NaiveBayes é necessário valores númericos
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])

# Implementação do algoritmo NaiveBayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores,classe)

# Teste1: história boa, dívida alta, garantias nenhuma, renda > 35
print(classificador.predict([[0,0,1,2]]))
# Teste2: história ruim, dívida alta, garantias adequada, renda < 15
print(classificador.predict([[3,0,0,0]]))

# Mostrar os tipo de classes
print(classificador.classes_)

# Quantidade de valores por classe
print(classificador.class_count_)

# Probabildiade de cada classe
print(classificador.class_prior_)