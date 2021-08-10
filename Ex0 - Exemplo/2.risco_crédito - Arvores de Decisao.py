import pandas as pd

# Importação dos dados e separação entre previsor e classe
base = pd.read_csv("risco_crédito.csv")
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# Numerização dos dados categóricos
# OBS: Os algoritmos NaiveBayes só funcionam com valores númericos
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])

# Implementação do algoritmo NaiveBayes
from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion = "entropy")
classificador.fit(previsores,classe)

# Mostrar os atributos mais importantes
print(classificador.feature_importances_)

# Exportar para op graphviz, para visualizar a árvore de decisão
export.export_graphviz( classificador,
                        out_file = "2.arvore.dot", 
                        feature_names = ['historia', 'divida', 'garantia', 'renda'],
                        class_names = ['alto', 'moderado', 'baixa'],
                        filled = True,
                        leaves_parallel = True)

# Teste1: história boa, dívida alta, garantias nenhuma, renda > 35
print(classificador.predict([[0,0,1,2]]))
# Teste2: história ruim, dívida alta, garantias adequada, renda < 15
print(classificador.predict([[3,0,0,0]]))