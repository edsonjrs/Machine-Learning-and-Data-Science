import pandas as pd

# Leitura da Base de Dados
base = pd.read_csv("Wine Quality - White.csv", sep=';')

# Separar previsores e classe
previsores = base.iloc[:, 0:11].values
classes = base.iloc[:, 11].values

# Separar base teste
from sklearn.model_selection import train_test_split
prev_treino, prev_teste, classe_treino, classe_teste = train_test_split(previsores, classes, test_size = 0.10, random_state = 0)

# Aplicação do algoritmo Arvores de Decisao
from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion = "entropy")
classificador.fit(prev_treino, classe_treino)

# Testar o algortimo com a base teste
resultado = classificador.predict(prev_teste)

# Mostrar as classes de resultados
print(classificador.classes_)

# Exportar para op graphviz, para visualizar a árvore de decisão
export.export_graphviz( classificador,
                        out_file = "2.arvore.dot", 
                        feature_names = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
                        class_names = ['3','4','5','6','7','8','9'],
                        filled = True,
                        leaves_parallel = True)

# Comparar os resultados predizidos pelo algoritmo o classe teste
from sklearn.metrics import confusion_matrix, accuracy_score
print ('Precisão: {:.2f}%'.format(accuracy_score(classe_teste, resultado)*100))
matriz = confusion_matrix(classe_teste, resultado)
print ('Matriz de precisão gerada com sucesso!')

