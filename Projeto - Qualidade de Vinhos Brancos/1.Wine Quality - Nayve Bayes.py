import pandas as pd

# Leitura da Base de Dados
base = pd.read_csv("Wine Quality - White.csv", sep=';')

# Separar previsores e classe
previsores = base.iloc[:, 0:11].values
classes = base.iloc[:, 11].values

## ------------------ Não é viável pelo algoritmo Naive Bayes ------------------
# Normailização dos dados
#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# previsores = scaler.fit_transform(previsores)
#
# -----------------------------------------------------------------------------

# Separar base teste
from sklearn.model_selection import train_test_split
prev_treino, prev_teste, classe_treino, classe_teste = train_test_split(previsores, classes, test_size = 0.10, random_state = 0)

# Aplicação do algoritmo Naive_Bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(prev_treino, classe_treino)

# Mostrar as classes de resultados
print(classificador.classes_)

# Mostrar a quantidade de resultados por classe
print(classificador.class_count_)

# Probabilidade de cada classe
print(classificador.class_prior_)

# Testar o algortimo com a base teste
resultado = classificador.predict(prev_teste)

# Comparar os resultados predizidos pelo algoritmo o classe teste
from sklearn.metrics import confusion_matrix, accuracy_score
print ('Precisão: {:.2f}%'.format(accuracy_score(classe_teste, resultado)*100))
matriz = confusion_matrix(classe_teste, resultado)
print ('Matriz de precisão gerada com sucesso!')

