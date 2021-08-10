import pandas as pd

# Leitura dos dados da base crédito
base = pd.read_csv("base_credit.csv")

# Mostrar dados gerais da tabela
base.describe()

# Colocar a média dos valores em idades menores que zero
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

# Localizar os valores nulos 
base.loc[pd.isnull(base['age'])]

# Separa os dados em previsores e classe
previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

# Tratamento de valores vazios através da média
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = imputer.transform(previsores[:, 0:3])

# Normailização dos dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Separando os dados em treinamento e teste
from sklearn.model_selection import train_test_split
prev_treinamento, prev_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25, random_state = 0)

# Implementação do algoritmo Arvores de Decisão
from sklearn.tree import DecisionTreeClassifier, export 
classificador = DecisionTreeClassifier(criterion = "entropy")
classificador.fit(prev_treinamento, classe_treinamento)

# Testar dados
resultado = classificador.predict(prev_teste)

# Exportar para op graphviz, para visualizar a árvore de decisão
export.export_graphviz( classificador,
                        out_file = "2.arvore.dot", 
                        feature_names = ['income','age','wloan'],
                        class_names = ['não', 'sim'],
                        filled = True,
                        leaves_parallel = True)


# Comparar os dados teste com os previsto pelo algortimo
from sklearn.metrics import confusion_matrix, accuracy_score
print ('Precisão: {:.2f}%'.format(accuracy_score(classe_teste, resultado)*100))
matriz = confusion_matrix(classe_teste, resultado)
print ('Matriz de precisão gerada com sucesso!')

