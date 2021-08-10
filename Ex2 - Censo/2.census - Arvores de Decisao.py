import pandas as pd

base = pd.read_csv("census.csv")

base.describe()

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:,14].values

# Transformação de variáveis categóricas para numéricas
from sklearn.preprocessing import LabelEncoder
encoder_previsores = LabelEncoder()
previsores[:,1] = encoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = encoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = encoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = encoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = encoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = encoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = encoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = encoder_previsores.fit_transform(previsores[:,13])

# Padronização dos dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Separando os dados em treinamento e teste
from sklearn.model_selection import train_test_split
prev_treinamento, prev_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.20, random_state = 0)

# Implementação do algoritmo NaiveBayes
from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion = "entropy")
classificador.fit(previsores,classe)

# Testar dados
resultado = classificador.predict(prev_teste)

# Exportar para op graphviz, para visualizar a árvore de decisão
export.export_graphviz( classificador,
                        out_file = "2.arvore.dot", 
                        feature_names = ['age','workclass','final-weight','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loos','hour-per-week','native-country'],
                        class_names = ['<=50K', '>50K'],
                        filled = True,
                        leaves_parallel = True)


# Comparar os dados teste com os previsto pelo algortimo
from sklearn.metrics import confusion_matrix, accuracy_score
print ('Precisão: {:.2f}%'.format(accuracy_score(classe_teste, resultado)*100))
matriz = confusion_matrix(classe_teste, resultado)
print ('Matriz de precisão gerada com sucesso!')
