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
