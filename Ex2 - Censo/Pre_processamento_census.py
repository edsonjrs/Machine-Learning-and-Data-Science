import pandas as pd

base = pd.read_csv("census.csv")

base.describe()

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:,14].values

# Transformação de variáveis categóricas para numériccas
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

# Binarização de varivéis categóricas
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

# Padronização dos dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Separando os dados em treinamento e teste
from sklearn.model_selection import train_test_split
prev_treinamento, prev_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25, random_state = 0)


