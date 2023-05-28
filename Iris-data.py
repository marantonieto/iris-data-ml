import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Adicionando nomes as colunas

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'specie']

# Lendo a database
df = pd.read_csv('iris.data', names=columns)

# garantindo que qualquer database inserido no modelo não terá nenhum valor nulo
if df.isnull().values.any():
    df.interpolate()

# Plotando a relação das variáveis independentes com a variável alvo

for label in columns[:-1]:
    plt.hist(df[df['specie'] == 'Iris-setosa'][label], color='blue', density=True, alpha=0.7, label='setosa')
    plt.hist(df[df['specie'] == 'Iris-virginica'][label], color='red', density=True, alpha=0.7, label='virginica')
    plt.hist(df[df['specie'] == 'Iris-versicolor'][label], color='green', density=True, alpha=0.7, label='versicolour')
    plt.ylabel(label)
    plt.xlabel('incidência')
    plt.legend()
    plt.show()


# Preparando os dados e fazendo a verificação cruzada

X = df.drop('specie', axis=1)
y = df['specie']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=4)

# tentativa de fazer a verificação cruzada apenas com o numpy e o pandas, possível, porém não é simples de controlar a
# randomização das amostras (verificação cruzada) o que faz com que o modelo resulte vários resultados diferentes
# em execuções diferentes.

"""
train, test = np.split(df.sample(frac=1), [int(len(df) * 0.8)])

train_X = train[train.columns[:-1]].values
train_y = train[train.columns[-1]].values

test_X = test[test.columns[:-1]].values
test_y = test[test.columns[-1]].values
"""

# Aplicação do modelo KNN (classificação)

# a escolha do k = 20 foi feita para aumentar a precisão da classificação.
knn_model = KNeighborsClassifier(n_neighbors=20)
knn_model.fit(train_X, train_y)

y_predict = knn_model.predict(test_X)

print(classification_report(test_y, y_predict))
