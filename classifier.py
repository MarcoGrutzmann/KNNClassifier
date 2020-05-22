import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from adpy_shared_utilities import plot_fruit_knn #Biblioteca do curso da Coursera

fruits = pd.read_table('fruit_data_with_colors.txt')

# Vamos dar uma olhada inicial nos dados
print(fruits.head())

#Não é comum vermos isso, mas é bem importante ver o tipo de dados (as vezes pandas muda o tipo do seu dataset na importação, o que pode trazer problemas)
print(fruits.dtypes)

# Criar scatter matrix
X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) #Separar dataset para realizar treinamento

# Faz mapeamento do nome dafruta (label) para facilitar 
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   
print(lookup_fruit_name)

# Vemos dar uma visualizada nos dados!
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()

# Criar separação da base para teste

X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

# padrão é 75% / 25% train-test split (separação de base de treino)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Criar objeto para classificação 
knn = KNeighborsClassifier(n_neighbors = 5)

#treinar o classificador

print(knn.fit(X_train, y_train))

#usar base de teste para estimar acurárcia dos dados
acc = knn.score(X_test, y_test)
print("\n Estimativa de acurácia: {}".format(acc))

# Verificar resultados com um input novo
fruit_prediction = knn.predict([[20, 4.3, 5.5]])
print("Primeiro teste")
print(fruit_prediction)
print(lookup_fruit_name[fruit_prediction[0]])

# segundo teste
print("Segundo teste")
fruit_prediction = knn.predict([[100, 6.3, 8.5]])
print(fruit_prediction)
print(lookup_fruit_name[fruit_prediction[0]])

# Plotar as fronteiras de decisão do Knn
plot_fruit_knn(X_train, y_train, 5, 'uniform')  

# Testando a sensibilidade ddos parâmetros do KNN
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])

# Testando acurácia do KNN com split de base para teste
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')