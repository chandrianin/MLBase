import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target  # Целевая переменная
target_names = iris.target_names  # Названия сортов для легенды

print("Названия сортов для классов 0, 1, 2:", target_names)

# Многоклассовая логистическая регрессия и отрисовка решения

# Тольк два признака: petal length и petal width.
X = df[['petal length (cm)', 'petal width (cm)']]  # Признаки
y = df['target']  # Целевая переменная

model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
model.fit(X.values, y)

print("\nМодель логистической регрессии обучена")

x_min, x_max = X.iloc[:, 0].min() - .5, X.iloc[:, 0].max() + .5  # Для petal length
y_min, y_max = X.iloc[:, 1].min() - .5, X.iloc[:, 1].max() + .5  # Для petal width

# Сетка точек
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Предсказания класса для каждой точки в сетке
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Перевод предсказанных классов обратно в сетку
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
contour = plt.contourf(xx, yy, Z, cmap="viridis", alpha=0.8)
scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap="viridis", edgecolors='k', s=20)

plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Области решения логистической регрессии (Iris)')
handles, _ = scatter.legend_elements()
plt.legend(handles, target_names, title="Classes")

plt.show()

# setosa-область четко отделена
# Граница между versicolor и virginica проходит где-то посередине
# Также есть точки одного класса в области другого - ошибки модели
