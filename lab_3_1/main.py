import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  # Для разделения данных
from sklearn.metrics import accuracy_score  # Метрика точности (score)
from sklearn.datasets import make_classification  # Для генерации датасета

# Загружаем датасет Iris
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target  # Целевая переменная
target_names = iris.target_names  # Названия классов

print("Названия сортов:", iris.target_names)

# 1: Рисуем зависимости признаков для всех 3 сортов
# Два графика: sepal (чашелистик) и petal (лепесток)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)  # 1 строка, 2 столбца, 1й элемент
scatter1 = plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['target'], cmap='viridis')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Sepal Length vs Sepal Width')
handles1, _ = scatter1.legend_elements()
legend1 = plt.legend(handles1, target_names, title="Classes")

plt.subplot(1, 2, 2)  # 1 строка, 2 столбца, 2й элемент
scatter2 = plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=df['target'], cmap='viridis')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Petal Length vs Petal Width')
handles2, _ = scatter2.legend_elements()
legend2 = plt.legend(handles2, target_names, title="Classes")

plt.tight_layout()
plt.show()

# По графикам видно, что setosa отделяется от двух других.

# 2: seaborn pairplot
# Рисует графики для всех пар признаков
sns.pairplot(df, hue='target', palette='viridis')  # hue красит точки по таргету
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# 3: Подготовка 2 бинарных датасетов
# 1) setosa (0) vs versicolor (1)
# 2) versicolor (1) vs virginica (2)

# setosa vs versicolor
df_binary1 = df[df['target'] != 2].copy()

# versicolor vs virginica
df_binary2 = df[df['target'] != 0].copy()
df_binary2['target'] = df_binary2['target'].replace({1: 0, 2: 1})

print("\nПодготовлены 2 бинарных датасета.")

print("\nКлассификация: setosa vs versicolor")

# 4 деление данных на обучающую и тестовую выборки
X1 = df_binary1.drop('target', axis=1)
y1 = df_binary1['target']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25,
                                                        random_state=0)

# 5
model1 = LogisticRegression(random_state=0)

# 6
model1.fit(X1_train, y1_train)

# 7 предсказания
y1_pred = model1.predict(X1_test)

# 8 точность модели
accuracy1 = accuracy_score(y1_test, y1_pred)
print(f"Точность модели (setosa vs versicolor) {accuracy1:.4f}")

print("\nКлассификация: versicolor vs virginica")

# 4 деление данных на обучающую и тестовую выборки
X2 = df_binary2.drop('target', axis=1)
y2 = df_binary2['target']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=0)

# 5
model2 = LogisticRegression(random_state=0)

# 6
model2.fit(X2_train, y2_train)

# 7
y2_pred = model2.predict(X2_test)

# 8
accuracy2 = accuracy_score(y2_test, y2_pred)
print(
    f"Точность модели (versicolor vs virginica): {accuracy2 :.4f}")

# 9: Генерация и классификация случайного датасета
print("\nКлассификация случайного датасета")

# 1000 точек, 2 признака, 1 кластер на класс
X_synth, y_synth = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                                       n_informative=2, random_state=1, n_clusters_per_class=1)

plt.figure(figsize=(6, 6))
plt.scatter(X_synth[:, 0], X_synth[:, 1], c=y_synth, cmap='viridis', marker='o',
            edgecolors='k')
plt.title("Сгенерированный датасет")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.show()

# 4
X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(X_synth, y_synth, test_size=0.25,
                                                                            random_state=0)

# 5
model_synth = LogisticRegression(random_state=0)

# 6
model_synth.fit(X_synth_train, y_synth_train)

# 7
y_synth_pred = model_synth.predict(X_synth_test)

# 8 точность
accuracy_synth = accuracy_score(y_synth_test, y_synth_pred)
print(f"Точность модели (случайный датасет): {accuracy_synth:.4f}")
