import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# Загрузка набора данных diabetes
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target  # Целевая переменная - что предсказываем

print("Описание набора данных:")
print(diabetes.DESCR)

print("\nПервые 5 строк данных:")
print(df.head())

print("\nСтатистическая информация:")
print(df.describe())

correlation_matrix = df.corr()
print("\nМатрица корелляции:")
print(correlation_matrix['target'].sort_values(ascending=False))

X_column = 'bmi'
X = df[[X_column]]  # Признак
y = df['target']  # Цель

# Реализация метода линейной регрессии
model_sklearn = LinearRegression()
model_sklearn.fit(X, y)  # Обучение модели
sklearn_slope = model_sklearn.coef_[0]  # Коэффициент наклона
sklearn_intercept = model_sklearn.intercept_  # Точка пересечения Y

print(f"\nКоэффициенты линейной регрессии (Scikit-Learn):")
print(f"  Угловой коэффицент (slope): {sklearn_slope:.4f}")
print(f"  Свободный член (intercept): {sklearn_intercept:.4f}")


# Реализация собственного алгоритма метода наименьших квадратов.
def custom_linear_regression(X, y):
    n = len(y)
    if n == 0:
        return 0, 0

    sum_x = np.sum(X[X.columns[0]])
    sum_y = np.sum(y)
    sum_xy = np.sum(X.values.flatten() * y)
    sum_x_squared = np.sum(X[X.columns[0]] ** 2)

    denominator = n * sum_x_squared - sum_x ** 2
    if denominator == 0:
        print("Ошибка: Знаменатель равен нулю. Невозможно вычислить наклон.")
        return None, None

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


slope_custom, intercept_custom = custom_linear_regression(X, y)

print(f"\nКоэффициенты линейной регрессии (собственный алгоритм):")
print(f"  Угловой коэффициент (slope): {slope_custom:.4f}")
print(f"  Свободный член (intercept): {intercept_custom :.4f}")

# Отрисовка данных и регрессионной прямой.
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Исходные данные')
plt.xlabel(X_column)
plt.ylabel('Прогрессирование диабета')
plt.title('Линейная регрессия (Diabetes Dataset)')

# Отрисовка прямой Scikit-Learn
y_pred_sklearn = model_sklearn.predict(X)
plt.plot(X, y_pred_sklearn, color='red',
         label=f'Регрессия (Scikit-Learn): y = {sklearn_slope:.2f}x + {sklearn_intercept:.2f}')

# Отрисовка прямой собственного алгоритма
if slope_custom is not None and intercept_custom is not None:
    y_pred_custom = slope_custom * X + intercept_custom
    plt.plot(X, y_pred_custom, color='green', linestyle='--',
             label=f'Регрессия (Custom): y = {slope_custom:.2f}x + {intercept_custom:.2f}')

plt.legend()
plt.grid(True)
plt.show()

# Вывод таблицы с результатами предсказаний (из Scikit-Learn).
predictions_sklearn = model_sklearn.predict(X)
results_df = pd.DataFrame({
    X_column: df[X_column],
    'Actual': y,
    'Predicted (Scikit-Learn)': predictions_sklearn  # Используем предсказания скилерна
})

print("\nТаблица с результатами предсказаний (Scikit-Learn):")
print(results_df.head())
