import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score  # Метрики MAE и R2

# Загрузка данных diabetes. Стандартный датасет для тестов.
diabetes = datasets.load_diabetes()
# Конвертируем в DataFrame. Удобный формат для работы с таблицами.
data_frame = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
data_frame['target'] = diabetes.target  # Целевая переменная

# Выбираем один признак для простой линейной регрессии.
feature_for_X = 'bmi'
X_data = data_frame[[feature_for_X]]
y_data = data_frame['target']

# Сначала делаем с Scikit-Learn. Тут все просто, готовый класс.
sklearn_model = LinearRegression()
sklearn_model.fit(X_data, y_data)  # Обучаем модель
sklearn_slope = sklearn_model.coef_[0]  # Коэф. наклона
sklearn_intercept = sklearn_model.intercept_  # Точка пересечения
sklearn_predictions = sklearn_model.predict(X_data)  # Получаем предсказания


# Реализация метода наименьших квадратов (OLS).
# Считаем наклон и перехват прямой.
def my_linear_regression(x_values, y_values):
    n = len(y_values)
    if n == 0:
        return 0, 0  # Если данных нет

    # Суммы для формул
    sum_x = np.sum(x_values)
    sum_y = np.sum(y_values)
    sum_xy = np.sum(x_values * y_values)
    sum_x_squared = np.sum(x_values ** 2)

    denominator = n * sum_x_squared - sum_x ** 2
    if denominator == 0:
        print("Ошибка: Знаменатель равен нулю. Невозможно вычислить наклон.")
        return None, None

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    return slope, intercept


my_slope, my_intercept = my_linear_regression(X_data.values.flatten(), y_data)
# Считаем предсказания своей модели
if my_slope is not None and my_intercept is not None:
    my_predictions = (my_slope * X_data.values + my_intercept).flatten()
else:
    # Если не посчиталось
    my_predictions = np.full_like(y_data, np.nan)


# Функция для MAPE. Показывает ошибку в процентах.
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Находим ненулевые значения, чтобы избежать деления на ноль
    non_zero_indices = y_true != 0
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]

    if y_true_non_zero.shape[0] == 0:
        return np.nan

        # Формула MAPE
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100


# Считаем метрики для Scikit-Learn модели
mae_sklearn = mean_absolute_error(y_data, sklearn_predictions)
r2_sklearn = r2_score(y_data, sklearn_predictions)
mape_sklearn = mean_absolute_percentage_error(y_data, sklearn_predictions)

print("\nМетрики (Scikit-Learn модель):")
print(f"  MAE: {mae_sklearn:.2f}")
print(f"  R2: {r2_sklearn:.2f}")
print(f"  MAPE: {mape_sklearn:.2f}%")

# Считаем метрики для нашей модели, если она посчиталась
if my_slope is not None and my_intercept is not None:
    mae_my = mean_absolute_error(y_data, my_predictions)

    # R2 для своей модели считаем вручную
    # R2 = 1 - (SS_res / SS_tot)
    y_mean = np.mean(y_data)
    ss_total = np.sum((y_data - y_mean) ** 2)
    ss_residual = np.sum((y_data - my_predictions) ** 2)
    r2_my = 1 - (ss_residual / ss_total) if ss_total != 0 else 0

    mape_my = mean_absolute_percentage_error(y_data, my_predictions)

    print("\nМетрики (Моя модель):")
    print(f"  MAE: {mae_my:.2f}")
    print(f"  R2: {r2_my:.2f}")
    print(f"  MAPE: {mape_my:.2f}%")

else:
    print("\nМоя модель не была рассчитана из-за ошибки.")

plt.figure(figsize=(10, 6))
plt.scatter(X_data, y_data, label='Реальные данные')  # Точки данных
plt.xlabel(feature_for_X)
plt.ylabel('Прогрессирование диабета')
plt.title('Линейная регрессия')

# Линия Scikit-Learn
plt.plot(X_data, sklearn_predictions, color='red', label=f'Scikit-Learn (R2={r2_sklearn:.2f})')

# Линия своей модели
if my_slope is not None and my_intercept is not None:
    plt.plot(X_data, my_predictions, color='green', linestyle='--',
             label=f'My (R2={r2_my:.2f})')  # Пунктир

plt.legend()
plt.grid(True)
plt.show()

# Выводим начало таблицы с предсказаниями Scikit-Learn для сравнения
results_df = pd.DataFrame({
    feature_for_X: data_frame[feature_for_X],
    'Actual': y_data,
    'Predicted (Scikit-Learn)': sklearn_predictions
})

print("\nПример предсказаний (Scikit-Learn):")
print(results_df.head())

# Ну вот как-то так. Сравнили две реализации.
# По идее, результаты должны быть очень близки.
# Это хороший пример для понимания основ машинного обучения и анализа данных.
# Можно поискать еще про "linear regression python", "OLS algorithm", "regression metrics".
