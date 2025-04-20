import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Функция для вывода базовой статистики по выбранным колонкам.
def display_statistics(df, x_col, y_col):
    # Печатаем количество, мин, макс, среднее для X и Y.
    print(f"Статистика для столбца '{x_col}'")
    print(f"  Количество: {df[x_col].count()}")
    print(f"  Минимум: {df[x_col].min()}")
    print(f"  Максимум: {df[x_col].max()}")
    print(f"  Среднее: {df[x_col].mean()}")

    print(f"\nСтатистика для столбца '{y_col}':")
    print(f"  Количество: {df[y_col].count()}")
    print(f"  Минимум: {df[y_col].min()}")
    print(f"  Максимум: {df[y_col].max()}")
    print(f"  Среднее: {df[y_col].mean()}")


# Рисуем исходные точки данных на графике.
def plot_data_points(df, x_col, y_col):
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Исходные данные")
    return fig, ax


# Реализация метода наименьших квадратов (OLS).
# Считаем наклон и перехват прямой.
def least_squares(df, x_col, y_col):
    x = df[x_col]
    y = df[y_col]
    n = len(x)
    if n == 0:
        return 0, 0  # Если данных нет

    # Суммы для формул
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x ** 2)

    denominator = n * sum_x_squared - sum_x ** 2
    if denominator == 0:
        print("Ошибка: Знаменатель равен нулю. Невозможно вычислить наклон.")
        return None, None

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


# Рисуем регрессионную прямую на уже созданном графике.
def plot_regression_line(fig, ax, df, x_col, y_col, slope, intercept):
    if slope is not None and intercept is not None:
        x_min = df[x_col].min()
        x_max = df[x_col].max()
        y_min = slope * x_min + intercept
        y_max = slope * x_max + intercept
        ax.plot([x_min, x_max], [y_min, y_max], color='red', label='Линейная регресси')
        ax.legend()
        ax.set_title("Исходные данные с линией регрессии")


# Рисуем квадраты ошибок на графике
def plot_error_squares(fig, ax, df, x_col, y_col, slope, intercept):
    if slope is not None and intercept is not None:
        for index, row in df.iterrows():
            x_val = row[x_col]
            y_actual = row[y_col]
            y_predicted = slope * x_val + intercept
            error = y_actual - y_predicted
            # Нижний левый угол, ширина=высота=|ошибка|, цвет, прозрачность
            ax.add_patch(plt.Rectangle((x_val, min(y_actual, y_predicted)), abs(error), abs(error),
                                       fill=True, color='lightcoral', alpha=0.5))
        ax.set_title("Исходные данные с линией регрессии и квадратами ошибок")


file_path = input("Введите путь к CSV файлу: ")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Файл не найден.")
    exit()

print("Доступные столбцы:", df.columns.tolist())
x_column = input("Выберите столбец для X: ")
y_column = input("Выберите столбец для Y: ")

if x_column in df.columns and y_column in df.columns:
    display_statistics(df, x_column, y_column)

    # Изображение исходных точек
    fig1, ax1 = plot_data_points(df, x_column, y_column)

    # Вычисляем параметры регрессионной прямой
    slope, intercept = least_squares(df, x_column, y_column)
    if slope is not None and intercept is not None:
        print(f"\nПараметры регрессионной прямой:")
        print(f"  Угловой коэффициент (slope): {slope:.4f}")
        print(f"  Свободный член (intercept): {intercept:.4f}")

        # Изображение исходных точек с линией регрессии
        fig2, ax2 = plot_data_points(df, x_column, y_column)
        plot_regression_line(fig2, ax2, df, x_column, y_column, slope, intercept)

        # Изображение исходных точек с линией регрессии и квадратами ошибок
        fig3, ax3 = plot_data_points(df, x_column, y_column)
        plot_regression_line(fig3, ax3, df, x_column, y_column, slope, intercept)
        plot_error_squares(fig3, ax3, df, x_column, y_column, slope, intercept)

        # Отображение всех графиков одновременно
        plt.show()

else:
    print("\nОшибка: Один или оба выбранных столбца не найдены в файле.")
