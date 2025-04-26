import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, \
    precision_recall_curve  # Метрики
from sklearn import tree

try:
    df = pd.read_csv("diabetes.csv")
    print("Датасет успешно загружен")
except Exception as e:
    print(f"Ошибка при загрузке датасета: {e}")
    exit()

print("\nПервые 5 строк данных:")
print(df.head())

print("\nСтатистика данных:")
print(df.describe())

initial_rows = len(df)

# Некоторые колонки датасета содержат нули, что логически не может являться верными значениям,
# поэтому эти строки будут удалены
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

df.dropna(inplace=True)

# Процент потерянных данных
final_rows = len(df)
lost_rows = initial_rows - final_rows
lost_percentage = (lost_rows / initial_rows) * 100
print(f"\nВсего строк до чистки: {initial_rows}")
print(f"Строк после чистки: {final_rows}")
print(f"Процент потерянных данных: {lost_percentage:.2f} %")

X = df.drop('Outcome', axis=1)  # Признаки
y = df['Outcome']  # Целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nДанные разделены: Обучающая ({len(X_train)} строк), Тестовая ({len(X_test)} строк).")

# Сравнение логистической регрессии и решающего дерева

# Логистическая Регрессия
print("\nЛогистическая Регрессия")
model_lr = LogisticRegression(random_state=0, max_iter=400)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
y_pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]  # Для ROC/PR

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print(f"F1-score: {f1_lr:.4f}")

# Решающее дерево
print("\nРешающее Дерево")
model_dt_std = DecisionTreeClassifier(random_state=0)
model_dt_std.fit(X_train, y_train)
y_pred_dt_std = model_dt_std.predict(X_test)
y_pred_proba_dt_std = model_dt_std.predict_proba(X_test)[:, 1]  # Для ROC/PR

accuracy_dt_std = accuracy_score(y_test, y_pred_dt_std)
precision_dt_std = precision_score(y_test, y_pred_dt_std)
recall_dt_std = recall_score(y_test, y_pred_dt_std)
f1_dt_std = f1_score(y_test, y_pred_dt_std)

print(f"Accuracy: {accuracy_dt_std:.4f}")
print(f"Precision: {precision_dt_std:.4f}")
print(f"Recall: {recall_dt_std:.4f}")
print(f"F1-score: {f1_dt_std:.4f}")

# Сравнивая полученные результаты, стоит обратить внимание на Recall, которой играет важную роль в нашем случае
# LR: 0.5926 ; DT: 0.667
# Отсюда можно сделать вывод, что решающее дерево справилось с задачей нахождения больных диабетом людей лучше
# В нашей ситуации это наиболее критичный показатель


# Исследование метрики в зависимости от глубины дерева

# Выберем метрику Accuracy
depths = range(1, 20)  # Глубины от 1 до 19
train_accuracies = []
test_accuracies = []

for depth in depths:
    model_dt = DecisionTreeClassifier(max_depth=depth, random_state=0)
    model_dt.fit(X_train, y_train)

    y_pred_train = model_dt.predict(X_train)
    y_pred_test = model_dt.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# График зависимости метрики от глубины
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, label='На обучающей выборке')
plt.plot(depths, test_accuracies, label='На тестовой выборке')
plt.xlabel("Максимальная глубина дерева")
plt.ylabel("Accuracy")
plt.title("Accuracy в зависимости от глубины решающего дерева")
plt.legend()
plt.grid(True)
plt.show()

# По графику видно, что оранжевая линия (тестовых данных) достигает своего максимума при глубине 5

optimal_depth = 5

model_dt_optimal = DecisionTreeClassifier(max_depth=optimal_depth, random_state=0)
model_dt_optimal.fit(X_train, y_train)

# Экспортирован файл "diabetes_tree_optimal_depth" для генерации изображения дерева через graphviz
dot_data = tree.export_graphviz(model_dt_optimal, out_file=None,
                                feature_names=X.columns.tolist(),
                                class_names=['Нет диабета', 'Диабет'],
                                filled=True, rounded=True, special_characters=True)

# Какие признаки модель использовала чаще всего при принятии решений.
importances = model_dt_optimal.feature_importances_
feature_names = X.columns.tolist()

# Пары (важность, имя признака), сортируем по важности
feature_importance_pairs = sorted(zip(importances, feature_names), reverse=True)
sorted_importances = [pair[0] for pair in feature_importance_pairs]
sorted_feature_names = [pair[1] for pair in feature_importance_pairs]

plt.figure(figsize=(10, 6))
plt.bar(sorted_feature_names, sorted_importances)
plt.xticks(rotation=90)
plt.title("Важность признаков")
plt.ylabel("Важность")
plt.show()

# PR/ROC

y_pred_dt_optimal = model_dt_optimal.predict(X_test)
y_pred_proba_dt_optimal = model_dt_optimal.predict_proba(X_test)[:, 1]

# PR
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba_dt_optimal)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR")
plt.grid(True)
plt.show()

# ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_dt_optimal)
auc_roc = roc_auc_score(y_test, y_pred_proba_dt_optimal)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.', label=f'AUC = {auc_roc:.4f}')
plt.xlabel("FPT")
plt.ylabel("TPR")
plt.title("ROC")
plt.legend()
plt.grid(True)
plt.show()
