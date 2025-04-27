import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

try:
    df = pd.read_csv("diabetes.csv")
    print("Датасет успешно загружен")
except Exception as e:
    print(f"Ошибка при загрузке датасета: {e}")
    exit()

# обрабатываем нули как пропуски и удаляем строки
cols_with_zeros_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Заменяем нули на NaN в указанных колонках
df[cols_with_zeros_as_missing] = df[cols_with_zeros_as_missing].replace(0, np.nan)

initial_rows = len(df)  # Сколько строк было до удаления
df.dropna(inplace=True)
final_rows = len(df)  # Сколько строк осталось

print(f"\nВсего строк до чистки: {initial_rows}")
print(f"Строк после чистки: {final_rows}")
print(
    f"Процент потерянных данных после удаления строк с пропусками: "
    f"{((initial_rows - final_rows) / initial_rows) * 100:.2f}%")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nДанные разделены: Обучающая ({len(X_train)} строк), Тестовая ({len(X_test)} строк).")

print("\nСлучайный лес")
model_rf_std = RandomForestClassifier(random_state=0)
start_time_rf_std = time.time()
model_rf_std.fit(X_train, y_train)
end_time_rf_std = time.time()

y_pred_rf_std = model_rf_std.predict(X_test)
y_pred_proba_rf_std = model_rf_std.predict_proba(X_test)[:, 1]

accuracy_rf_std = accuracy_score(y_test, y_pred_rf_std)
precision_rf_std = precision_score(y_test, y_pred_rf_std)
recall_rf_std = recall_score(y_test, y_pred_rf_std)
f1_rf_std = f1_score(y_test, y_pred_rf_std)
roc_auc_rf_std = roc_auc_score(y_test, y_pred_proba_rf_std)

print(f"Время обучения: {end_time_rf_std - start_time_rf_std:.4f} сек")
print(f"Accuracy: {accuracy_rf_std:.4f}")
print(f"Precision: {precision_rf_std:.4f}")
print(f"Recall: {recall_rf_std:.4f}")
print(f"F1-score: {f1_rf_std:.4f}")
print(f"ROC AUC: {roc_auc_rf_std:.4f}")

depths = range(1, 11)
train_accuracies_depth = []
test_accuracies_depth = []

for depth in depths:
    model_rf = RandomForestClassifier(max_depth=depth, random_state=0)
    model_rf.fit(X_train, y_train)

    y_pred_train = model_rf.predict(X_train)
    y_pred_test = model_rf.predict(X_test)

    train_accuracies_depth.append(accuracy_score(y_train, y_pred_train))
    test_accuracies_depth.append(accuracy_score(y_test, y_pred_test))

# График accuracy от max_depth
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies_depth, label='На обучающей выборке')
plt.plot(depths, test_accuracies_depth, label='На тестовой выборке')
plt.xlabel("Максимальная глубина дерева в лесу (max_depth)")
plt.ylabel("Accuracy")
plt.title("Accuracy случайного леса в зависимости от max_depth")
plt.legend()
plt.grid(True)
plt.show()

# Как влияет случайный выбор признаков на каждом узле
n_features_range = range(1, X.shape[1] + 1)  # Пробуем от 1 до всех признаков
train_accuracies_features = []
test_accuracies_features = []

fixed_depth = 3
print(f"(Используется глубина дерева = {fixed_depth})")

for max_feat in n_features_range:
    model_rf = RandomForestClassifier(max_depth=fixed_depth, max_features=max_feat, random_state=0)
    model_rf.fit(X_train, y_train)

    y_pred_train = model_rf.predict(X_train)
    y_pred_test = model_rf.predict(X_test)

    train_accuracies_features.append(accuracy_score(y_train, y_pred_train))
    test_accuracies_features.append(accuracy_score(y_test, y_pred_test))

# График accuracy от max_features
plt.figure(figsize=(10, 6))
plt.plot(n_features_range, train_accuracies_features, label='На обучающей выборке')
plt.plot(n_features_range, test_accuracies_features, label='На тестовой выборке')
plt.xlabel("Количество признаков на разбиение (max_features)")
plt.ylabel("Accuracy")
plt.title("Accuracy случайного леса в зависимости от max_features")
plt.legend()
plt.grid(True)
plt.show()

# Как влияет количество деревьев в лесу на качество и время обучения
n_estimators_range = range(10, 201, 10)  # Пробуем количество деревьев от 10 до 200 с шагом 10
train_accuracies_estimators = []
test_accuracies_estimators = []
training_times = []

fixed_depth_for_n = 3
fixed_features_for_n = 7
print(f"(Используется параметры: max_depth={fixed_depth_for_n}, max_features={fixed_features_for_n})")

for n_est in n_estimators_range:
    model_rf = RandomForestClassifier(max_depth=fixed_depth_for_n, max_features=fixed_features_for_n,
                                      n_estimators=n_est, random_state=0)

    start_time = time.time()
    model_rf.fit(X_train, y_train)
    end_time = time.time()
    training_times.append(end_time - start_time)

    y_pred_train = model_rf.predict(X_train)
    y_pred_test = model_rf.predict(X_test)

    train_accuracies_estimators.append(accuracy_score(y_train, y_pred_train))
    test_accuracies_estimators.append(accuracy_score(y_test, y_pred_test))

# График accuracy от n_estimators
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_accuracies_estimators, label='На обучающей выборке')
plt.plot(n_estimators_range, test_accuracies_estimators, label='На тестовой выборке')
plt.xlabel("Количество деревьев (n_estimators)")
plt.ylabel("Accuracy")
plt.title("Accuracy случайного леса в зависимости от n_estimators")
plt.legend()
plt.grid(True)
plt.show()

# График времени обучения от n_estimators
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, training_times)
plt.xlabel("Количество деревьев (n_estimators)")
plt.ylabel("Время обучения (сек)")
plt.title("Время обучения случайного леса в зависимости от n_estimators")
plt.grid(True)
plt.show()

print("\nXGBoost")
xgb_n_estimators = 100
xgb_max_depth = 7
xgb_learning_rate = 0.9
xgb_random_state = 0

print(
    f"(Параметры XGBoost: n_estimators={xgb_n_estimators}, max_depth={xgb_max_depth}, learning_rate={xgb_learning_rate})")

model_xgb = xgb.XGBClassifier(
    n_estimators=xgb_n_estimators,
    max_depth=xgb_max_depth,
    learning_rate=xgb_learning_rate,
    eval_metric='logloss',
    random_state=xgb_random_state
)

start_time_xgb = time.time()
model_xgb.fit(X_train, y_train)
end_time_xgb = time.time()

y_pred_xgb = model_xgb.predict(X_test)
y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"Время обучения: {end_time_xgb - start_time_xgb:.4f} сек")
print(f"Accuracy: {accuracy_xgb:.4f}")
print(f"Precision: {precision_xgb:.4f}")
print(f"Recall: {recall_xgb:.4f}")
print(f"F1-score: {f1_xgb:.4f}")
print(f"ROC AUC: {roc_auc_xgb:.4f}")

print("\nСравнение XGBoost и  случайного леса")

print("Случайный лес:")
print(f"  Время обучения: {end_time_rf_std - start_time_rf_std:.4f} сек")
print(f"  Accuracy: {accuracy_rf_std:.4f}")
print(f"  F1-score: {f1_rf_std:.4f}")
print(f"  ROC AUC: {roc_auc_rf_std:.4f}")

print("\nXGBoost:")
print(f"  Время обучения: {end_time_xgb - start_time_xgb:.4f} сек")
print(f"  Accuracy: {accuracy_xgb:.4f}")
print(f"  F1-score: {f1_xgb:.4f}")
print(f"  ROC AUC: {roc_auc_xgb:.4f}")
