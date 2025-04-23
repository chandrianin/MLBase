import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # Чтобы поделить данные на обучение и тест
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  # Метрика, чтобы узнать, насколько модель точная

try:
    df = pd.read_csv("Titanic.csv")
except Exception as e:
    print(f"Ошибка при загрузке датасета: {e}")
    exit()

# Данные до чистки
print("\nПервые 5 строк до предобработки:")
print(df.head())

# Предобработка данных

# Сколько строк было изначально
initial_rows = len(df)

# Удаляем строки с пропусками
df.dropna(inplace=True)

# Удаляем столбцы: 'PassengerId', 'Name', 'Ticket', 'Cabin'
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df.drop(columns=columns_to_drop, inplace=True)

# Перекодируем 'Sex', 'Embarked' в числовые
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 'Embarked': 'C', 'Q', 'S' -> 1, 2, 3
df['Embarked'] = df['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

# Данные после чистки
print("\nПервые 5 строк после предобработки:")
print(df.head())

# Процент потерянных данных
final_rows = len(df)
lost_rows = initial_rows - final_rows
lost_percentage = (lost_rows / initial_rows) * 100
print(f"\nВсего строк до чистки: {initial_rows}")
print(f"Строк после чистки: {final_rows}")
print(f"Процент потерянных данных: {lost_percentage:.2f} %")

X = df.drop('Survived', axis=1)  # Признаки
y = df['Survived']  # Целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nДанные разделены на обучающую ({len(X_train)} строк) и тестовую ({len(X_test)} строк) выборки")

model = LogisticRegression(random_state=0, max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели на тестовой выборке: {accuracy:.3f}")

print("\nОценка влияния признака 'Embarked':")

X_no_embarked = df.drop(['Survived', 'Embarked'], axis=1)
X_no_embarked_train, X_no_embarked_test, y_train, y_test = train_test_split(X_no_embarked, y, test_size=0.2,
                                                                            random_state=42)

model_no_embarked = LogisticRegression(random_state=0, max_iter=200)
model_no_embarked.fit(X_no_embarked_train, y_train)

y_pred_no_embarked = model_no_embarked.predict(X_no_embarked_test)
accuracy_no_embarked = accuracy_score(y_test, y_pred_no_embarked)

print(f"Точность модели без признака 'Embarked': {accuracy_no_embarked:.3f}")

# Сравниваем точность с Embarked и без.
print("Разница в точности:", accuracy - accuracy_no_embarked)
