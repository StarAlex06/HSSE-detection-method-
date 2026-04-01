# 2_find_thresholds.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import extract_style_features, extract_semantic_features, extract_perplexity_features, \
    extract_stability_features
from tqdm import tqdm
import itertools

# --- Загрузка моделей и scaler'ов ---
print("Загрузка моделей...")
model_style = joblib.load('model_style.pkl')
model_semantic = joblib.load('model_semantic.pkl')
model_perplexity = joblib.load('model_perplexity.pkl')
model_stability = joblib.load('model_stability.pkl')

scaler_style = joblib.load('scaler_style.pkl')
scaler_semantic = joblib.load('scaler_semantic.pkl')
scaler_perplexity = joblib.load('scaler_perplexity.pkl')
scaler_stability = joblib.load('scaler_stability.pkl')

# --- Загрузка валидационных данных ---
print("Загрузка val.csv...")
df_val = pd.read_csv('val.csv')
texts_val = df_val['text'].tolist()
y_val = df_val['label'].values

# --- Извлечение признаков для валидации ---
print("Извлечение признаков для валидации...")
X_style_val = np.array([extract_style_features(t) for t in texts_val])
X_semantic_val = np.array([extract_semantic_features(t) for t in texts_val])
X_perplexity_val = np.array([extract_perplexity_features(t) for t in texts_val])
X_stability_val = np.array([extract_stability_features(t) for t in texts_val])

# Масштабирование
X_style_val_scaled = scaler_style.transform(X_style_val)
X_semantic_val_scaled = scaler_semantic.transform(X_semantic_val)
X_perplexity_val_scaled = scaler_perplexity.transform(X_perplexity_val)
X_stability_val_scaled = scaler_stability.transform(X_stability_val)

# --- Получение вероятностей ---
print("Получение предсказаний...")
p_style = model_style.predict_proba(X_style_val_scaled)[:, 1]  # Вероятность ИИ
p_semantic = model_semantic.predict_proba(X_semantic_val_scaled)[:, 1]
p_perplexity = model_perplexity.predict_proba(X_perplexity_val_scaled)[:, 1]
p_stability = model_stability.predict_proba(X_stability_val_scaled)[:, 1]

# --- Поиск оптимальных порогов ---
# Для ускорения будем перебирать не все комбинации, а оптимизировать последовательно
# Но для наглядности сделаем упрощенный вариант

print("Поиск оптимальных порогов (это может занять минуту)...")

best_thresholds = {'style': 0.5, 'semantic': 0.5, 'perplexity': 0.5, 'stability': 0.5}
best_f1 = 0
best_precision = 0

# Сетка порогов для перебора
thresholds = np.arange(0.3, 0.9, 0.05)

# Перебираем все комбинации? Их может быть много (12^4 = 20736)
# Для простоты переберем с помощью циклов (можно заменить на оптимизатор)
for t1 in tqdm(thresholds, desc="Стилометрия"):
    for t2 in thresholds:
        for t3 in thresholds:
            for t4 in thresholds:

                # Логика "ИЛИ": текст - ИИ, если хотя бы одна модель уверена выше порога
                final_pred = (p_style > t1) | (p_semantic > t2) | (p_perplexity > t3) | (p_stability > t4)

                # Считаем метрики
                current_f1 = f1_score(y_val, final_pred)
                current_precision = precision_score(y_val, final_pred)

                # Мы хотим максимизировать F1, но можно добавить условие на минимальную точность
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_precision = current_precision
                    best_thresholds = {
                        'style': t1,
                        'semantic': t2,
                        'perplexity': t3,
                        'stability': t4
                    }

print("\n" + "=" * 50)
print("НАЙДЕННЫЕ ПОРОГИ:")
print(f"  Стилометрия: {best_thresholds['style']:.2f}")
print(f"  Семантика: {best_thresholds['semantic']:.2f}")
print(f"  Перплексия: {best_thresholds['perplexity']:.2f}")
print(f"  Стабильность: {best_thresholds['stability']:.2f}")
print(f"Метрики на валидации:")
print(f"  F1 Score: {best_f1:.3f}")
print(f"  Precision: {best_precision:.3f}")
print("=" * 50)

# Сохраняем пороги
import json

with open('thresholds.json', 'w') as f:
    json.dump(best_thresholds, f)

print("Пороги сохранены в thresholds.json")