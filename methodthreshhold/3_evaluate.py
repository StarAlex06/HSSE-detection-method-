# 3_evaluate.py
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from utils import extract_style_features, extract_semantic_features, extract_perplexity_features, extract_stability_features

# --- Загрузка всего ---
print("Загрузка моделей и порогов...")
model_style = joblib.load('model_style.pkl')
model_semantic = joblib.load('model_semantic.pkl')
model_perplexity = joblib.load('model_perplexity.pkl')
model_stability = joblib.load('model_stability.pkl')

scaler_style = joblib.load('scaler_style.pkl')
scaler_semantic = joblib.load('scaler_semantic.pkl')
scaler_perplexity = joblib.load('scaler_perplexity.pkl')
scaler_stability = joblib.load('scaler_stability.pkl')

with open('thresholds.json', 'r') as f:
    thresholds = json.load(f)

# --- Загрузка тестовых данных ---
print("Загрузка test.csv...")
df_test = pd.read_csv('test.csv', sep=';')
texts_test = df_test['text'].tolist()
y_test = df_test['label'].values

# --- Извлечение признаков ---
print("Извлечение признаков...")
X_style_test = np.array([extract_style_features(t) for t in texts_test])
X_semantic_test = np.array([extract_semantic_features(t) for t in texts_test])
X_perplexity_test = np.array([extract_perplexity_features(t) for t in texts_test])
X_stability_test = np.array([extract_stability_features(t) for t in texts_test])

# Масштабирование
X_style_test_scaled = scaler_style.transform(X_style_test)
X_semantic_test_scaled = scaler_semantic.transform(X_semantic_test)
X_perplexity_test_scaled = scaler_perplexity.transform(X_perplexity_test)
X_stability_test_scaled = scaler_stability.transform(X_stability_test)

# --- Предсказания ---
p_style = model_style.predict_proba(X_style_test_scaled)[:, 1]
p_semantic = model_semantic.predict_proba(X_semantic_test_scaled)[:, 1]
p_perplexity = model_perplexity.predict_proba(X_perplexity_test_scaled)[:, 1]
p_stability = model_stability.predict_proba(X_stability_test_scaled)[:, 1]

# Логика "ИЛИ" с найденными порогами
final_pred = (
    (p_style > thresholds['style']) |
    (p_semantic > thresholds['semantic']) |
    (p_perplexity > thresholds['perplexity']) |
    (p_stability > thresholds['stability'])
).astype(int)

# --- Оценка ---
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, final_pred):.3f}")
print(f"Precision: {precision_score(y_test, final_pred):.3f}")
print(f"Recall:    {recall_score(y_test, final_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, final_pred):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_pred))
print("\nClassification Report:")
print(classification_report(y_test, final_pred, target_names=['Человек', 'ИИ']))