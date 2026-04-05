# 1_train_models.py (GPU версия)
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import extract_style_features, extract_stability_features
import warnings

warnings.filterwarnings('ignore')

# --- Определяем устройство ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Доступно памяти: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- Загрузка данных ---
print("Загрузка train.csv...")
df = pd.read_csv('train.csv')
texts = df['text'].tolist()
y = df['label'].values

# --- 1. Стилометрия и Стабильность ---
print("Извлечение стилометрии и стабильности...")
X_style = []
X_stability = []
for text in tqdm(texts, desc="Стилометрия/стабильность"):
    X_style.append(extract_style_features(text))
    X_stability.append(extract_stability_features(text))
X_style = np.array(X_style)
X_stability = np.array(X_stability)

# --- 2. Семантика (на GPU, батчами) ---
from utils import extract_semantic_batch


print("Извлечение семантики...")
X_semantic = extract_semantic_batch(texts)


# --- 3. Перплексия (на GPU, батчами) ---
print("Загрузка модели перплексии на GPU...")

from utils import extract_perplexity_batch
print("Вычисление перплексии...")
X_perplexity = extract_perplexity_batch(texts, batch_size=8)

# --- Освобождаем GPU память ---
torch.cuda.empty_cache()

# --- Масштабирование и обучение моделей  ---
print("Масштабирование признаков...")
scaler_style = StandardScaler().fit(X_style)
scaler_semantic = StandardScaler().fit(X_semantic)
scaler_perplexity = StandardScaler().fit(X_perplexity)
scaler_stability = StandardScaler().fit(X_stability)

X_style_scaled = scaler_style.transform(X_style)
X_semantic_scaled = scaler_semantic.transform(X_semantic)
X_perplexity_scaled = scaler_perplexity.transform(X_perplexity)
X_stability_scaled = scaler_stability.transform(X_stability)

# --- Обучение моделей  ---
print("Обучение моделей...")
model_style = LogisticRegression(random_state=42).fit(X_style_scaled, y)
model_semantic = LogisticRegression(random_state=42).fit(X_semantic_scaled, y)
model_perplexity = LogisticRegression(random_state=42).fit(X_perplexity_scaled, y)
model_stability = LogisticRegression(random_state=42).fit(X_stability_scaled, y)

# --- Сохранение ---
print("Сохранение моделей...")
joblib.dump(model_style, 'model_style.pkl')
joblib.dump(model_semantic, 'model_semantic.pkl')
joblib.dump(model_perplexity, 'model_perplexity.pkl')
joblib.dump(model_stability, 'model_stability.pkl')

joblib.dump(scaler_style, 'scaler_style.pkl')
joblib.dump(scaler_semantic, 'scaler_semantic.pkl')
joblib.dump(scaler_perplexity, 'scaler_perplexity.pkl')
joblib.dump(scaler_stability, 'scaler_stability.pkl')

print("Готово!")