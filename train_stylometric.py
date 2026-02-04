import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
from tqdm import tqdm
from config import Config
from utils import load_data, extract_stylometric_features


def extract_all_stylometric_features(texts):
    """Извлекает стилометрические признаки для всех текстов."""
    features_list = []

    print("Извлечение стилометрических признаков...")
    for text in tqdm(texts):
        features = extract_stylometric_features(text)
        # Преобразуем в список в правильном порядке
        feature_vector = [features[feat] for feat in Config.STYLOMETRIC_FEATURES]
        features_list.append(feature_vector)

    return np.array(features_list)


def train_stylometric_model():
    """Обучение стилометрической модели."""
    print("Загрузка данных...")
    train_texts, train_labels = load_data(Config.TRAIN_PATH)
    val_texts, val_labels = load_data(Config.VAL_PATH)

    print("Извлечение признаков...")
    X_train = extract_all_stylometric_features(train_texts)
    X_val = extract_all_stylometric_features(val_texts)

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    print(f"Train features shape: {X_train.shape}")
    print(f"Val features shape: {X_val.shape}")

    # Масштабирование признаков
    print("Масштабирование признаков...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Обучение модели
    print("Обучение Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    # Оценка
    train_preds = model.predict(X_train_scaled)
    train_probs = model.predict_proba(X_train_scaled)[:, 1]

    val_preds = model.predict(X_val_scaled)
    val_probs = model.predict_proba(X_val_scaled)[:, 1]

    train_acc = accuracy_score(y_train, train_preds)
    train_f1 = f1_score(y_train, train_preds)
    train_auc = roc_auc_score(y_train, train_probs)

    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_probs)

    print("\n=== Результаты ===")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Train F1: {train_f1:.4f}")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val F1: {val_f1:.4f}")
    print(f"Val AUC: {val_auc:.4f}")

    # Сохранение моделей
    print("\nСохранение моделей...")
    joblib.dump(model, Config.STYLOMETRIC_MODEL_PATH)
    joblib.dump(scaler, Config.SCALER_PATH)

    print(f"Модель сохранена в: {Config.STYLOMETRIC_MODEL_PATH}")
    print(f"Scaler сохранен в: {Config.SCALER_PATH}")

    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': Config.STYLOMETRIC_FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nТоп-10 важных признаков:")
    print(feature_importance.head(10))


if __name__ == "__main__":
    train_stylometric_model()