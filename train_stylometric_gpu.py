import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib
from tqdm import tqdm
from config_gpu import config
from utils_gpu import load_data_gpu, batch_extract_stylometric_features


def train_stylometric_model_gpu():
    """Обучает стилометрическую модель."""
    print("=" * 70)
    print("🎨 HSSE - ОБУЧЕНИЕ СТИЛОМЕТРИЧЕСКОЙ МОДЕЛИ")
    print("=" * 70)

    if not config.check_data_files():
        return

    print("📥 Загрузка данных...")
    train_texts, train_labels = load_data_gpu(config.TRAIN_PATH)
    val_texts, val_labels = load_data_gpu(config.VAL_PATH)

    print("\n🔍 Извлечение стилометрических признаков...")
    X_train = batch_extract_stylometric_features(train_texts)
    X_val = batch_extract_stylometric_features(val_texts)

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    print(f"\n📊 Размеры данных:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")

    # Масштабирование
    print("\n⚖️  Масштабирование признаков...")
    scaler = RobustScaler()  # Устойчив к выбросам
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Обучение модели
    print("\n🌲 Обучение Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,  # Используем все ядра CPU
        random_state=42,
        verbose=1
    )

    model.fit(X_train_scaled, y_train)

    # Оценка
    print("\n📈 Оценка модели:")

    # Train
    train_preds = model.predict(X_train_scaled)
    train_probs = model.predict_proba(X_train_scaled)[:, 1]

    # Validation
    val_preds = model.predict(X_val_scaled)
    val_probs = model.predict_proba(X_val_scaled)[:, 1]

    # Метрики
    train_metrics = {
        'accuracy': accuracy_score(y_train, train_preds),
        'f1': f1_score(y_train, train_preds),
        'auc': roc_auc_score(y_train, train_probs)
    }

    val_metrics = {
        'accuracy': accuracy_score(y_val, val_preds),
        'f1': f1_score(y_val, val_preds),
        'auc': roc_auc_score(y_val, val_probs)
    }

    print("\n📊 Результаты:")
    print(f"   Обучающая выборка:")
    print(f"     Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"     F1 Score: {train_metrics['f1']:.4f}")
    print(f"     AUC ROC:  {train_metrics['auc']:.4f}")

    print(f"   Валидационная выборка:")
    print(f"     Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"     F1 Score: {val_metrics['f1']:.4f}")
    print(f"     AUC ROC:  {val_metrics['auc']:.4f}")

    # Classification report
    print("\n📋 Classification Report (Val):")
    print(classification_report(y_val, val_preds, target_names=['Human', 'AI']))

    # Сохранение моделей
    print("\n💾 Сохранение моделей...")
    joblib.dump(model, config.STYLOMETRIC_MODEL_PATH, compress=3)
    joblib.dump(scaler, config.STYLOMETRIC_SCALER_PATH, compress=3)

    print(f"   Модель: {config.STYLOMETRIC_MODEL_PATH}")
    print(f"   Scaler: {config.STYLOMETRIC_SCALER_PATH}")

    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': config.STYLOMETRIC_FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🏆 Топ-10 важных стилометрических признаков:")
    print(feature_importance.head(10).to_string(index=False))

    return model, scaler


if __name__ == "__main__":
    train_stylometric_model_gpu()