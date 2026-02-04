import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from config import Config
from utils import load_features


def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """Визуализация важности признаков."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Модель не поддерживает feature_importances_ или coef_")
        return

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(Config.MODELS_DIR / "feature_importance.png")
    plt.show()

    # Таблица важности признаков
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })
    print("\nВажность признаков:")
    print(importance_df)


def plot_correlation_matrix(features, feature_names):
    """Визуализация корреляционной матрицы."""
    corr_matrix = np.corrcoef(features.T)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title("Корреляционная матрица признаков")
    plt.tight_layout()
    plt.savefig(Config.MODELS_DIR / "correlation_matrix.png")
    plt.show()


def train_meta_classifier():
    """Обучение мета-классификатора."""
    print("Загрузка признаков...")

    # Загружаем признаки
    train_data = load_features('train')
    val_data = load_features('val')

    X_train = train_data['X']
    y_train = train_data['y']
    X_val = val_data['X']
    y_val = val_data['y']

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")

    # Имена признаков
    feature_names = ['semantic_score', 'stylometric_score', 'perplexity_gap', 'stability_score']

    # Визуализация корреляций
    plot_correlation_matrix(X_train, feature_names)

    # Выбор модели
    if Config.META_MODEL_TYPE == "lightgbm":
        print("Используем LightGBM...")
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    elif Config.META_MODEL_TYPE == "xgboost":
        print("Используем XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    else:
        print("Используем LightGBM по умолчанию...")
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )

    # Обучение модели
    print("Обучение мета-классификатора...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='logloss',
        verbose=10
    )

    # Предсказания
    train_preds = model.predict(X_train)
    train_probs = model.predict_proba(X_train)[:, 1]

    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]

    # Метрики
    print("\n=== Результаты на тренировочных данных ===")
    print(f"Accuracy: {accuracy_score(y_train, train_preds):.4f}")
    print(f"F1 Score: {f1_score(y_train, train_preds):.4f}")
    print(f"AUC ROC: {roc_auc_score(y_train, train_probs):.4f}")
    print(f"Precision: {precision_score(y_train, train_preds):.4f}")
    print(f"Recall: {recall_score(y_train, train_preds):.4f}")

    print("\n=== Результаты на валидационных данных ===")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    print(f"F1 Score: {f1_score(y_val, val_preds):.4f}")
    print(f"AUC ROC: {roc_auc_score(y_val, val_probs):.4f}")
    print(f"Precision: {precision_score(y_val, val_preds):.4f}")
    print(f"Recall: {recall_score(y_val, val_preds):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Validation)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(Config.MODELS_DIR / "confusion_matrix.png")
    plt.show()

    # Classification report
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, val_preds, target_names=['Human', 'AI']))

    # Визуализация важности признаков
    plot_feature_importance(model, feature_names)

    # Сохранение модели
    print("\nСохранение модели...")
    joblib.dump(model, Config.META_MODEL_PATH)
    print(f"Модель сохранена в: {Config.META_MODEL_PATH}")

    # Анализ распределения признаков
    analyze_features_distribution(train_data, val_data, feature_names)


def analyze_features_distribution(train_data, val_data, feature_names):
    """Анализ распределения признаков."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, feature_name in enumerate(feature_names):
        train_feature = train_data['X'][:, i]
        val_feature = val_data['X'][:, i]

        # Разделяем по классам
        train_human = train_feature[train_data['y'] == 0]
        train_ai = train_feature[train_data['y'] == 1]
        val_human = val_feature[val_data['y'] == 0]
        val_ai = val_feature[val_data['y'] == 1]

        axes[i].hist(train_human, alpha=0.5, label='Train Human', bins=30, density=True)
        axes[i].hist(train_ai, alpha=0.5, label='Train AI', bins=30, density=True)
        axes[i].set_title(feature_name)
        axes[i].legend()
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(Config.MODELS_DIR / "features_distribution.png")
    plt.show()


if __name__ == "__main__":
    train_meta_classifier()