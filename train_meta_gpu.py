import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from config_gpu import config
from utils_gpu import load_gpu_features


def plot_hsse_features(features, labels, feature_names):
    """Визуализация распределения признаков HSSE."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = ['blue', 'red']  # Human, AI

    for i, feature_name in enumerate(feature_names):
        human_vals = features[labels == 0, i]
        ai_vals = features[labels == 1, i]

        axes[i].hist(human_vals, alpha=0.5, color=colors[0],
                     bins=30, density=True, label='Human')
        axes[i].hist(ai_vals, alpha=0.5, color=colors[1],
                     bins=30, density=True, label='AI')

        axes[i].set_title(f'HSSE Feature: {feature_name}', fontsize=12)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Distribution of HSSE Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(config.MODELS_DIR / "hsse_features_distribution.png", dpi=100)
    plt.show()


def train_meta_classifier_gpu():
    """Обучает мета-классификатор на признаках HSSE."""
    print("=" * 70)
    print("🎯 HSSE - ОБУЧЕНИЕ МЕТА-КЛАССИФИКАТОРА")
    print("=" * 70)

    print("📥 Загрузка признаков HSSE...")

    # Загружаем признаки
    train_data = load_gpu_features("hsse_train")
    val_data = load_gpu_features("hsse_val")

    X_train = train_data['X']
    y_train = train_data['y']
    X_val = val_data['X']
    y_val = val_data['y']

    print(f"\n📊 Размеры данных:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")

    # Визуализация признаков
    feature_names = ['Semantic Score', 'Stylometric Score', 'Perplexity Gap', 'Stability Score']
    plot_hsse_features(X_train, y_train, feature_names)

    # Корреляционная матрица
    corr_matrix = np.corrcoef(X_train.T)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Correlation Matrix of HSSE Features')
    plt.tight_layout()
    plt.savefig(config.MODELS_DIR / "hsse_correlation_matrix.png", dpi=100)
    plt.show()

    # Выбор модели
    if config.META_MODEL_TYPE == "lightgbm":
        print("\n🌲 Обучение LightGBM...")
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        # Простой tuning
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 63],
            'max_depth': [5, 7, 9]
        }

    elif config.META_MODEL_TYPE == "xgboost":
        print("\n🌀 Обучение XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    elif config.META_MODEL_TYPE == "catboost":
        print("\n🐱 Обучение CatBoost...")
        model = CatBoostClassifier(
            iterations=200,
            depth=7,
            learning_rate=0.05,
            verbose=0,
            random_seed=42
        )
    else:
        print(f"\n⚠️  Неизвестный тип модели: {config.META_MODEL_TYPE}")
        print("Использую LightGBM по умолчанию")
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )

    # Обучение
    print("\n🚀 Обучение мета-классификатора...")
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
    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТЫ МЕТА-КЛАССИФИКАТОРА")
    print("=" * 50)

    print("\nОбучающая выборка:")
    print(f"  Accuracy:  {accuracy_score(y_train, train_preds):.4f}")
    print(f"  F1 Score:  {f1_score(y_train, train_preds):.4f}")
    print(f"  AUC ROC:   {roc_auc_score(y_train, train_probs):.4f}")
    print(f"  Precision: {precision_score(y_train, train_preds):.4f}")
    print(f"  Recall:    {recall_score(y_train, train_preds):.4f}")

    print("\nВалидационная выборка:")
    print(f"  Accuracy:  {accuracy_score(y_val, val_preds):.4f}")
    print(f"  F1 Score:  {f1_score(y_val, val_preds):.4f}")
    print(f"  AUC ROC:   {roc_auc_score(y_val, val_probs):.4f}")
    print(f"  Precision: {precision_score(y_val, val_preds):.4f}")
    print(f"  Recall:    {recall_score(y_val, val_preds):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, val_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'],
                yticklabels=['Human', 'AI'])
    plt.title('Confusion Matrix - HSSE Meta-Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(config.MODELS_DIR / "hsse_confusion_matrix.png", dpi=100)
    plt.show()

    # Classification report
    print("\n📋 Classification Report (Validation):")
    print(classification_report(y_val, val_preds, target_names=['Human', 'AI']))

    # Важность признаков
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        print("\n🏆 Важность признаков HSSE:")
        print(importance_df.to_string(index=False))

        # График важности признаков
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(importance)), importance_df['Importance'])
        plt.yticks(range(len(importance)), importance_df['Feature'])
        plt.xlabel('Importance')
        plt.title('Feature Importance - HSSE Meta-Classifier')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(config.MODELS_DIR / "hsse_feature_importance.png", dpi=100)
        plt.show()

    # Сохранение модели
    print("\n💾 Сохранение мета-классификатора...")
    joblib.dump(model, config.META_MODEL_PATH, compress=3)
    print(f"   Модель сохранена: {config.META_MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_meta_classifier_gpu()