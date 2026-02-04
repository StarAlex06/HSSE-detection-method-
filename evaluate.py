import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from utils import load_features


class HSSEDetector:
    """Финальный детектор HSSE."""

    def __init__(self):
        print("Загрузка моделей HSSE...")

        # Загружаем мета-классификатор
        self.meta_model = joblib.load(Config.META_MODEL_PATH)

        # Загружаем семантическую модель
        from extract_features import SemanticModelWrapper
        self.semantic_model = SemanticModelWrapper(Config.SEMANTIC_MODEL_PATH)

        # Загружаем стилометрическую модель и scaler
        self.stylometric_model = joblib.load(Config.STYLOMETRIC_MODEL_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)

        # Инициализируем остальные компоненты
        from perplexity import get_perplexity_calculator
        from transformations import get_transformator
        from utils import extract_stylometric_features

        self.perplexity_calc = get_perplexity_calculator()
        self.transformator = get_transformator()
        self.extract_stylometric_features = extract_stylometric_features

        print("HSSE детектор загружен!")

    def predict_single(self, text: str) -> dict:
        """Предсказание для одного текста."""
        from extract_features import HSSEFeatureExtractor
        extractor = HSSEFeatureExtractor()

        # Извлекаем признаки
        features = extractor.extract_features_single(text)

        # Предсказание
        prob = self.meta_model.predict_proba(features.reshape(1, -1))[0, 1]
        prediction = 1 if prob > 0.5 else 0

        # Интерпретация
        feature_names = ['semantic_score', 'stylometric_score', 'perplexity_gap', 'stability_score']
        feature_dict = {name: float(value) for name, value in zip(feature_names, features)}

        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': 'AI-generated' if prediction == 1 else 'Human-written',
            'probability': float(prob),
            'confidence': 'high' if abs(prob - 0.5) > 0.3 else 'medium' if abs(prob - 0.5) > 0.1 else 'low',
            'features': feature_dict
        }

    def predict_batch(self, texts: list) -> list:
        """Предсказание для списка текстов."""
        results = []
        for text in texts:
            results.append(self.predict_single(text))
        return results


def ablation_study():
    """Ablation study: оценка важности каждого компонента."""
    print("\n=== Ablation Study ===")

    # Загружаем данные
    test_data = load_features('test')
    X_test = test_data['X']
    y_test = test_data['y']

    # Загружаем модель
    model = joblib.load(Config.META_MODEL_PATH)

    # Базовые результаты (все признаки)
    base_preds = model.predict(X_test)
    base_probs = model.predict_proba(X_test)[:, 1]
    base_f1 = f1_score(y_test, base_preds)
    base_auc = roc_auc_score(y_test, base_probs)

    print(f"Baseline (all features): F1 = {base_f1:.4f}, AUC = {base_auc:.4f}")

    # Тестируем каждый признак отдельно
    feature_names = ['semantic_score', 'stylometric_score', 'perplexity_gap', 'stability_score']

    for i, feature_name in enumerate(feature_names):
        # Используем только один признак
        X_single = X_test[:, i].reshape(-1, 1)

        # Обучаем простую модель на одном признаке
        from sklearn.linear_model import LogisticRegression
        single_model = LogisticRegression()

        # Для тренировки используем train данные
        train_data = load_features('train')
        X_train_single = train_data['X'][:, i].reshape(-1, 1)
        y_train = train_data['y']

        single_model.fit(X_train_single, y_train)

        # Предсказания
        single_preds = single_model.predict(X_single)
        single_probs = single_model.predict_proba(X_single)[:, 1]

        single_f1 = f1_score(y_test, single_preds)
        single_auc = roc_auc_score(y_test, single_probs)

        print(f"{feature_name}: F1 = {single_f1:.4f}, AUC = {single_auc:.4f}")

    # Тестируем комбинации без одного признака
    print("\n=== Feature Importance (leave-one-out) ===")

    for i in range(4):
        # Удаляем один признак
        mask = [j for j in range(4) if j != i]
        X_reduced = X_test[:, mask]

        # Обучаем модель на уменьшенных данных
        from lightgbm import LGBMClassifier

        train_data = load_features('train')
        X_train_reduced = train_data['X'][:, mask]
        y_train = train_data['y']

        reduced_model = LGBMClassifier(n_estimators=50)
        reduced_model.fit(X_train_reduced, y_train)

        reduced_preds = reduced_model.predict(X_reduced)
        reduced_probs = reduced_model.predict_proba(X_reduced)[:, 1]

        reduced_f1 = f1_score(y_test, reduced_preds)
        reduced_auc = roc_auc_score(y_test, reduced_probs)

        print(f"Without {feature_names[i]}: F1 = {reduced_f1:.4f} (Δ={base_f1 - reduced_f1:+.4f}), "
              f"AUC = {reduced_auc:.4f} (Δ={base_auc - reduced_auc:+.4f})")


def evaluate_on_test():
    """Оценка на тестовых данных."""
    print("\n=== Final Evaluation on Test Set ===")

    # Загружаем данные
    test_data = load_features('test')
    X_test = test_data['X']
    y_test = test_data['y']

    # Загружаем модель
    model = joblib.load(Config.META_MODEL_PATH)

    # Предсказания
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    # Метрики
    metrics = {
        'Accuracy': accuracy_score(y_test, test_preds),
        'F1 Score': f1_score(y_test, test_preds),
        'AUC ROC': roc_auc_score(y_test, test_probs),
        'Precision': precision_score(y_test, test_preds),
        'Recall': recall_score(y_test, test_preds)
    }

    print("\n=== Test Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(Config.MODELS_DIR / "test_confusion_matrix.png")
    plt.show()

    # Classification report
    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_preds, target_names=['Human', 'AI']))

    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {metrics["AUC ROC"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Config.MODELS_DIR / "test_roc_curve.png")
    plt.show()

    return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("HSSE DETECTOR - FINAL EVALUATION")
    print("=" * 60)

    # Основная оценка
    metrics = evaluate_on_test()

    # Ablation study
    ablation_study()

    # Демонстрация работы детектора
    print("\n" + "=" * 60)
    print("DEMO: Prediction Examples")
    print("=" * 60)

    detector = HSSEDetector()

    # Примеры текстов
    examples = [
        "The quick brown fox jumps over the lazy dog. This is a simple sentence that demonstrates basic English grammar and syntax.",
        "In light of recent developments, it is imperative to acknowledge the multifaceted nature of contemporary challenges that necessitate comprehensive analytical frameworks for optimal resolution.",
        "I was walking down the street when I saw this really cute dog, you know? It was like a little fluffy thing, and the owner was trying to get it to stop barking at a pigeon or something.",
        "The inherent complexity of quantum computational paradigms necessitates a thorough re-evaluation of traditional algorithmic approaches to ensure maximal computational efficiency within the constraints of quantum decoherence phenomena."
    ]

    for i, text in enumerate(examples, 1):
        result = detector.predict_single(text)
        print(f"\nExample {i}:")
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Confidence: {result['confidence']}")