import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from config_gpu import config
from utils_gpu import load_gpu_features, extract_stylometric_features_gpu
from perplexity_gpu import get_perplexity_calculator
from transformations_gpu import get_transformations


class HSSEDetectorGPU:
    """Финальный детектор HSSE с GPU оптимизацией."""

    def __init__(self):
        print("🎯 Инициализация HSSE детектора...")

        # Загружаем мета-классификатор
        self.meta_model = joblib.load(config.META_MODEL_PATH)

        # Загружаем семантическую модель
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        try:
            self.semantic_tokenizer = AutoTokenizer.from_pretrained(config.SEMANTIC_MODEL_PATH)
            self.semantic_model = AutoModelForSequenceClassification.from_pretrained(
                config.SEMANTIC_MODEL_PATH
            ).to(config.DEVICE)
            print(f"   ✅ Семантическая модель загружена из {config.SEMANTIC_MODEL_PATH}")
        except Exception as e:
            print(f"   ⚠️ Не удалось загрузить модель из {config.SEMANTIC_MODEL_PATH}: {e}")
            print(f"   ↪ Загружаю базовую модель {config.SEMANTIC_MODEL_NAME}")
            self.semantic_tokenizer = AutoTokenizer.from_pretrained(config.SEMANTIC_MODEL_NAME)
            self.semantic_model = AutoModelForSequenceClassification.from_pretrained(
                config.SEMANTIC_MODEL_NAME,
                num_labels=2
            ).to(config.DEVICE)

        self.semantic_model.eval()

        # Загружаем стилометрическую модель
        self.stylometric_model = joblib.load(config.STYLOMETRIC_MODEL_PATH)
        self.stylometric_scaler = joblib.load(config.SCALER_PATH)

        # Инициализируем перплексию и трансформации
        self.perplexity_calc = get_perplexity_calculator()
        self.transformator = get_transformations()

        print("✅ HSSE детектор готов!")

    def extract_features(self, text: str) -> np.ndarray:
        """Извлекает все 4 признака HSSE для одного текста."""
        features = []

        # 1. Semantic Score
        encoding = self.semantic_tokenizer(
            text,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(config.DEVICE)
            attention_mask = encoding['attention_mask'].to(config.DEVICE)

            if config.USE_AMP and config.DEVICE.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.semantic_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = self.semantic_model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)
            p_sem = probs[0, 1].item()

        features.append(p_sem)

        # 2. Stylometric Score
        style_features = extract_stylometric_features_gpu(text)
        style_vector = np.array([style_features[feat] for feat in config.STYLOMETRIC_FEATURES]).reshape(1, -1)
        style_vector_scaled = self.stylometric_scaler.transform(style_vector)
        p_sty = self.stylometric_model.predict_proba(style_vector_scaled)[0, 1]
        features.append(p_sty)

        # 3. Perplexity Gap
        delta_ppl = self.perplexity_calc.calculate_perplexity_gap(text)
        delta_ppl_norm = np.clip(delta_ppl, -5, 5) / 10.0
        features.append(delta_ppl_norm)

        # 4. Stability Score
        transformed_texts = self.transformator.apply_transformations(text, config.N_TRANSFORMATIONS)
        all_probs = [p_sem]

        for transformed_text in transformed_texts:
            try:
                encoding = self.semantic_tokenizer(
                    transformed_text,
                    max_length=config.MAX_LENGTH,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                with torch.no_grad():
                    input_ids = encoding['input_ids'].to(config.DEVICE)
                    attention_mask = encoding['attention_mask'].to(config.DEVICE)

                    if config.USE_AMP and config.DEVICE.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            outputs = self.semantic_model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        outputs = self.semantic_model(input_ids=input_ids, attention_mask=attention_mask)

                    probs = torch.softmax(outputs.logits, dim=1)
                    prob = probs[0, 1].item()
                    all_probs.append(prob)
            except:
                continue

        stability = np.var(all_probs) if len(all_probs) > 1 else 0.0
        features.append(stability)

        return np.array(features)

    def predict(self, text: str, return_proba: bool = False):
        """
        Предсказывает, является ли текст AI-generated.

        Args:
            text: входной текст
            return_proba: если True, возвращает вероятность

        Returns:
            prediction (0/1) или вероятность
        """
        features = self.extract_features(text).reshape(1, -1)
        proba = self.meta_model.predict_proba(features)[0, 1]

        if return_proba:
            return proba
        else:
            return 1 if proba > 0.5 else 0

    def predict_batch(self, texts: list, batch_size: int = 32):
        """Предсказание для батча текстов."""
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            for text in batch_texts:
                proba = self.predict(text, return_proba=True)
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'probability': proba,
                    'prediction': 'AI' if proba > 0.5 else 'Human',
                    'confidence': 'High' if abs(proba - 0.5) > 0.3 else
                    'Medium' if abs(proba - 0.5) > 0.1 else 'Low'
                })

        return results


def evaluate_hsse_gpu():
    """Финальная оценка HSSE метода на тестовых данных."""
    print("=" * 70)
    print("🎯 HSSE - ФИНАЛЬНАЯ ОЦЕНКА НА GPU")
    print("=" * 70)

    # Загружаем тестовые признаки
    print("📥 Загрузка тестовых данных...")
    test_data = load_gpu_features("hsse_test")

    X_test = test_data['X']
    y_test = test_data['y']

    print(f"Тестовые данные: {X_test.shape}")

    # Загружаем мета-классификатор
    meta_model = joblib.load(config.META_MODEL_PATH)

    # Предсказания
    print("\n🔮 Прогнозирование...")
    y_pred = meta_model.predict(X_test)
    y_proba = meta_model.predict_proba(X_test)[:, 1]

    # Метрики
    print("\n" + "=" * 50)
    print("📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ HSSE")
    print("=" * 50)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUC ROC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }

    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'],
                yticklabels=['Human', 'AI'])
    plt.title('Confusion Matrix - HSSE Final Evaluation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(config.MODELS_DIR / "hsse_final_confusion_matrix.png", dpi=100)
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'HSSE (AUC = {metrics["AUC ROC"]:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - HSSE Method')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.MODELS_DIR / "hsse_roc_curve.png", dpi=100)
    plt.show()

    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

    return metrics


def ablation_study_gpu():
    """Ablation study для оценки вклада каждого признака HSSE."""
    print("\n" + "=" * 70)
    print("🔬 HSSE - ABLATION STUDY")
    print("=" * 70)

    # Загружаем данные
    test_data = load_gpu_features("hsse_test")
    X_test = test_data['X']
    y_test = test_data['y']

    feature_names = ['Semantic', 'Stylometric', 'Perplexity Gap', 'Stability']

    # Базовые результаты (все признаки)
    meta_model = joblib.load(config.META_MODEL_PATH)
    y_pred_full = meta_model.predict(X_test)
    f1_full = f1_score(y_test, y_pred_full)
    auc_full = roc_auc_score(y_test, meta_model.predict_proba(X_test)[:, 1])

    print(f"\n📊 Базовые результаты (все 4 признака):")
    print(f"  F1 Score: {f1_full:.4f}")
    print(f"  AUC ROC:  {auc_full:.4f}")

    print("\n📉 Ablation Study (без одного признака):")

    for i in range(4):
        # Удаляем один признак
        X_reduced = np.delete(X_test, i, axis=1)

        # Обучаем новую модель на уменьшенных признаках
        from lightgbm import LGBMClassifier

        # Загружаем тренировочные данные для переобучения
        train_data = load_gpu_features("hsse_train")
        X_train_reduced = np.delete(train_data['X'], i, axis=1)
        y_train = train_data['y']

        # Быстрое обучение
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_reduced, y_train)

        # Оценка
        y_pred_reduced = model.predict(X_reduced)
        y_proba_reduced = model.predict_proba(X_reduced)[:, 1]

        f1_reduced = f1_score(y_test, y_pred_reduced)
        auc_reduced = roc_auc_score(y_test, y_proba_reduced)

        print(f"\n  Без {feature_names[i]}:")
        print(f"    F1 Score: {f1_reduced:.4f} (Δ = {f1_full - f1_reduced:+.4f})")
        print(f"    AUC ROC:  {auc_reduced:.4f} (Δ = {auc_full - auc_reduced:+.4f})")

        # Если падение значительное, признак важен
        if f1_full - f1_reduced > 0.05:
            print(f"    ⚠️  {feature_names[i]} критически важен!")


if __name__ == "__main__":
    # Финальная оценка
    metrics = evaluate_hsse_gpu()

    # Ablation study
    ablation_study_gpu()

    # Демонстрация работы детектора
    print("\n" + "=" * 70)
    print("🎭 ДЕМОНСТРАЦИЯ HSSE ДЕТЕКТОРА")
    print("=" * 70)

    detector = HSSEDetectorGPU()

    # Примеры текстов
    examples = [
        "I was walking to the store when I saw a really cute dog. It was barking at a squirrel in a tree.",
        "The optimization of machine learning algorithms requires careful consideration of hyperparameter tuning to achieve optimal performance metrics.",
        "Yesterday, my friend and I went to see a movie. The theater was pretty crowded, but we still managed to get good seats.",
        "The implementation of deep neural networks necessitates the utilization of gradient-based optimization techniques for effective parameter estimation."
    ]

    for i, text in enumerate(examples, 1):
        result = detector.predict(text, return_proba=True)
        print(f"\nПример {i}:")
        print(f"  Текст: {text[:60]}...")
        print(f"  Вероятность AI: {result:.3f}")
        print(f"  Предсказание: {'🤖 AI-текст' if result > 0.5 else '👤 Человеческий текст'}")
        print(
            f"  Уверенность: {'Высокая' if abs(result - 0.5) > 0.3 else 'Средняя' if abs(result - 0.5) > 0.1 else 'Низкая'}")