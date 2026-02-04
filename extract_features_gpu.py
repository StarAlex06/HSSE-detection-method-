import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config_gpu import config
from utils_gpu import load_data_gpu, extract_stylometric_features_gpu, save_gpu_features
from perplexity_gpu import get_perplexity_calculator
from transformations_gpu import get_transformations


class SemanticModelGPU:
    """Обертка для семантической модели на GPU."""

    def __init__(self, model_path):
        self.device = config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict_proba(self, text: str) -> float:
        """Возвращает вероятность того, что текст AI-generated."""
        encoding = self.tokenizer(
            text,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            if config.USE_AMP:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)

        return probs[0, 1].item()


class HSSEFeatureExtractorGPU:
    """Извлекает все 4 признака HSSE метода на GPU."""

    def __init__(self):
        print("=" * 70)
        print("🔍 HSSE - ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ НА GPU")
        print("=" * 70)

        # 1. Загружаем семантическую модель
        print("\n1️⃣  Загрузка семантической модели...")
        self.semantic_model = SemanticModelGPU(config.SEMANTIC_MODEL_PATH)

        # 2. Загружаем стилометрическую модель
        print("2️⃣  Загрузка стилометрической модели...")
        self.stylometric_model = joblib.load(config.STYLOMETRIC_MODEL_PATH)
        self.stylometric_scaler = joblib.load(config.STYLOMETRIC_SCALER_PATH)

        # 3. Инициализируем калькулятор перплексии
        print("3️⃣  Инициализация калькулятора перплексии...")
        self.perplexity_calc = get_perplexity_calculator()

        # 4. Инициализируем трансформации
        print("4️⃣  Инициализация трансформаций...")
        self.transformator = get_transformations()

        print("✅ HSSE Feature Extractor готов!")

    def extract_single(self, text: str) -> np.ndarray:
        """
        Извлекает все 4 признака HSSE для одного текста.

        Returns:
            [semantic_score, stylometric_score, perplexity_gap, stability_score]
        """
        if not text or len(text.strip()) < 10:
            return np.zeros(4, dtype=np.float32)

        features = []

        # 1. Semantic Score (p_sem)
        try:
            p_sem = self.semantic_model.predict_proba(text)
            features.append(p_sem)
        except:
            features.append(0.5)

        # 2. Stylometric Score (p_sty)
        try:
            style_features = extract_stylometric_features_gpu(text)
            style_vector = np.array([style_features[feat] for feat in config.STYLOMETRIC_FEATURES]).reshape(1, -1)
            style_vector_scaled = self.stylometric_scaler.transform(style_vector)
            p_sty = self.stylometric_model.predict_proba(style_vector_scaled)[0, 1]
            features.append(p_sty)
        except:
            features.append(0.5)

        # 3. Perplexity Gap (Δ_ppl)
        try:
            delta_ppl = self.perplexity_calc.calculate_perplexity_gap(text)
            # Нормализуем к диапазону [-0.5, 0.5]
            delta_ppl_norm = np.clip(delta_ppl, -5, 5) / 10.0
            features.append(delta_ppl_norm)
        except:
            features.append(0.0)

        # 4. Stability Score (s)
        try:
            # Применяем трансформации
            transformed_texts = self.transformator.apply_transformations(
                text,
                config.N_TRANSFORMATIONS
            )

            # Собираем вероятности для всех вариантов
            all_probs = [features[0]]  # Начинаем с оригинальной вероятности

            for transformed_text in transformed_texts:
                try:
                    prob = self.semantic_model.predict_proba(transformed_text)
                    all_probs.append(prob)
                except:
                    continue

            # Вычисляем дисперсию как меру нестабильности
            if len(all_probs) > 1:
                stability = np.var(all_probs)
            else:
                stability = 0.0

            features.append(stability)

        except Exception as e:
            print(f"Ошибка Stability Score: {e}")
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def extract_batch(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Извлекает признаки HSSE для батча текстов.

        Returns:
            features_array: массив признаков [n_samples, 4]
            raw_features: список сырых данных для анализа
        """
        features_list = []
        raw_features_list = []

        print(f"📊 Извлечение признаков для {len(texts)} текстов...")

        for i, text in enumerate(tqdm(texts, desc="HSSE Features")):
            try:
                features = self.extract_single(str(text))
                features_list.append(features)

                # Сохраняем сырые данные для анализа
                raw_features = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'p_sem': features[0],
                    'p_sty': features[1],
                    'delta_ppl': features[2],
                    'stability': features[3],
                    'features': features.tolist()
                }
                raw_features_list.append(raw_features)

            except Exception as e:
                print(f"Ошибка для текста {i}: {e}")
                features_list.append(np.zeros(4, dtype=np.float32))

        return np.array(features_list), raw_features_list


def extract_hsse_features():
    """Основная функция для извлечения и сохранения признаков HSSE."""
    extractor = HSSEFeatureExtractorGPU()

    # Для каждого датасета
    datasets = {
        'train': config.TRAIN_PATH,
        'val': config.VAL_PATH,
        'test': config.TEST_PATH
    }

    for name, path in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"📁 Обработка {name} датасета")
        print('=' * 60)

        # Загружаем данные
        texts, labels = load_data_gpu(path)

        # Извлекаем признаки
        features, raw_features = extractor.extract_batch(texts)

        # Сохраняем признаки
        features_dict = {
            'X': features,
            'y': np.array(labels),
            'texts': texts,
            'raw_features': raw_features,
            'feature_names': ['semantic_score', 'stylometric_score', 'perplexity_gap', 'stability_score']
        }

        save_gpu_features(features_dict, f"hsse_{name}")

        print(f"✅ Извлечено {len(features)} образцов")
        print(f"   Размерность: {features.shape}")

        # Статистика признаков
        print(f"\n📈 Статистика признаков для {name}:")
        for i, name in enumerate(['Semantic', 'Stylometric', 'Perplexity Gap', 'Stability']):
            feature_values = features[:, i]
            print(f"   {name}: mean={np.mean(feature_values):.4f}, "
                  f"std={np.std(feature_values):.4f}, "
                  f"min={np.min(feature_values):.4f}, "
                  f"max={np.max(feature_values):.4f}")


if __name__ == "__main__":
    extract_hsse_features()