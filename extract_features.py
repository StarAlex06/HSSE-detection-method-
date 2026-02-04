import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import Config
from utils import load_data, extract_stylometric_features, save_features
from perplexity import get_perplexity_calculator
from transformations import get_transformator


class SemanticModelWrapper:
    """Обертка для семантической модели."""

    def __init__(self, model_path):
        self.device = Config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict_proba(self, text):
        """Возвращает вероятность того, что текст AI-generated."""
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_LENGTH,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)

        return probs[0, 1].item()


class HSSEFeatureExtractor:
    """Извлекает все признаки HSSE."""

    def __init__(self):
        print("Инициализация HSSE Feature Extractor...")

        # Загружаем семантическую модель
        print("Загрузка семантической модели...")
        self.semantic_model = SemanticModelWrapper(Config.SEMANTIC_MODEL_PATH)

        # Загружаем стилометрическую модель и scaler
        print("Загрузка стилометрической модели...")
        self.stylometric_model = joblib.load(Config.STYLOMETRIC_MODEL_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)

        # Инициализируем калькулятор перплексии
        self.perplexity_calc = get_perplexity_calculator()

        # Инициализируем трансформатор
        self.transformator = get_transformator()

        print("HSSE Feature Extractor готов!")

    def extract_features_single(self, text: str) -> np.ndarray:
        """Извлекает признаки HSSE для одного текста."""
        if not text or len(text.strip()) == 0:
            return np.zeros(4)

        # 1. Semantic Score
        try:
            p_sem = self.semantic_model.predict_proba(text)
        except:
            p_sem = 0.5

        # 2. Stylometric Score
        try:
            style_features = extract_stylometric_features(text)
            style_vector = np.array([style_features[feat] for feat in Config.STYLOMETRIC_FEATURES]).reshape(1, -1)
            style_vector_scaled = self.scaler.transform(style_vector)
            p_sty = self.stylometric_model.predict_proba(style_vector_scaled)[0, 1]
        except:
            p_sty = 0.5

        # 3. Perplexity Gap
        try:
            delta_ppl = self.perplexity_calc.calculate_perplexity_gap(text)
            # Нормализуем (эмпирически установленные границы)
            delta_ppl_norm = np.clip(delta_ppl, -5, 5) / 10  # Нормализуем к [-0.5, 0.5]
        except:
            delta_ppl_norm = 0

        # 4. Stability Score
        try:
            # Применяем трансформации
            transformed_texts = self.transformator.apply_transformations(text, n_transformations=3)

            # Собираем вероятности для всех вариантов
            probs = [p_sem]
            for transformed_text in transformed_texts[:3]:  # Берем максимум 3 трансформации
                try:
                    prob = self.semantic_model.predict_proba(transformed_text)
                    probs.append(prob)
                except:
                    continue

            # Вычисляем дисперсию
            if len(probs) > 1:
                stability = np.var(probs)
            else:
                stability = 0
        except:
            stability = 0

        return np.array([p_sem, p_sty, delta_ppl_norm, stability])

    def extract_features_batch(self, texts: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Извлекает признаки HSSE для батча текстов."""
        features_list = []
        raw_features_list = []

        print("Извлечение признаков HSSE...")
        for i, text in enumerate(tqdm(texts, desc="Processing texts")):
            try:
                features = self.extract_features_single(str(text))
                features_list.append(features)

                # Сохраняем также сырые признаки для анализа
                raw_features = {
                    'p_sem': features[0],
                    'p_sty': features[1],
                    'delta_ppl': features[2],
                    'stability': features[3],
                    'text': text
                }
                raw_features_list.append(raw_features)

            except Exception as e:
                print(f"Error processing text {i}: {e}")
                features_list.append(np.zeros(4))

        return np.array(features_list), raw_features_list


def extract_and_save_features():
    """Основная функция для извлечения и сохранения признаков."""
    extractor = HSSEFeatureExtractor()

    # Для каждого датасета
    datasets = {
        'train': Config.TRAIN_PATH,
        'val': Config.VAL_PATH,
        'test': Config.TEST_PATH
    }

    for name, path in datasets.items():
        print(f"\n=== Обработка {name} датасета ===")

        # Загружаем данные
        texts, labels = load_data(path)

        # Извлекаем признаки
        features, raw_features = extractor.extract_features_batch(texts)

        # Сохраняем признаки
        features_dict = {
            'X': features,
            'y': np.array(labels),
            'raw_features': raw_features
        }

        save_features(features_dict, name)

        print(f"Извлечено {len(features)} образцов")
        print(f"Размерность признаков: {features.shape}")


if __name__ == "__main__":
    extract_and_save_features()