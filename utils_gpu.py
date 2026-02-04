import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Any, Optional
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import torch
from tqdm import tqdm
import joblib
import os
from pathlib import Path
from config_gpu import config


# Загрузка ресурсов NLTK
def setup_nltk():
    """Загружает необходимые ресурсы NLTK."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')
        nltk.download('wordnet')
    return True


# Инициализируем NLTK
setup_nltk()


def load_data_gpu(file_path: Path, max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """
    Загружает данные с оптимизацией для GPU.

    Args:
        file_path: путь к файлу
        max_samples: максимальное количество примеров (для тестирования)

    Returns:
        texts: список текстов
        labels: список меток
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    # Определяем формат файла
    if file_path.suffix.lower() == '.csv':
        if max_samples:
            df = pd.read_csv(file_path, nrows=max_samples)
        else:
            df = pd.read_csv(file_path)
    else:
        if max_samples:
            df = pd.read_excel(file_path, nrows=max_samples)
        else:
            df = pd.read_excel(file_path)

    # Проверяем колонки
    required_columns = ['text', 'label']
    for col in required_columns:
        if col not in df.columns:
            # Пробуем найти альтернативные названия
            alt_names = {
                'text': ['content', 'sentence', 'document', 'Text', 'Content'],
                'label': ['target', 'is_ai', 'ai_generated', 'Label', 'Target']
            }

            found = False
            for alt in alt_names[col]:
                if alt in df.columns:
                    df = df.rename(columns={alt: col})
                    found = True
                    break

            if not found:
                raise ValueError(f"Колонка '{col}' не найдена. Доступные колонки: {list(df.columns)}")

    # Удаляем пустые значения
    df = df.dropna(subset=['text', 'label'])

    # Преобразуем метки в int
    df['label'] = df['label'].astype(int)

    # Берем выборку если нужно
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)

    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    print(f"📊 Загружено {len(texts)} примеров из {file_path.name}")
    print(f"   Распределение: Human (0): {labels.count(0)}, AI (1): {labels.count(1)}")

    return texts, labels


def extract_stylometric_features_gpu(text: str) -> Dict[str, float]:
    """
    Извлекает стилометрические признаки с оптимизацией.

    Args:
        text: входной текст

    Returns:
        Словарь с признаками
    """
    if not text or len(text.strip()) == 0:
        return {feature: 0.0 for feature in config.STYLOMETRIC_FEATURES}

    # Токенизация
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Фильтруем только буквы для некоторых вычислений
    words_alpha = [w for w in words if w.isalpha()]

    if len(words_alpha) == 0:
        return {feature: 0.0 for feature in config.STYLOMETRIC_FEATURES}

    # Части речи
    pos_tags = pos_tag(words_alpha)

    # Счетчики POS
    pos_counts = {}
    for _, tag in pos_tags:
        pos_counts[tag] = pos_counts.get(tag, 0) + 1

    # Списки стоп-слов
    stop_words = set(stopwords.words('english'))
    function_words = [w for w in words_alpha if w in stop_words]
    content_words = [w for w in words_alpha if w not in stop_words]

    # Вычисляем признаки
    features = {}

    # Базовые статистики
    features['text_length'] = len(text)
    features['num_sentences'] = len(sentences)
    features['num_words'] = len(words_alpha)
    features['num_unique_words'] = len(set(words_alpha))

    # Лексическое разнообразие
    features['ttr'] = features['num_unique_words'] / features['num_words'] if features['num_words'] > 0 else 0

    # Длина предложений
    sent_lengths = [len(sent.split()) for sent in sentences if len(sent.split()) > 0]
    features['avg_sentence_length'] = np.mean(sent_lengths) if sent_lengths else 0
    features['std_sentence_length'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

    # Длина слов
    word_lengths = [len(w) for w in words_alpha]
    features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0

    # Частота знаков препинания
    features['comma_freq'] = text.count(',') / len(text) if len(text) > 0 else 0
    features['period_freq'] = text.count('.') / len(text) if len(text) > 0 else 0
    features['question_freq'] = text.count('?') / len(text) if len(text) > 0 else 0
    features['exclamation_freq'] = text.count('!') / len(text) if len(text) > 0 else 0

    # Части речи
    total_pos = sum(pos_counts.values()) if pos_counts else 1
    features['noun_ratio'] = sum(pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('NN')) / total_pos
    features['verb_ratio'] = sum(pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('VB')) / total_pos
    features['adjective_ratio'] = sum(pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('JJ')) / total_pos
    features['adverb_ratio'] = sum(pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('RB')) / total_pos
    features['pronoun_ratio'] = sum(pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('PR')) / total_pos

    # Стоп-слова
    features['function_word_ratio'] = len(function_words) / len(words_alpha) if words_alpha else 0
    features['content_word_ratio'] = len(content_words) / len(words_alpha) if words_alpha else 0

    # Другие признаки
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0

    return features


def batch_extract_stylometric_features(texts: List[str]) -> np.ndarray:
    """
    Извлекает стилометрические признаки для батча текстов.

    Args:
        texts: список текстов

    Returns:
        Матрица признаков
    """
    features_list = []

    for text in tqdm(texts, desc="Извлечение стилометрических признаков"):
        features = extract_stylometric_features_gpu(text)
        feature_vector = [features[feat] for feat in config.STYLOMETRIC_FEATURES]
        features_list.append(feature_vector)

    return np.array(features_list)


def save_gpu_features(features_dict: Dict[str, Any], name: str):
    """Сохраняет признаки в файл."""
    path = config.FEATURES_DIR / f"{name}.pkl"
    joblib.dump(features_dict, path, compress=3)
    print(f"💾 Признаки сохранены: {path}")


def load_gpu_features(name: str) -> Dict[str, Any]:
    """Загружает признаки из файла."""
    path = config.FEATURES_DIR / f"{name}.pkl"
    return joblib.load(path)


def cleanup_gpu_memory():
    """Очищает память GPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()