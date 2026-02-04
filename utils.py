import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Any
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import torch
from tqdm import tqdm
import joblib
import os
from pathlib import Path

# Загрузка ресурсов NLTK (выполнить один раз)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')


def load_data(file_path: str) -> Tuple[List[str], List[int]]:
    """
    Загружает данные из файла (поддерживает CSV и Excel).

    Args:
        file_path: путь к файлу (.csv, .xlsx, .xls)

    Returns:
        texts: список текстов
        labels: список меток (0 - human, 1 - AI)
    """
    # Проверяем существование файла
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    # Определяем расширение файла
    file_ext = Path(file_path).suffix.lower()

    # Загружаем данные в зависимости от формата
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {file_ext}. Используйте .csv, .xlsx или .xls")

    # Проверяем наличие нужных колонок
    if 'text' not in df.columns:
        # Пробуем найти альтернативные названия колонок
        possible_text_columns = ['text', 'Text', 'TEXT', 'content', 'Content', 'CONTENT', 'sentence', 'Sentence']
        for col in possible_text_columns:
            if col in df.columns:
                df = df.rename(columns={col: 'text'})
                break

        if 'text' not in df.columns:
            raise ValueError(f"В файле должна быть колонка 'text'. Найдены колонки: {list(df.columns)}")

    if 'label' not in df.columns:
        # Пробуем найти альтернативные названия колонок для меток
        possible_label_columns = ['label', 'Label', 'LABEL', 'is_ai', 'is_ai_generated', 'target', 'Target']
        for col in possible_label_columns:
            if col in df.columns:
                df = df.rename(columns={col: 'label'})
                break

        if 'label' not in df.columns:
            raise ValueError(f"В файле должна быть колонка 'label'. Найдены колонки: {list(df.columns)}")

    # Удаляем пустые значения
    df = df.dropna(subset=['text', 'label'])

    # Преобразуем в списки
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    # Проверяем метки
    unique_labels = set(labels)
    if unique_labels != {0, 1}:
        print(f"Внимание: метки должны быть 0 или 1. Найдены значения: {unique_labels}")

    print(f"Загружено {len(texts)} примеров из {file_path}")
    print(f"Распределение меток: Human (0): {labels.count(0)}, AI (1): {labels.count(1)}")

    return texts, labels

def extract_stylometric_features(text: str) -> Dict[str, float]:
    """
    Извлекает стилометрические признаки из текста.

    Args:
        text: входной текст

    Returns:
        Словарь с признаками
    """
    if not text or len(text.strip()) == 0:
        return {feature: 0.0 for feature in Config.STYLOMETRIC_FEATURES}

    # Токенизация
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Фильтруем только буквы
    words_alpha = [w for w in words if w.isalpha()]

    if len(words_alpha) == 0:
        return {feature: 0.0 for feature in Config.STYLOMETRIC_FEATURES}

    # Части речи
    pos_tags = pos_tag(words_alpha)
    pos_counts = {}
    for _, tag in pos_tags:
        pos_counts[tag] = pos_counts.get(tag, 0) + 1

    # Вычисляем признаки
    features = {
        'text_length': len(text),
        'num_sentences': len(sentences),
        'num_words': len(words_alpha),
        'num_unique_words': len(set(words_alpha)),
    }

    # Лексическое разнообразие (Type-Token Ratio)
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

    # Части речи (приблизительно через первые две буквы тега)
    total_pos = sum(pos_counts.values())
    features['noun_ratio'] = sum(
        pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('NN')) / total_pos if total_pos > 0 else 0
    features['verb_ratio'] = sum(
        pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('VB')) / total_pos if total_pos > 0 else 0
    features['adjective_ratio'] = sum(
        pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('JJ')) / total_pos if total_pos > 0 else 0
    features['adverb_ratio'] = sum(
        pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('RB')) / total_pos if total_pos > 0 else 0
    features['pronoun_ratio'] = sum(
        pos_counts.get(tag, 0) for tag in pos_counts if tag.startswith('PR')) / total_pos if total_pos > 0 else 0

    # Другие признаки
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0

    return features


def save_features(features_dict: Dict[str, Any], name: str):
    """
    Сохраняет признаки в файл.
    """
    path = Config.FEATURES_DIR / f"{name}.pkl"
    joblib.dump(features_dict, path)
    print(f"Признаки сохранены в {path}")


def load_features(name: str) -> Dict[str, Any]:
    """
    Загружает признаки из файла.
    """
    path = Config.FEATURES_DIR / f"{name}.pkl"
    return joblib.load(path)