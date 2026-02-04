import random
import numpy as np
from typing import List, Callable
import torch
from transformers import MarianMTModel, MarianTokenizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
from config_gpu import config
from utils_gpu import cleanup_gpu_memory

# Загружаем NLTK ресурсы
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class GPUTransformations:
    """Трансформации текстов для Stability Score с GPU ускорением."""

    def __init__(self):
        self.device = config.DEVICE

        # Загружаем модели для back-translation на GPU
        print("🌍 Загрузка моделей трансляции для Stability Score...")

        try:
            # Модель для перевода English -> German
            self.en_de_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
            self.en_de_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de').to(self.device)
            self.en_de_model.eval()

            # Модель для перевода German -> English
            self.de_en_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
            self.de_en_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en').to(self.device)
            self.de_en_model.eval()

            self.has_translation = True
            print(f"✅ Модели трансляции загружены на {self.device}")
        except Exception as e:
            print(f"⚠️  Не удалось загрузить модели трансляции: {e}")
            self.has_translation = False

    def back_translation_gpu(self, text: str) -> str:
        """Обратный перевод на GPU (English -> German -> English)."""
        if not self.has_translation:
            return text

        try:
            # English -> German
            batch = self.en_de_tokenizer([text], return_tensors="pt", padding=True, truncation=True)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                if config.USE_AMP:
                    with torch.cuda.amp.autocast():
                        translated = self.en_de_model.generate(**batch)
                else:
                    translated = self.en_de_model.generate(**batch)

            german_text = self.en_de_tokenizer.decode(translated[0], skip_special_tokens=True)

            # German -> English
            batch = self.de_en_tokenizer([german_text], return_tensors="pt", padding=True, truncation=True)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                if config.USE_AMP:
                    with torch.cuda.amp.autocast():
                        back_translated = self.de_en_model.generate(**batch)
                else:
                    back_translated = self.de_en_model.generate(**batch)

            result = self.de_en_tokenizer.decode(back_translated[0], skip_special_tokens=True)

            return result

        except Exception as e:
            print(f"Ошибка back-translation: {e}")
            return text

    def synonym_replacement(self, text: str, replace_ratio: float = 0.1) -> str:
        """Замена слов на синонимы."""
        words = word_tokenize(text)
        if len(words) <= 1:
            return text

        n_replace = max(1, int(len(words) * replace_ratio))
        indices_to_replace = random.sample(range(len(words)), n_replace)

        new_words = words.copy()

        for idx in indices_to_replace:
            word = words[idx]

            # Получаем синонимы
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ').lower()
                    if synonym != word.lower():
                        synonyms.append(synonym)

            if synonyms:
                # Выбираем случайный синоним
                new_word = random.choice(synonyms)
                new_words[idx] = new_word

        return ' '.join(new_words)

    def random_deletion(self, text: str, delete_ratio: float = 0.1) -> str:
        """Случайное удаление слов."""
        words = text.split()
        if len(words) <= 1:
            return text

        n_delete = max(1, int(len(words) * delete_ratio))
        indices_to_keep = sorted(random.sample(range(len(words)), len(words) - n_delete))

        new_words = [words[i] for i in indices_to_keep]
        return ' '.join(new_words)

    def random_swap(self, text: str, swap_ratio: float = 0.1) -> str:
        """Случайная перестановка слов."""
        words = text.split()
        if len(words) <= 1:
            return text

        n_swap = max(1, int(len(words) * swap_ratio))

        for _ in range(n_swap):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return ' '.join(words)

    def apply_transformations(self, text: str, n_transformations: int = None) -> List[str]:
        """
        Применяет несколько трансформаций к тексту.

        Returns:
            Список трансформированных текстов
        """
        if n_transformations is None:
            n_transformations = config.N_TRANSFORMATIONS

        transformations = []

        # Собираем доступные трансформации
        available_transforms = []

        if self.has_translation:
            available_transforms.append(('back_translation', self.back_translation_gpu))

        available_transforms.extend([
            ('synonym_replacement', lambda t: self.synonym_replacement(t, 0.1)),
            ('random_deletion', lambda t: self.random_deletion(t, 0.1)),
            ('random_swap', lambda t: self.random_swap(t, 0.1))
        ])

        # Выбираем случайные трансформации
        selected_transforms = random.sample(
            available_transforms,
            min(n_transformations, len(available_transforms))
        )

        for name, transform in selected_transforms:
            try:
                transformed = transform(text)
                if transformed != text and len(transformed) > 10:
                    transformations.append(transformed)
            except Exception as e:
                print(f"Ошибка трансформации {name}: {e}")

        return transformations


# Синглтон для трансформатора
_transformations = None


def get_transformations():
    """Возвращает экземпляр трансформатора."""
    global _transformations
    if _transformations is None:
        _transformations = GPUTransformations()
    return _transformations