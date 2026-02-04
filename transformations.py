import random
import numpy as np
from typing import List
from googletrans import Translator
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import torch

# Для синонимов
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-eng')


class TextTransformations:
    """Класс для преобразования текстов."""

    def __init__(self):
        self.translator = Translator()

    def back_translation(self, text, intermediate_lang='de'):
        """Обратный перевод (английский -> другой язык -> английский)."""
        try:
            translated = self.translator.translate(text, dest=intermediate_lang)
            back_translated = self.translator.translate(translated.text, dest='en')
            return back_translated.text
        except Exception as e:
            print(f"Error in back translation: {e}")
            return text

    def synonym_replacement(self, text, replace_ratio=0.1):
        """Замена слов на синонимы."""
        words = word_tokenize(text)
        new_words = words.copy()

        num_replacements = max(1, int(len(words) * replace_ratio))
        words_to_replace = random.sample(range(len(words)), num_replacements)

        for idx in words_to_replace:
            word = words[idx]
            synonyms = wordnet.synsets(word)

            if synonyms:
                # Получаем все леммы (основные формы слов)
                lemmas = []
                for syn in synonyms:
                    lemmas.extend(syn.lemmas())

                if lemmas:
                    # Выбираем синоним, который отличается от исходного слова
                    synonym = random.choice(lemmas).name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        new_words[idx] = synonym

        return ' '.join(new_words)

    def random_deletion(self, text, delete_ratio=0.1):
        """Случайное удаление слов."""
        words = text.split()
        if len(words) == 1:
            return text

        num_deletions = max(1, int(len(words) * delete_ratio))
        indices_to_keep = sorted(random.sample(range(len(words)), len(words) - num_deletions))
        new_words = [words[i] for i in indices_to_keep]

        return ' '.join(new_words)

    def random_swap(self, text, swap_ratio=0.1):
        """Случайная перестановка слов."""
        words = text.split()
        if len(words) < 2:
            return text

        num_swaps = max(1, int(len(words) * swap_ratio))

        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return ' '.join(words)

    def apply_transformations(self, text, n_transformations=3):
        """Применяет несколько трансформаций к тексту."""
        transformations = [
            lambda x: self.back_translation(x, 'de'),
            lambda x: self.back_translation(x, 'fr'),
            lambda x: self.synonym_replacement(x, 0.1),
            lambda x: self.random_deletion(x, 0.1),
            lambda x: self.random_swap(x, 0.1),
        ]

        selected_transformations = random.sample(transformations, min(n_transformations, len(transformations)))
        transformed_texts = []

        for transform in selected_transformations:
            try:
                transformed = transform(text)
                transformed_texts.append(transformed)
            except Exception as e:
                print(f"Transformation error: {e}")
                continue

        return transformed_texts


# Синглтон для трансформаций
_transformations = None


def get_transformator():
    """Возвращает экземпляр трансформатора."""
    global _transformations
    if _transformations is None:
        _transformations = TextTransformations()
    return _transformations