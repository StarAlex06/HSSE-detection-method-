import os
from pathlib import Path
import torch


class Config:
    # Пути к данным
    # Если у вас есть папка 'data' с файлами внутри
    DATA_DIR = Path("data")  # Папка 'data' в текущей директории

    # Если файлы CSV в папке data
    TRAIN_PATH = DATA_DIR / "train.csv"
    VAL_PATH = DATA_DIR / "val.csv"
    TEST_PATH = DATA_DIR / "test.csv"

    # Проверяем существование файлов
    def check_files_exist(self):
        """Проверяет существование всех файлов данных."""
        print("=" * 50)
        print("ПРОВЕРКА ФАЙЛОВ ДАННЫХ")
        print("=" * 50)

        files_to_check = [
            ("Обучающие данные", self.TRAIN_PATH),
            ("Валидационные данные", self.VAL_PATH),
            ("Тестовые данные", self.TEST_PATH)
        ]

        all_exist = True
        for name, path in files_to_check:
            exists = path.exists()
            status = "✅ НАЙДЕН" if exists else "❌ НЕ НАЙДЕН"
            print(f"{name}: {status}")
            print(f"  Путь: {path}")

            if exists:
                try:
                    # Показываем размер файла
                    size_kb = path.stat().st_size / 1024
                    print(f"  Размер: {size_kb:.1f} KB")

                    # Показываем количество строк
                    import pandas as pd
                    if path.suffix.lower() == '.csv':
                        df = pd.read_csv(path, nrows=1)
                    elif path.suffix.lower() in ['.xlsx', '.xls']:
                        df = pd.read_excel(path, nrows=1)

                    # Читаем весь файл для подсчета строк
                    if path.suffix.lower() == '.csv':
                        df_full = pd.read_csv(path)
                    elif path.suffix.lower() in ['.xlsx', '.xls']:
                        df_full = pd.read_excel(path)

                    print(f"  Количество строк: {len(df_full)}")
                    print(f"  Колонки: {list(df_full.columns)}")

                    # Проверяем наличие нужных колонок
                    if 'text' not in df_full.columns:
                        print("  ⚠️  Колонка 'text' не найдена!")
                    if 'label' not in df_full.columns:
                        print("  ⚠️  Колонка 'label' не найдена!")

                except Exception as e:
                    print(f"  ⚠️  Ошибка при чтении файла: {e}")
            else:
                all_exist = False
                # Показываем, что есть в директории
                if path.parent.exists():
                    print(f"  Файлы в папке {path.parent}:")
                    for f in path.parent.iterdir():
                        print(f"    - {f.name}")

            print()

        print(f"Текущая рабочая директория: {os.getcwd()}")
        print(f"Абсолютный путь к папке data: {self.DATA_DIR.absolute()}")
        print("=" * 50)

        return all_exist

    # Создаем папку data, если ее нет
    if not DATA_DIR.exists():
        print(f"Создаем папку 'data'...")
        DATA_DIR.mkdir(exist_ok=True)

    # Пути для сохранения моделей
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)

    SEMANTIC_MODEL_PATH = MODELS_DIR / "semantic_model"
    STYLOMETRIC_MODEL_PATH = MODELS_DIR / "stylometric_model.pkl"
    SCALER_PATH = MODELS_DIR / "scaler.pkl"
    META_MODEL_PATH = MODELS_DIR / "meta_model.pkl"
    FEATURES_DIR = MODELS_DIR / "features"
    FEATURES_DIR.mkdir(exist_ok=True)

    # Параметры модели
    SEMANTIC_MODEL_NAME = "distilbert-base-uncased"  # Легкая и быстрая модель
    USE_AMP = True
    MAX_LENGTH = 256  # Уменьшаем для экономии памяти
    SEMANTIC_BATCH_SIZE = 8  # Уменьшаем batch size
    SEMANTIC_EPOCHS = 3
    LEARNING_RATE = 2e-5

    # Параметры для перплексии
    AR_MODEL_NAME = "distilgpt2"  # Легкая модель GPT-2
    MLM_MODEL_NAME = "distilbert-base-uncased"  # Легкая модель BERT

    # Параметры для стилометрических признаков
    STYLOMETRIC_FEATURES = [
        'text_length',
        'ttr',  # Type-Token Ratio
        'avg_sentence_length',
        'std_sentence_length',
        'avg_word_length',
        'comma_freq',
        'period_freq',
        'question_freq',
        'exclamation_freq',
        'noun_ratio',
        'verb_ratio',
        'adjective_ratio',
        'adverb_ratio',
        'pronoun_ratio',
        'num_unique_words',
        'num_sentences',
        'num_words',
        'capital_ratio',
        'digit_ratio'
    ]

    # Настройки устройства
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Настройки для мета-классификатора
    META_MODEL_TYPE = "lightgbm"  # "lightgbm" или "xgboost"


config = Config()