import time
import sys
from pathlib import Path

print("=" * 80)
print("🚀 HSSE - ПОЛНЫЙ ПАЙПЛАЙН ОБНАРУЖЕНИЯ AI-ТЕКСТОВ НА GPU")
print("=" * 80)
print("Метод: Hybrid Semantic-Stylometric Ensemble with Stability Estimation")
print("Признаки: 1) Semantic Score, 2) Stylometric Score, 3) Perplexity Gap, 4) Stability Score")
print("=" * 80)

# Импортируем конфиг
from config_gpu import config

print(f"\n🎯 Конфигурация:")
print(f"   Устройство: {config.DEVICE}")
print(f"   Семантическая модель: {config.SEMANTIC_MODEL_NAME}")
print(f"   Batch size: {config.SEMANTIC_BATCH_SIZE}")
print(f"   Max length: {config.MAX_LENGTH}")
print(f"   Mixed Precision: {config.USE_AMP}")
print(f"   Признаки HSSE: 4")


def run_pipeline():
    """Запускает полный пайплайн HSSE."""

    steps = [
        {
            "name": "1. Обучение семантической модели",
            "module": "train_semantic_gpu",
            "function": "train_semantic_model_gpu"
        },
        {
            "name": "2. Обучение стилометрической модели",
            "module": "train_stylometric_gpu",
            "function": "train_stylometric_model_gpu"
        },
        {
            "name": "3. Извлечение признаков HSSE",
            "module": "extract_features_gpu",
            "function": "extract_hsse_features"
        },
        {
            "name": "4. Обучение мета-классификатора",
            "module": "train_meta_gpu",
            "function": "train_meta_classifier_gpu"
        },
        {
            "name": "5. Финальная оценка",
            "module": "evaluate_gpu",
            "function": "evaluate_hsse_gpu"
        }
    ]

    print("\n📋 Шаги пайплайна:")
    for step in steps:
        print(f"   {step['name']}")

    print("\n" + "=" * 80)
    print("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА HSSE")
    print("=" * 80)

    total_start = time.time()

    for i, step in enumerate(steps, 1):
        print(f"\n▶️  Шаг {i}/{len(steps)}: {step['name']}")
        print("-" * 60)

        step_start = time.time()

        try:
            # Динамический импорт
            module = __import__(step['module'])
            func = getattr(module, step['function'])

            # Запуск
            func()

            step_time = time.time() - step_start
            print(f"✅ Завершено за {step_time / 60:.1f} минут")

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️  Пайплайн остановлен на шаге {i}")
            break

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("🎉 ПАЙПЛАЙН HSSE ЗАВЕРШЕН!")
    print("=" * 80)
    print(f"Общее время: {total_time / 3600:.1f} часов")
    print(f"\n📁 Модели сохранены в: {config.MODELS_DIR}")
    print(f"📊 Признаки сохранены в: {config.FEATURES_DIR}")

    print("\n🔧 Использование детектора:")
    print("""
from evaluate_gpu import HSSEDetectorGPU

# Инициализация
detector = HSSEDetectorGPU()

# Предсказание для одного текста
text = "Your text here"
result = detector.predict(text, return_proba=True)
print(f"Вероятность AI: {result:.3f}")
    """)

    print("=" * 80)


if __name__ == "__main__":
    # Проверяем наличие данных
    if not config.check_data_files():
        print("❌ Ошибка: файлы данных не найдены!")
        print("Убедитесь, что в папке 'data' есть файлы train.csv, val.csv, test.csv")
        sys.exit(1)

    # Запускаем пайплайн
    run_pipeline()