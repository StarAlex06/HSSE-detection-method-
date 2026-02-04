import subprocess
import sys


def install_for_py313():
    """Устанавливает совместимые версии для Python 3.13."""

    print("Установка для Python 3.13...")

    # Устанавливаем PyTorch без строгой версии
    print("\n1. Установка PyTorch...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])

    # Базовые пакеты
    print("\n2. Установка базовых пакетов...")
    base_packages = [
        "transformers>=4.30.0",
        "scikit-learn>=1.2.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "nltk>=3.8.0",
        "lightgbm>=4.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "joblib>=1.2.0",
        "colorama>=0.4.6",
        "openpyxl>=3.1.0",
        "sentencepiece>=0.1.99",
        "tokenizers>=0.13.0"
    ]

    for pkg in base_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            print(f"  ✅ {pkg}")
        except:
            print(f"  ⚠️  {pkg}")

    # NLTK данные
    print("\n3. Установка NLTK данных...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')
        nltk.download('wordnet')
        print("  ✅ NLTK данные")
    except Exception as e:
        print(f"  ⚠️  NLTK: {e}")

    print("\n✅ Установка завершена!")


if __name__ == "__main__":
    install_for_py313()