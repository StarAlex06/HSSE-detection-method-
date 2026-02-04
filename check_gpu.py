import torch
import sys

print("=" * 60)
print("ПРОВЕРКА GPU И УСТАНОВКИ PyTorch")
print("=" * 60)

# 1. Проверяем версию Python
print(f"Python версия: {sys.version}")

# 2. Проверяем PyTorch
print(f"\nPyTorch версия: {torch.__version__}")

# 3. Проверяем CUDA
print(f"CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU НАЙДЕНА!")
    print(f"Количество GPU: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Память: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        print(
            f"  Вычислительная способность: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")

    # Текущее использование памяти
    print(f"\nТекущее использование памяти GPU:")
    print(f"  Выделено: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"  Зарезервировано: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    # Проверяем версию CUDA toolkit
    print(f"\nВерсия CUDA в PyTorch: {torch.version.cuda}")

else:
    print("❌ GPU НЕ НАЙДЕНА!")
    print("\nВозможные причины:")
    print("1. Видеокарта NVIDIA не установлена")
    print("2. Драйверы NVIDIA не установлены или устарели")
    print("3. PyTorch установлен без поддержки CUDA")
    print("4. CUDA toolkit не установлен")
    print("5. Видеокарта не поддерживает CUDA")

# 4. Проверяем, какая версия PyTorch установлена
print(f"\nПуть к torch: {torch.__file__}")

# 5. Проверяем, есть ли cuda в сборке
print(f"\nСборка PyTorch:")
print(f"  С CUDA: {torch.cuda.is_available()}")
print(f"  Версия CUDA: {torch.version.cuda}")
print(f"  Версия cuDNN: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")

print("\n" + "=" * 60)