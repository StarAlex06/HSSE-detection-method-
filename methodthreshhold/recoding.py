import pandas as pd
import csv
import os

print("="*50)
print("КОНВЕРТАЦИЯ ДАТАСЕТА")
print("="*50)

# Проверяем существующие файлы
print("Проверка файлов:")
print(f"test.csv существует: {os.path.exists('test.csv')}")

# Читаем исходный файл с правильными параметрами
print("\nЧтение исходного файла...")
try:
    # Читаем, пропуская первые 2 строки, разделитель ;
    df = pd.read_csv('test.csv', 
                     encoding='utf-8',
                     sep=';',
                     skiprows=2,
                     header=None,
                     names=['text', 'label'],
                     quoting=csv.QUOTE_NONE)  # Не обрабатываем кавычки специально
    
    print(f"Успешно прочитано {len(df)} строк")
    print("\nПервые 3 строки данных:")
    for i in range(min(3, len(df))):
        print(f"  {i+1}. Текст: {df['text'].iloc[i][:50]}... -> Метка: {df['label'].iloc[i]}")
    
except Exception as e:
    print(f"Ошибка при чтении: {e}")
    exit(1)

# Создаем файл для модели
print("\nСоздание файла для модели...")

# Вариант А: С разделителем-запятой (стандартный для pandas)
df.to_csv('test_for_model.csv', 
          encoding='utf-8',
          sep=',',
          index=False)

# Вариант Б: С явным указанием кавычек (надежнее)
with open('test_for_model_fixed.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(['text', 'label'])  # Заголовки
    for _, row in df.iterrows():
        writer.writerow([row['text'], row['label']])

print("\nСозданы файлы:")
print("1. test_for_model.csv - стандартный CSV с запятыми")
print("2. test_for_model_fixed.csv - CSV с кавычками вокруг всех полей")

# Проверяем созданные файлы
print("\nПроверка созданных файлов:")
for filename in ['test_for_model.csv', 'test_for_model_fixed.csv']:
    try:
        df_check = pd.read_csv(filename, encoding='utf-8')
        print(f"{filename}: {len(df_check)} строк, колонки: {list(df_check.columns)}")
    except Exception as e:
        print(f"{filename}: ОШИБКА - {e}")

print("\n" + "="*50)
print("ГОТОВО! Используйте test_for_model_fixed.csv в скрипте 3_evaluate.py")
print("="*50)


