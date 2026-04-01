import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table
import numpy as np

# Данные для таблицы (лучшие примеры из твоих результатов)
table_data = [
    [" Человек", "Привет, Иван! Давай встретимся завтра в 15:00...", "0.50", "0.19", " ТЕКСТ ЧЕЛОВЕКА"],
    [" Человек", "Поздравляю с днем рождения! Желаю здоровья...", "0.50", "0.12", " ТЕКСТ ЧЕЛОВЕКА"],
    [" Человек", "Сегодня отличная погода, давай сходим в парк...", "0.50", "0.13", " ТЕКСТ ЧЕЛОВЕКА"],
    [" AI (легит.)", "Искусственный интеллект развивается стремительно...", "0.58", "0.15", " AI-ТЕКСТ (не атака)"],
    [" AI (легит.)", "Сегодня прекрасная погода. Солнце светит...", "0.58", "0.15", " AI-ТЕКСТ (не атака)"],
    [" BEC атака", "СРОЧНО! Переведите 50000 руб. на счет 884-332-111...", "0.70", "0.93", " ДИПФЕЙК-АТАКА"],
    [" BEC атака", "Генеральный директор просит срочно перевести 500000...", "0.70", "0.84", " ДИПФЕЙК-АТАКА"],
    [" BEC атака", "Здравствуйте, я новый финансовый директор...", "0.75", "0.82", " ДИПФЕЙК-АТАКА"],
    [" Фишинг", "Ваш аккаунт скомпрометирован! Подтвердите пароль...", "0.58", "0.63", " ДИПФЕЙК-АТАКА"],
    [" Фишинг", "СРОЧНО! Клиент перевел деньги, нужна верификация...", "0.53", "0.60", " ДИПФЕЙК-АТАКА"],
]

# Создаем фигуру
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Создаем таблицу
table = ax.table(cellText=table_data,
                  colLabels=["Тип текста", "Текст (фрагмент)", "AI\nвероятн.", "Attack\nвероятн.", "Вердикт"],
                  loc='center',
                  cellLoc='left',
                  colWidths=[0.12, 0.45, 0.08, 0.08, 0.2])

# Настройка стиля
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Цвета для строк
colors = {
    ' Человек': '#e8f5e9',
    ' AI (легит.)': '#e3f2fd',
    ' BEC атака': '#ffebee',
    ' Фишинг': '#ffebee',
}

for i, row in enumerate(table_data):
    for j in range(5):
        cell = table[(i+1, j)]
        text_type = row[0]
        if 'Человек' in text_type:
            cell.set_facecolor('#e8f5e9')
        elif 'AI' in text_type:
            cell.set_facecolor('#e3f2fd')
        elif 'атака' in text_type:
            cell.set_facecolor('#ffebee')
        
        # Жирный шрифт для вердикта
        if j == 4:
            if 'ДИПФЕЙК' in row[4]:
                cell.set_text_props(weight='bold', color='#d32f2f')
            elif 'ЧЕЛОВЕКА' in row[4]:
                cell.set_text_props(weight='bold', color='#388e3c')
            elif 'AI' in row[4]:
                cell.set_text_props(weight='bold', color='#1976d2')

# Заголовок
ax.set_title("РЕЗУЛЬТАТЫ ДЕТЕКЦИИ ТЕКСТОВЫХ ДИПФЕЙК-АТАК", 
             fontsize=16, fontweight='bold', pad=20)

# Легенда
legend_elements = [
    mpatches.Patch(facecolor='#e8f5e9', label=' Человеческие тексты'),
    mpatches.Patch(facecolor='#e3f2fd', label=' AI-тексты (легитимные)'),
    mpatches.Patch(facecolor='#ffebee', label=' Дипфейк-атаки'),
]

ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)

plt.tight_layout()
plt.savefig('deepfake_detection_results.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()

print("✅ Таблица сохранена как 'deepfake_detection_results.png'")

# ==============================
# КРАСИВАЯ СВОДНАЯ СТАТИСТИКА ДЛЯ ПРЕЗЕНТАЦИИ
# ==============================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Данные из твоих результатов
categories = ['Человеческие\nтексты', 'AI-тексты\n(легитимные)', 'BEC атаки', 'Фишинг\nатаки', 'Соц. инженерия']
detection_rate = [40, 20, 83, 50, 0]  # Точность детекции по типам
colors = ['#4caf50', '#2196f3', '#f44336', '#ff9800', '#9c27b0']

# Создаем фигуру с двумя подграфиками
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# График 1: Точность детекции по типам
bars = axes[0].bar(categories, detection_rate, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Порог 50%')
axes[0].set_ylabel('Точность детекции (%)', fontsize=12)
axes[0].set_title('Точность детекции по типам текстов', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 100)
axes[0].grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
for bar, val in zip(bars, detection_rate):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[0].legend()

# График 2: Круговая диаграмма - распределение атак по типам
attack_counts = [6, 6, 3]  # BEC, Фишинг, Соц. инженерия
attack_labels = ['BEC атаки\n(6 текстов)', 'Фишинг атаки\n(6 текстов)', 'Социальная инженерия\n(3 текста)']
attack_colors = ['#f44336', '#ff9800', '#9c27b0']

wedges, texts, autotexts = axes[1].pie(attack_counts, labels=attack_labels, colors=attack_colors,
                                        autopct='%1.0f%%', startangle=90,
                                        textprops={'fontsize': 11})
axes[1].set_title('Распределение дипфейк-атак по типам', fontsize=14, fontweight='bold')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('detection_statistics.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()

print("✅ Статистика сохранена как 'detection_statistics.png'")