import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
import warnings
import os  # Добавьте этот импорт

warnings.filterwarnings('ignore')

# Создаем папку для визуализаций, если её нет
os.makedirs('visuals', exist_ok=True)

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


# ==================== ЗАГРУЗКА ДАННЫХ И МОДЕЛЕЙ ====================
print("Загрузка данных и моделей...")

# Загружаем тестовые данные
df_test = pd.read_csv('test.csv', sep=';')
texts_test = df_test['text'].tolist()
y_test = df_test['label'].values

# Загружаем модели
model_style = joblib.load('model_style.pkl')
model_semantic = joblib.load('model_semantic.pkl')
model_perplexity = joblib.load('model_perplexity.pkl')
model_stability = joblib.load('model_stability.pkl')

# Загружаем scaler'ы
scaler_style = joblib.load('scaler_style.pkl')
scaler_semantic = joblib.load('scaler_semantic.pkl')
scaler_perplexity = joblib.load('scaler_perplexity.pkl')
scaler_stability = joblib.load('scaler_stability.pkl')

# Загружаем пороги
with open('thresholds.json', 'r') as f:
    thresholds = json.load(f)

# ==================== ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ ====================
print("Получение предсказаний...")

# Для одного текста используем функции из utils
from utils import extract_style_features, extract_semantic_features, extract_perplexity_features, extract_stability_features

# Собираем предсказания для всех тестовых текстов
X_style_test = np.array([extract_style_features(t) for t in texts_test])
X_semantic_test = np.array([extract_semantic_features(t) for t in texts_test])
X_perplexity_test = np.array([extract_perplexity_features(t) for t in texts_test])
X_stability_test = np.array([extract_stability_features(t) for t in texts_test])

# Масштабируем
X_style_test_scaled = scaler_style.transform(X_style_test)
X_semantic_test_scaled = scaler_semantic.transform(X_semantic_test)
X_perplexity_test_scaled = scaler_perplexity.transform(X_perplexity_test)
X_stability_test_scaled = scaler_stability.transform(X_stability_test)

# Получаем вероятности для каждой модели
p_style = model_style.predict_proba(X_style_test_scaled)[:, 1]
p_semantic = model_semantic.predict_proba(X_semantic_test_scaled)[:, 1]
p_perplexity = model_perplexity.predict_proba(X_perplexity_test_scaled)[:, 1]
p_stability = model_stability.predict_proba(X_stability_test_scaled)[:, 1]

# Финальное предсказание с порогами
final_pred = (
    (p_style > thresholds['style']) | 
    (p_semantic > thresholds['semantic']) | 
    (p_perplexity > thresholds['perplexity']) | 
    (p_stability > thresholds['stability'])
).astype(int)

# ==================== 1. МАТРИЦА ОШИБОК (CONFUSION MATRIX) ====================
print("Создание матрицы ошибок...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Матрица для итоговой модели
cm = confusion_matrix(y_test, final_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Человек', 'ИИ'],
            yticklabels=['Человек', 'ИИ'])
axes[0].set_title('Матрица ошибок - Итоговая модель', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Предсказано', fontsize=12)
axes[0].set_ylabel('Реальность', fontsize=12)

# Нормированная матрица (проценты)
cm_norm = confusion_matrix(y_test, final_pred, normalize='true')
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
            xticklabels=['Человек', 'ИИ'],
            yticklabels=['Человек', 'ИИ'])
axes[1].set_title('Матрица ошибок (нормированная)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Предсказано', fontsize=12)
axes[1].set_ylabel('Реальность', fontsize=12)

plt.tight_layout()
plt.savefig('visuals/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 2. ROC-КРИВЫЕ ДЛЯ ВСЕХ МОДЕЛЕЙ ====================
print("Построение ROC-кривых...")

fig, ax = plt.subplots(figsize=(12, 8))

models = [
    (p_style, 'Стилометрия', 'blue'),
    (p_semantic, 'Семантика', 'red'),
    (p_perplexity, 'Перплексия', 'green'),
    (p_stability, 'Стабильность', 'orange'),
    (final_pred, 'Итоговая модель', 'purple')
]

for probs, name, color in models:
    if name == 'Итоговая модель':
        # Для бинарных предсказаний своя логика
        fpr, tpr, _ = roc_curve(y_test, probs)
    else:
        fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color=color, lw=2, 
            label=f'{name} (AUC = {roc_auc:.3f})')

# Диагональная линия (случайное угадывание)
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Случайное угадывание')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (Ложные срабатывания)', fontsize=12)
ax.set_ylabel('True Positive Rate (Верные обнаружения)', fontsize=12)
ax.set_title('ROC-кривые для всех моделей', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visuals/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 3. КАЛИБРОВОЧНЫЕ КРИВЫЕ ====================
print("Построение калибровочных кривых...")

fig, ax = plt.subplots(figsize=(12, 8))

for probs, name, color in [
    (p_style, 'Стилометрия', 'blue'),
    (p_semantic, 'Семантика', 'red'),
    (p_perplexity, 'Перплексия', 'green'),
    (p_stability, 'Стабильность', 'orange')
]:
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, probs, n_bins=10, strategy='uniform'
    )
    
    ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
            color=color, lw=2, label=name, markersize=8)

# Идеально калиброванная модель
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Идеальная калибровка')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel('Средняя предсказанная вероятность', fontsize=12)
ax.set_ylabel('Фактическая доля ИИ-текстов', fontsize=12)
ax.set_title('Калибровочные кривые моделей', fontsize=14, fontweight='bold')
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visuals/calibration_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 4. РАСПРЕДЕЛЕНИЕ ВЕРОЯТНОСТЕЙ ====================
print("Построение распределений вероятностей...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

model_data = [
    (p_style, 'Стилометрия', axes[0, 0]),
    (p_semantic, 'Семантика', axes[0, 1]),
    (p_perplexity, 'Перплексия', axes[1, 0]),
    (p_stability, 'Стабильность', axes[1, 1])
]

for probs, name, ax in model_data:
    # Разделяем по реальным классам
    probs_human = probs[y_test == 0]
    probs_ai = probs[y_test == 1]
    
    # Строим гистограммы
    ax.hist(probs_human, bins=30, alpha=0.7, label='Человек', 
            color='green', density=True)
    ax.hist(probs_ai, bins=30, alpha=0.7, label='ИИ', 
            color='red', density=True)
    
    # Добавляем порог, если есть
    if name.lower() in thresholds:
        threshold = thresholds[name.lower()]
        ax.axvline(x=threshold, color='blue', linestyle='--', 
                   linewidth=2, label=f'Порог: {threshold}')
    
    ax.set_xlabel('Вероятность ИИ', fontsize=11)
    ax.set_ylabel('Плотность', fontsize=11)
    ax.set_title(f'{name} - распределение вероятностей', fontsize=12, fontweight='bold')
    ax.legend(loc='upper center', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visuals/probability_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 5. ВАЖНОСТЬ ПРИЗНАКОВ ====================
print("Визуализация важности признаков...")

# Собираем веса из моделей
feature_names_style = ['Ср. длина предл.', 'Лекс. разнообразие', 
                       'Доля пунктуации', 'Ср. длина слова', 
                       'Доля длинных слов', 'Std длина предл.']
feature_names_semantic = [
    'Среднее значение эмбеддинга',
    'Стандартное отклонение эмбеддинга',
    'Норма эмбеддинга',
    'Максимум эмбеддинга',
    'Минимум эмбеддинга',
    'Средняя семантическая близость предложений'
]
feature_names_perplexity = [
    'Перплексия (норм.)',
    'Лог-перплексия (норм.)',
    'Энтропия последовательности',
    'Дисперсия потерь токенов'
]
feature_names_stability = ['Стаб. предложений', 'Богатство частей речи', 'Повторяемость']

# Коэффициенты (веса) из логистической регрессии
coef_style = model_style.coef_[0]
coef_semantic = model_semantic.coef_[0]
coef_perplexity = model_perplexity.coef_[0]
coef_stability = model_stability.coef_[0]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Стилометрия
ax = axes[0, 0]
ax.clear()  # Очищаем предыдущие настройки
colors = ['red' if x > 0 else 'green' for x in coef_style]
bars = ax.barh(range(len(coef_style)), coef_style, color=colors, alpha=0.7)
ax.set_yticks(range(len(coef_style)))
ax.set_yticklabels(feature_names_style, fontsize=10)
ax.set_xlabel('Вес признака (>0: указывает на ИИ, <0: на человека)', fontsize=11)
ax.set_title('Стилометрия - важность признаков', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

# Семантика
ax = axes[0, 1]
ax.clear()

colors = ['red' if x > 0 else 'green' for x in coef_semantic]

bars = ax.barh(range(len(coef_semantic)), coef_semantic,
               color=colors, alpha=0.7)

ax.set_yticks(range(len(coef_semantic)))
ax.set_yticklabels(feature_names_semantic, fontsize=10)

ax.set_xlabel('Вес признака', fontsize=11)
ax.set_title('Семантика - важность признаков', fontsize=12, fontweight='bold')

ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

# Перплексия
ax = axes[1, 0]
ax.clear()  # Очищаем предыдущие настройки
colors = ['red' if x > 0 else 'green' for x in coef_perplexity]
bars = ax.barh(range(len(coef_perplexity)), coef_perplexity, color=colors, alpha=0.7)
ax.set_yticks(range(len(coef_perplexity)))
ax.set_yticklabels(feature_names_perplexity, fontsize=10)
ax.set_xlabel('Вес признака', fontsize=11)
ax.set_title('Перплексия - важность признаков', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

# Стабильность
ax = axes[1, 1]
ax.clear()  # Очищаем предыдущие настройки
colors = ['red' if x > 0 else 'green' for x in coef_stability]
bars = ax.barh(range(len(coef_stability)), coef_stability, color=colors, alpha=0.7)
ax.set_yticks(range(len(coef_stability)))
ax.set_yticklabels(feature_names_stability, fontsize=10)
ax.set_xlabel('Вес признака', fontsize=11)
ax.set_title('Стабильность - важность признаков', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('visuals/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()



# ==================== 6. СРАВНЕНИЕ ПОРОГОВ ====================
print("Визуализация порогов...")

fig, ax = plt.subplots(figsize=(12, 6))

models_list = list(thresholds.keys())
threshold_values = list(thresholds.values())
colors = ['blue', 'red', 'green', 'orange']

bars = ax.bar(models_list, threshold_values, color=colors, alpha=0.7)

# Добавляем значения на столбцы
for bar, val in zip(bars, threshold_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim(0, 1)
ax.set_ylabel('Значение порога', fontsize=12)
ax.set_xlabel('Модель', fontsize=12)
ax.set_title('Оптимальные пороги для каждой модели', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visuals/thresholds_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 7. ПРОИЗВОДИТЕЛЬНОСТЬ МОДЕЛЕЙ ====================
print("Сравнение производительности моделей...")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Считаем метрики для каждой модели (с порогами)
models_preds = {
    'Стилометрия': (p_style > thresholds['style']).astype(int),
    'Семантика': (p_semantic > thresholds['semantic']).astype(int),
    'Перплексия': (p_perplexity > thresholds['perplexity']).astype(int),
    'Стабильность': (p_stability > thresholds['stability']).astype(int),
    'Итоговая': final_pred
}

metrics = {}
for name, pred in models_preds.items():
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1-Score': f1_score(y_test, pred)
    }

# Создаем DataFrame для удобства
df_metrics = pd.DataFrame(metrics).T

fig, ax = plt.subplots(figsize=(14, 8))

# Тепловая карта метрик
sns.heatmap(df_metrics, annot=True, fmt='.3f', cmap='YlOrRd', 
            ax=ax, vmin=0, vmax=1, linewidths=0.5)
ax.set_title('Сравнение производительности моделей', fontsize=14, fontweight='bold')
ax.set_xlabel('Метрика', fontsize=12)
ax.set_ylabel('Модель', fontsize=12)

plt.tight_layout()
plt.savefig('visuals/model_performance.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 8. ДИАГРАММА ВЕННА (ПЕРЕСЕЧЕНИЕ ОШИБОК) ====================
print("Построение диаграммы Венна...")

try:
    from matplotlib_venn import venn3, venn3_circles
    
    # Находим ошибки каждой модели
    errors_style = set(np.where((p_style > thresholds['style']) != y_test)[0])
    errors_semantic = set(np.where((p_semantic > thresholds['semantic']) != y_test)[0])
    errors_perplexity = set(np.where((p_perplexity > thresholds['perplexity']) != y_test)[0])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    venn = venn3([errors_style, errors_semantic, errors_perplexity],
                 ('Стилометрия', 'Семантика', 'Перплексия'))
    
    ax.set_title('Пересечение ошибок моделей', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visuals/venn_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()
except ImportError:
    print("Для диаграммы Венна установите: pip install matplotlib-venn")

# ==================== 9. СВОДНАЯ СТАТИСТИКА ====================
print("Создание сводной статистики...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.axis('tight')

# Собираем статистику
stats_data = [
    ['Метрика', 'Значение'],
    ['Всего текстов', f'{len(y_test)}'],
    ['Человеческих', f'{sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)'],
    ['ИИ-текстов', f'{sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)'],
    ['Accuracy', f'{accuracy_score(y_test, final_pred):.3f}'],
    ['Precision', f'{precision_score(y_test, final_pred):.3f}'],
    ['Recall', f'{recall_score(y_test, final_pred):.3f}'],
    ['F1-Score', f'{f1_score(y_test, final_pred):.3f}'],
    ['Порог стилометрии', f'{thresholds["style"]:.2f}'],
    ['Порог семантики', f'{thresholds["semantic"]:.2f}'],
    ['Порог перплексии', f'{thresholds["perplexity"]:.2f}'],
    ['Порог стабильности', f'{thresholds["stability"]:.2f}'],
]

table = ax.table(cellText=stats_data, loc='center', cellLoc='left', colWidths=[0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

# Стилизация
for i, key in enumerate(stats_data):
    if i == 0:
        for j in range(2):
            table[(i, j)].set_facecolor('#4472C4')
            table[(i, j)].set_text_props(weight='bold', color='white')
    elif i % 2 == 1:
        for j in range(2):
            table[(i, j)].set_facecolor('#D9E1F2')

ax.set_title('Сводная статистика модели', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('visuals/summary_statistics.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 10. СОЗДАНИЕ ОТЧЕТА ====================
print("Создание HTML-отчета...")

html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Отчет по детекции ИИ-текстов</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        h1 {{ color: #333; text-align: center; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        .metrics {{ display: flex; flex-wrap: wrap; justify-content: space-around; margin: 30px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; min-width: 150px; text-align: center; margin: 10px; }}
        .metric-value {{ font-size: 36px; font-weight: bold; }}
        .metric-label {{ font-size: 14px; opacity: 0.9; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin: 20px 0; }}
        .image-card {{ background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .image-card img {{ width: 100%; height: auto; display: block; }}
        .image-card p {{ padding: 15px; margin: 0; background: #f8f9fa; font-weight: bold; text-align: center; }}
        .thresholds {{ display: flex; justify-content: space-around; margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .threshold-item {{ text-align: center; }}
        .threshold-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Отчет по детекции ИИ-текстов</h1>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{accuracy_score(y_test, final_pred):.3f}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{precision_score(y_test, final_pred):.3f}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{recall_score(y_test, final_pred):.3f}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{f1_score(y_test, final_pred):.3f}</div>
                <div class="metric-label">F1-Score</div>
            </div>
        </div>
        
        <h2>⚙️ Оптимальные пороги</h2>
        <div class="thresholds">
            <div class="threshold-item">
                <div>Стилометрия</div>
                <div class="threshold-value">{thresholds['style']:.2f}</div>
            </div>
            <div class="threshold-item">
                <div>Семантика</div>
                <div class="threshold-value">{thresholds['semantic']:.2f}</div>
            </div>
            <div class="threshold-item">
                <div>Перплексия</div>
                <div class="threshold-value">{thresholds['perplexity']:.2f}</div>
            </div>
            <div class="threshold-item">
                <div>Стабильность</div>
                <div class="threshold-value">{thresholds['stability']:.2f}</div>
            </div>
        </div>
        
        <h2>📈 Визуализации</h2>
        <div class="image-grid">
            <div class="image-card">
                <img src="visuals/confusion_matrix.png" alt="Confusion Matrix">
                <p>Матрица ошибок</p>
            </div>
            <div class="image-card">
                <img src="visuals/roc_curves.png" alt="ROC Curves">
                <p>ROC-кривые</p>
            </div>
            <div class="image-card">
                <img src="visuals/calibration_curves.png" alt="Calibration Curves">
                <p>Калибровочные кривые</p>
            </div>
            <div class="image-card">
                <img src="visuals/probability_distributions.png" alt="Probability Distributions">
                <p>Распределения вероятностей</p>
            </div>
            <div class="image-card">
                <img src="visuals/feature_importance.png" alt="Feature Importance">
                <p>Важность признаков</p>
            </div>
            <div class="image-card">
                <img src="visuals/thresholds_comparison.png" alt="Thresholds">
                <p>Сравнение порогов</p>
            </div>
            <div class="image-card">
                <img src="visuals/model_performance.png" alt="Model Performance">
                <p>Производительность моделей</p>
            </div>
            <div class="image-card">
                <img src="visuals/summary_statistics.png" alt="Summary">
                <p>Сводная статистика</p>
            </div>
        </div>
        
        <p style="text-align: center; margin-top: 30px; color: #999;">
            Сгенерировано автоматически | Всего протестировано текстов: {len(y_test)}
        </p>
    </div>
</body>
</html>
"""

with open('visuals/report.html', 'w', encoding='utf-8') as f:
    f.write(html_report)

print("Отчет сохранен в visuals/report.html")
print(" Все графики сохранены в папке visuals/")
