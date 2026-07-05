# HSSE Detection Method

Метод для определения AI-сгенерированных текстов по четырём независимым группам признаков:

1. **Stylometric** - длина предложений и слов, лексическое разнообразие, пунктуация.
2. **Semantic** - статистики sentence embeddings и связность соседних предложений.
3. **Perplexity** - признаки на основе языковой модели GPT-2.
4. **Stability** - устойчивость структуры текста: вариативность предложений, повторяемость слов, простые морфологические признаки.

Каждая группа признаков обучает отдельную логистическую регрессию. Итоговое решение принимается по правилу **OR**: текст считается AI-сгенерированным, если хотя бы одна модель превысила свой оптимальный порог.

## Структура проекта

```text
.
├── README.md
├── requirements.txt
└── methodthreshhold/
    ├── 1_train_models.py       # обучение 4 моделей и scaler'ов
    ├── 2_find_thresholds.py    # подбор порогов на validation split
    ├── 3_evaluate.py           # оценка на test split
    ├── 4_predict.py            # предсказание для одного текста
    ├── 5_visualize.py          # графики и HTML-отчёт
    ├── utils.py                # извлечение признаков
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── thresholds.json
    ├── model_*.pkl
    ├── scaler_*.pkl
    └── visuals/
```

> Папка `methodthreshhold` оставлена как есть, чтобы не ломать уже сохранённые артефакты и привычные пути. Рабочий метод находится именно в ней.

## Установка

Рекомендуется использовать отдельное окружение Python 3.10+.

```bash
pip install -r requirements.txt
```

Первый запуск скачает модели `paraphrase-multilingual-MiniLM-L12-v2` и `gpt2` из Hugging Face. Если доступна CUDA, вычисления пойдут на GPU, иначе на CPU.

## Данные

Скрипты ожидают CSV-файлы внутри `methodthreshhold/`:

- `train.csv` - обучение моделей;
- `val.csv` - подбор порогов;
- `test.csv` - финальная оценка.

Обязательные колонки:

- `text` - текст;
- `label` - метка класса, где `0` означает человек, `1` означает AI.

В текущем проекте `train.csv` и `val.csv` используют запятую как разделитель, а `test.csv` использует точку с запятой. Это уже учтено в скриптах оценки и визуализации.

## Запуск полного пайплайна

Все команды ниже выполняются из папки метода:

```bash
cd methodthreshhold
```

1. Обучить модели:

```bash
python 1_train_models.py
```

Скрипт сохранит:

- `model_style.pkl`
- `model_semantic.pkl`
- `model_perplexity.pkl`
- `model_stability.pkl`
- `scaler_style.pkl`
- `scaler_semantic.pkl`
- `scaler_perplexity.pkl`
- `scaler_stability.pkl`

2. Подобрать пороги на validation split:

```bash
python 2_find_thresholds.py
```

Результат сохраняется в `thresholds.json`.

3. Оценить качество на test split:

```bash
python 3_evaluate.py
```

4. Построить графики и HTML-отчёт:

```bash
python 5_visualize.py
```

Результаты сохраняются в `methodthreshhold/visuals/`.

## Предсказание одного текста

Можно передать путь к текстовому файлу:

```bash
python 4_predict.py path/to/text.txt
```

Или запустить без аргументов и вставить текст вручную:

```bash
python 4_predict.py
```

Скрипт выводит итоговый класс, вероятности по четырём моделям, пороги и признаки, которые сработали.

## Что было убрано

Из проекта удалены старые экспериментальные ветки и дубли:

- `method1/` - отдельный LightGBM-эксперимент, не связанный с текущим пороговым методом;
- `deepfakemethod/*.py` - демонстрационный детектор deepfake-атак, другая задача и другая логика;
- `data/` - дублирующие CSV для старого эксперимента;
- служебные файлы `.idea/`;
- временные тестовые `.txt` и диагностические скрипты, не входящие в основной пайплайн.

Основной поддерживаемый метод теперь один: `methodthreshhold/1_train_models.py` -> `2_find_thresholds.py` -> `3_evaluate.py` -> `4_predict.py` / `5_visualize.py`.
