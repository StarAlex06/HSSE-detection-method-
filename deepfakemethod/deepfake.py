# ==============================
# РЕАЛИСТИЧНАЯ ДЕМОНСТРАЦИЯ ДЕТЕКТОРА ДИПФЕЙК-АТАК
# С ЗАГРУЗКОЙ РЕАЛЬНЫХ МОДЕЛЕЙ (ruBERT, ruGPT3)
# ==============================

import numpy as np
import pandas as pd
import re
import torch
from collections import Counter
import math
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Устанавливаем русский шрифт для графиков
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("     ЗАГРУЗКА РУССКИХ ЯЗЫКОВЫХ МОДЕЛЕЙ (может занять 2-3 минуты)")
print("="*70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# ==============================
# 1. ЗАГРУЗКА РУССКИХ МОДЕЛЕЙ (реальные модели)
# ==============================

from transformers import (
    AutoTokenizer, AutoModel,
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoModelForMaskedLM
)

print("\n1. Загрузка ruBERT (семантическая модель)...")
try:
    rubert_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    rubert_model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased").to(device)
    rubert_model.eval()
    print("   ✓ ruBERT загружен")
except:
    print("   Ошибка, используем упрощенную версию")
    rubert_tokenizer = None
    rubert_model = None

print("\n2. Загрузка ruGPT3 (для перплексии)...")
try:
    gpt2_tokenizer_ru = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    gpt2_model_ru = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2").to(device)
    gpt2_model_ru.eval()
    gpt2_tokenizer_ru.pad_token = gpt2_tokenizer_ru.eos_token
    print("    ruGPT3 загружена")
except:
    print("    Ошибка, используем упрощенную версию")
    gpt2_tokenizer_ru = None
    gpt2_model_ru = None

print("\n3. Загрузка ruBERT MLM (для перплексии)...")
try:
    bert_mlm_ru = AutoModelForMaskedLM.from_pretrained("DeepPavlov/rubert-base-cased").to(device)
    bert_mlm_ru.eval()
    print("    ruBERT MLM загружена")
except:
    print("    Ошибка, используем упрощенную версию")
    bert_mlm_ru = None

# ==============================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def simple_russian_tokenize(text):
    """Простая русская токенизация"""
    words = re.findall(r'[а-яёa-z]+', text.lower())
    return words

def get_rubert_embedding(text):
    """Получение эмбеддинга через ruBERT"""
    if rubert_tokenizer is None or rubert_model is None:
        return np.zeros(768)
    
    try:
        inputs = rubert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = rubert_model(**inputs)
        
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    except:
        return np.zeros(768)

def gpt2_perplexity_ru(text):
    """Перплексия через ruGPT3"""
    if gpt2_tokenizer_ru is None or gpt2_model_ru is None:
        return 50.0 + np.random.rand() * 30
    
    try:
        encodings = gpt2_tokenizer_ru(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = gpt2_model_ru(**encodings, labels=encodings["input_ids"])
        
        loss = outputs.loss
        return torch.exp(loss).item()
    except:
        return 60.0

def bert_perplexity_ru(text):
    """Перплексия через ruBERT MLM"""
    if bert_mlm_ru is None:
        return 40.0 + np.random.rand() * 20
    
    try:
        tokens = rubert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = bert_mlm_ru(**tokens, labels=tokens["input_ids"])
        
        loss = outputs.loss
        return torch.exp(loss).item()
    except:
        return 45.0

def perplexity_feature_ru(text):
    """Разность перплексий"""
    try:
        gpt_ppl = gpt2_perplexity_ru(text)
        bert_ppl = bert_perplexity_ru(text)
        return gpt_ppl - bert_ppl
    except:
        return 10.0

# ==============================
# 3. СТИЛОМЕТРИЧЕСКИЕ ПРИЗНАКИ (реалистичные)
# ==============================

def stylometric_features_russian(text):
    """Расширенные стилометрические признаки"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = simple_russian_tokenize(text)
    
    num_sent = len(sentences)
    num_words = len(words)
    avg_sent_len = num_words / (num_sent + 1e-6)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    
    # Пунктуация
    punctuation = len(re.findall(r'[.,!?;:—\-"\'()«»]', text))
    commas = text.count(',')
    dots = text.count('.')
    questions = text.count('?')
    exclamations = text.count('!')
    
    # Лексическое разнообразие
    unique_words = len(set(words))
    lexical_diversity = unique_words / (num_words + 1e-6)
    
    # Признаки атак
    urgent_words_ru = ['срочно', 'немедленно', 'сейчас', 'быстрее', 'срочное']
    urgent_count = sum(1 for w in words if w in urgent_words_ru)
    
    authority_words_ru = ['директор', 'начальник', 'руководство', 'генеральный']
    authority_count = sum(1 for w in words if w in authority_words_ru)
    
    digits = len(re.findall(r'\d', text))
    uppercase = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase / (len(text) + 1e-6)
    
    # Статистика предложений
    sent_lengths = [len(simple_russian_tokenize(s)) for s in sentences]
    sent_len_std = np.std(sent_lengths) if sent_lengths else 0
    
    return {
        'num_sentences': num_sent,
        'num_words': num_words,
        'avg_sent_len': avg_sent_len,
        'avg_word_len': avg_word_len,
        'punctuation': punctuation,
        'lexical_diversity': lexical_diversity,
        'urgent_count': urgent_count,
        'authority_count': authority_count,
        'digits': digits,
        'uppercase_ratio': uppercase_ratio,
        'sent_len_std': sent_len_std
    }

# ==============================
# 4. ПРИЗНАКИ ДЛЯ ДЕТЕКЦИИ АТАК
# ==============================

def malicious_intent_score_ru(text):
    text_lower = text.lower()

    # Веса
    WEIGHTS = {
        'financial': 0.50,
        'credentials': 0.35,
        'urgency': 0.15
    }

    score = 0.0

    # Финансовые паттерны
    financial_patterns = [r'\d+\s*(руб|рублей)', r'перевес', r'перевод', r'счет\s*\d+', r'оплат', r'деньги']
    if any(re.search(pattern, text_lower) for pattern in financial_patterns):
        score += WEIGHTS['financial']

    # Паттерны учетных данных
    credential_patterns = [r'парол', r'логин', r'доступ', r'код', r'аккаунт']
    if any(re.search(pattern, text_lower) for pattern in credential_patterns):
        score += WEIGHTS['credentials']

    # Слова срочности
    urgency_words = ['срочно', 'немедленно', 'сейчас же']
    if any(word in text_lower for word in urgency_words):
        score += WEIGHTS['urgency']

    return score

def social_engineering_score_ru(text):
    """Оценка социальной инженерии"""
    text_lower = text.lower()
    urgency = bool(re.search(r'(срочно|немедленно)', text_lower))
    authority = bool(re.search(r'(директор|начальник|руководство|генеральный|шеф)', text_lower))
    scarcity = bool(re.search(r'(только|последний|единственный|дедлайн)', text_lower))
    fear = bool(re.search(r'(нарушение|заблокирован|взлом|скомпрометирован)', text_lower))
    
    return (urgency * 0.3 + authority * 0.3 + scarcity * 0.2 + fear * 0.2)

def action_request_score_ru(text):
    """Оценка запросов действий"""
    text_lower = text.lower()
    critical = ['перевес', 'перевод', 'оплати', 'счет', 'пароль', 'логин', 'доступ', 'подтверди']
    count = sum(1 for word in critical if word in text_lower)
    return min(count / 6.0, 1.0)

def text_entropy_ru(text):
    """Энтропия текста"""
    words = simple_russian_tokenize(text)
    if not words:
        return 0
    word_freq = Counter(words)
    total = len(words)
    entropy = -sum((freq/total) * math.log2(freq/total) for freq in word_freq.values())
    return entropy / math.log2(total) if total > 1 else 0

def punctuation_diversity_ru(text):
    """Разнообразие пунктуации"""
    punct_chars = ['!', '?', '.', ',', ';', ':', '-', '"', '«', '»']
    punct_counts = [text.count(p) for p in punct_chars]
    return np.std(punct_counts) if sum(punct_counts) > 0 else 0

# ==============================
# 5. ДЕТЕКТОР НА ОСНОВЕ РЕАЛЬНЫХ МОДЕЛЕЙ
# ==============================

class RealisticDeepfakeDetector:
    """Реалистичный детектор с использованием реальных моделей"""
    
    def __init__(self):
        self.ai_detector_trained = False
        self.attack_detector_trained = False
        
    def extract_all_features(self, text):
        """Извлечение всех признаков с использованием реальных моделей"""
        
        # Стилометрические признаки
        style = stylometric_features_russian(text)
        style_vector = list(style.values())
        
        # Семантические признаки (ruBERT)
        semantic_vector = get_rubert_embedding(text)
        
        # Перплексия
        try:
            ppl = perplexity_feature_ru(text)
        except:
            ppl = 15.0
        
        # Признаки атак
        attack_features = [
            malicious_intent_score_ru(text),
            social_engineering_score_ru(text),
            action_request_score_ru(text),
            text_entropy_ru(text),
            punctuation_diversity_ru(text)
        ]
        
        return {
            'style': style_vector,
            'semantic': semantic_vector,
            'perplexity': ppl,
            'attack_features': attack_features,
            'style_dict': style
        }
    
    def predict_ai_probability(self, text):
        """Предсказание вероятности AI-генерации"""
        features = self.extract_all_features(text)
        
        # Реалистичная эвристика на основе реальных признаков
        score = 0.0
        weights = {'perplexity': 0.25, 'lexical': 0.2, 'semantic': 0.3, 'length': 0.15, 'punctuation': 0.1}
        
        # Перплексия (AI-тексты часто имеют более высокую перплексию)
        ppl = features['perplexity']
        if ppl > 60:
            score += weights['perplexity']
        elif ppl > 40:
            score += weights['perplexity'] * 0.6
        elif ppl > 20:
            score += weights['perplexity'] * 0.3
        
        # Лексическое разнообразие (AI-тексты часто имеют более высокое разнообразие)
        lexical = features['style_dict']['lexical_diversity']
        if lexical > 0.7:
            score += weights['lexical']
        elif lexical > 0.5:
            score += weights['lexical'] * 0.6
        
        # Длина текста (AI часто пишет длиннее)
        if features['style_dict']['num_words'] > 50:
            score += weights['length']
        elif features['style_dict']['num_words'] > 20:
            score += weights['length'] * 0.5
        
        # Наличие маркеров AI
        if features['style_dict']['uppercase_ratio'] < 0.05:
            score += 0.05
        
        # Добавляем базовую вероятность
        score = min(score + 0.25, 0.95)
        
        # Корректировка на основе признаков атак (атаки почти всегда AI)
        if sum(features['attack_features'][:3]) > 1.5:
            score = max(score, 0.7)
        
        return score
    
    def predict_attack_probability(self, text):
        """Предсказание вероятности дипфейк-атаки"""
        features = self.extract_all_features(text)
        
        # Взвешенная сумма признаков атак
        weights = [0.35, 0.30, 0.20, 0.10, 0.05]
        attack_score = sum(f * w for f, w in zip(features['attack_features'], weights))
        
        # Корректировка на основе стилометрии
        if features['style_dict']['urgent_count'] > 0:
            attack_score += 0.1
        if features['style_dict']['authority_count'] > 0:
            attack_score += 0.1
        if features['style_dict']['digits'] > 3:
            attack_score += 0.05
        
        # Если текст очень короткий, снижаем вероятность
        if features['style_dict']['num_words'] < 5:
            attack_score *= 0.3
        
        return min(attack_score, 0.98)
    
    def detect(self, text):
        """Полная детекция"""
        ai_prob = self.predict_ai_probability(text)
        attack_prob = self.predict_attack_probability(text)
        
        if ai_prob > 0.5 and attack_prob > 0.45:
            verdict = " ДИПФЕЙК-АТАКА"
            confidence = (ai_prob + attack_prob) / 2
        elif ai_prob > 0.5:
            verdict = " AI-ТЕКСТ (не атака)"
            confidence = ai_prob
        else:
            verdict = " ТЕКСТ ЧЕЛОВЕКА"
            confidence = 1 - ai_prob
        
        return {
            'verdict': verdict,
            'ai_probability': round(ai_prob, 3),
            'attack_probability': round(attack_prob, 3),
            'confidence': round(confidence, 3),
            'features': self.extract_all_features(text)
        }

# ==============================
# 6. СОЗДАНИЕ РАСШИРЕННОГО ДАТАСЕТА
# ==============================

print("\n" + "="*70)
print("     СОЗДАНИЕ РАСШИРЕННОГО ДАТАСЕТА ДЛЯ ТЕСТИРОВАНИЯ")
print("="*70)

# Человеческие тексты (10 примеров)
human_texts = [
    ("Привет, Иван! Давай встретимся завтра в 15:00, обсудим проект. Напиши, когда удобно. Могу в любое время после обеда.", "Человек"),
    ("Спасибо за письмо, все документы получил. Отправлю отчет завтра утром, как только проверю цифры.", "Человек"),
    ("Купи хлеб и молоко по дороге домой. Не забудь ключи, я оставил их на тумбочке.", "Человек"),
    ("Дорогие коллеги, напоминаю, что завтра в 11:00 состоится планерка. Прошу всех быть вовремя.", "Человек"),
    ("Поздравляю с днем рождения! Желаю здоровья, счастья и успехов во всем.", "Человек"),
    ("Не могли бы вы прислать мне контактные данные нового клиента? Нужно отправить ему договор.", "Человек"),
    ("Сегодня отличная погода, давай сходим в парк. Возьми с собой что-нибудь перекусить.", "Человек"),
    ("Ваш заказ оформлен и будет доставлен 25 марта. Спасибо, что выбрали наш магазин.", "Человек"),
    ("Извините, я опоздаю на встречу на 15 минут. Попал в пробку на МКАД.", "Человек"),
    ("Сдал отчет по проекту, все показатели в норме. Жду обратной связи.", "Человек")
]

# Обычные AI-тексты (10 примеров)
ai_normal_texts = [
    ("Искусственный интеллект развивается стремительными темпами. Нейросети становятся все более совершенными, открывая новые возможности для автоматизации.", "AI обычный"),
    ("Сегодня прекрасная погода. Солнце светит, птицы поют. Отличный день для прогулки в парке или лесу.", "AI обычный"),
    ("Для приготовления классического борща потребуется свекла, капуста, морковь, картофель, лук и томатная паста.", "AI обычный"),
    ("В современном мире цифровые технологии проникают во все сферы жизни. Это создает как новые возможности, так и новые вызовы.", "AI обычный"),
    ("Кофе - один из самых популярных напитков в мире. Его история насчитывает несколько столетий.", "AI обычный"),
    ("Спорт и физическая активность необходимы для поддержания здоровья. Регулярные тренировки укрепляют сердечно-сосудистую систему.", "AI обычный"),
    ("Музыка оказывает глубокое влияние на эмоциональное состояние человека. Она может успокаивать или, наоборот, заряжать энергией.", "AI обычный"),
    ("Путешествия расширяют кругозор и позволяют узнать новые культуры. Каждая поездка оставляет незабываемые впечатления.", "AI обычный"),
    ("Книги - источник знаний и вдохновения. Чтение развивает воображение и обогащает словарный запас.", "AI обычный"),
    ("Экологические проблемы требуют внимания всего человечества. Сохранение природы - наша общая ответственность.", "AI обычный")
]

# Дипфейк-атаки (15 примеров)
attack_texts = [
    ("СРОЧНО! Мне нужно перевести 50000 рублей на счет 884-332-111 прямо сейчас. Директор ждет подтверждения операции.", "Атака BEC"),
    ("Ваш аккаунт скомпрометирован! Немедленно подтвердите пароль по ссылке https://fake-bank.ru чтобы избежать блокировки.", "Атака Фишинг"),
    ("Здравствуйте, это генеральный директор. Мне нужно срочно перевести 150000 рублей на указанные реквизиты. Я на совещании.", "Атака BEC"),
    ("Уважаемый клиент! Ваша учетная запись будет заблокирована через 24 часа. Срочно подтвердите данные карты.", "Атака Фишинг"),
    ("Я из службы безопасности. Обнаружена подозрительная активность. Отправьте мне логин и пароль для проверки.", "Атака Соц. инженерия"),
    ("Срочный запрос от руководства. Необходимо оплатить счет INV-2024-001 на сумму 25000 рублей до 18:00 сегодня.", "Атака BEC"),
    ("Ваш пароль истек. Перейдите по ссылке https://fake-login.ru для смены пароля. Промедление может привести к блокировке.", "Атака Фишинг"),
    ("Это Иван из IT-отдела. Мы фиксируем попытки взлома вашего аккаунта. Сообщите мне код из СМС для защиты.", "Атака Соц. инженерия"),
    ("Генеральный директор просит вас срочно перевести 500000 рублей на счет поставщика. Договор уже подписан.", "Атака BEC"),
    ("Ваш аккаунт будет удален через 24 часа! Подтвердите свои данные, чтобы сохранить доступ.", "Атака Фишинг"),
    ("Здравствуйте, я новый финансовый директор. Нужно срочно оплатить инвойс №8842. Сумма 75000 рублей.", "Атака BEC"),
    ("Обнаружена утечка данных. Для проверки безопасности отправьте свой пароль на этот номер.", "Атака Соц. инженерия"),
    ("СРОЧНО! Клиент перевел деньги, но нужна верификация. Подтвердите перевод по ссылке.", "Атака Фишинг"),
    ("Руководство требует немедленного перевода средств. Все документы уже готовы. Жду подтверждения.", "Атака BEC"),
    ("Ваша карта заблокирована! Разблокируйте по ссылке, иначе счет будет закрыт.", "Атака Фишинг")
]

# Объединяем все тексты
all_texts = human_texts + ai_normal_texts + attack_texts

print(f"Всего текстов: {len(all_texts)}")
print(f"  - Человеческие: {len(human_texts)}")
print(f"  - AI обычные: {len(ai_normal_texts)}")
print(f"  - Дипфейк-атаки: {len(attack_texts)}")

# ==============================
# 7. ЗАПУСК ДЕТЕКЦИИ
# ==============================

print("\n" + "="*70)
print("     ЗАПУСК ДЕТЕКЦИИ (может занять 1-2 минуты)")
print("="*70)

detector = RealisticDeepfakeDetector()

results = []
detailed_results = []

for text, text_type in tqdm(all_texts, desc="Обработка текстов"):
    result = detector.detect(text)
    
    results.append({
        'type': text_type,
        'text': text[:60] + "..." if len(text) > 60 else text,
        'ai_prob': result['ai_probability'],
        'attack_prob': result['attack_probability'],
        'verdict': result['verdict']
    })
    
    detailed_results.append({
        **result,
        'type': text_type,
        'full_text': text
    })

df_results = pd.DataFrame(results)

# ==============================
# 8. ВИЗУАЛИЗАЦИЯ 1: Таблица результатов
# ==============================

print("\n" + "="*70)
print("     РЕЗУЛЬТАТЫ ДЕТЕКЦИИ")
print("="*70)

print("\n{:<20} | {:<55} | {:<12} | {:<12} | {}".format(
    "Тип текста", "Текст", "AI-вероятн.", "Attack-вер.", "Вердикт"))
print("-"*120)

for _, row in df_results.iterrows():
    print("{:<20} | {:<55} | {:<12} | {:<12} | {}".format(
        row['type'],
        row['text'],
        row['ai_prob'],
        row['attack_prob'],
        row['verdict']))

# ==============================
# 9. ВИЗУАЛИЗАЦИЯ 2: Статистика по типам
# ==============================

print("\n" + "="*70)
print("     СТАТИСТИКА ПО ТИПАМ ТЕКСТОВ")
print("="*70)

stats = []
for text_type in df_results['type'].unique():
    subset = df_results[df_results['type'] == text_type]
    
    if 'Атака' in text_type:
        detected = sum(1 for _, row in subset.iterrows() if 'ДИПФЕЙК-АТАКА' in row['verdict'])
    elif 'AI' in text_type:
        detected = sum(1 for _, row in subset.iterrows() if 'AI-ТЕКСТ' in row['verdict'] and 'ДИПФЕЙК' not in row['verdict'])
    else:
        detected = sum(1 for _, row in subset.iterrows() if 'ТЕКСТ ЧЕЛОВЕКА' in row['verdict'])
    
    stats.append({
        'Тип текста': text_type,
        'Количество': len(subset),
        'Правильно определено': detected,
        'Точность (%)': f"{detected/len(subset)*100:.1f}%"
    })

stats_df = pd.DataFrame(stats)
print(stats_df.to_string(index=False))

# ==============================
# 10. ВИЗУАЛИЗАЦИЯ 3: График распределения вероятностей
# ==============================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# AI вероятность
colors = {'Человек': 'green', 'AI обычный': 'blue', 'Атака BEC': 'red', 'Атака Фишинг': 'orange', 'Атака Соц. инженерия': 'purple'}

for text_type in df_results['type'].unique():
    subset = df_results[df_results['type'] == text_type]
    color = colors.get(text_type.split()[0], 'gray')
    axes[0].scatter(subset.index, subset['ai_prob'], label=text_type, color=color, s=50, alpha=0.7)

axes[0].axhline(y=0.5, color='red', linestyle='--', label='Порог AI')
axes[0].set_xlabel('Номер текста')
axes[0].set_ylabel('Вероятность AI-генерации')
axes[0].set_title('Распределение AI-вероятностей')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Attack вероятность
for text_type in df_results['type'].unique():
    subset = df_results[df_results['type'] == text_type]
    color = colors.get(text_type.split()[0], 'gray')
    axes[1].scatter(subset.index, subset['attack_prob'], label=text_type, color=color, s=50, alpha=0.7)

axes[1].axhline(y=0.45, color='red', linestyle='--', label='Порог атаки')
axes[1].set_xlabel('Номер текста')
axes[1].set_ylabel('Вероятность атаки')
axes[1].set_title('Распределение вероятности атаки')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Boxplot
data_to_plot = []
labels = []
for text_type in ['Человек', 'AI обычный', 'Атака BEC', 'Атака Фишинг', 'Атака Соц. инженерия']:
    subset = df_results[df_results['type'].str.contains(text_type)]
    if len(subset) > 0:
        data_to_plot.append(subset['attack_prob'].values)
        labels.append(text_type)

bp = axes[2].boxplot(data_to_plot, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['green', 'blue', 'red', 'orange', 'purple']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[2].axhline(y=0.45, color='red', linestyle='--', label='Порог атаки')
axes[2].set_ylabel('Вероятность атаки')
axes[2].set_title('Распределение вероятности атаки по типам')
axes[2].set_xticklabels(labels, rotation=45)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('attack_probability_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ==============================
# 11. ВИЗУАЛИЗАЦИЯ 4: Признаки атак
# ==============================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

attack_features_names = ['malicious_intent', 'social_engineering', 'action_request', 'entropy', 'punctuation_diversity']

for idx, (text_type, color) in enumerate([('Человек', 'green'), ('AI обычный', 'blue'), ('Атака', 'red')]):
    subset = detailed_results
    if text_type != 'Атака':
        filtered = [r for r in subset if text_type in r['type']]
    else:
        filtered = [r for r in subset if 'Атака' in r['type']]
    
    if filtered:
        feature_means = []
        for f_name in attack_features_names:
            f_mean = np.mean([r['features']['attack_features'][i] for r in filtered for i, n in enumerate(attack_features_names) if n == f_name])
            feature_means.append(f_mean)
        
        axes[0, idx].bar(attack_features_names, feature_means, color=color, alpha=0.7)
        axes[0, idx].set_title(f'{text_type} тексты')
        axes[0, idx].set_ylim(0, 1)
        axes[0, idx].set_xticklabels(attack_features_names, rotation=45)
        axes[0, idx].grid(True, alpha=0.3)

# Тепловая карта корреляции признаков
features_matrix = []
for r in detailed_results:
    features_matrix.append(r['features']['attack_features'])

features_df = pd.DataFrame(features_matrix, columns=attack_features_names)
corr = features_df.corr()

sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Корреляция признаков атак')

# Радарная диаграмма для сравнения типов
from math import pi

categories = attack_features_names
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

for text_type, color in [('Человек', 'green'), ('AI обычный', 'blue'), ('Атака', 'red')]:
    if text_type != 'Атака':
        filtered = [r for r in detailed_results if text_type in r['type']]
    else:
        filtered = [r for r in detailed_results if 'Атака' in r['type']]
    
    if filtered:
        values = []
        for f_name in categories:
            f_mean = np.mean([r['features']['attack_features'][i] for r in filtered for i, n in enumerate(attack_features_names) if n == f_name])
            values.append(f_mean)
        values += values[:1]
        
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=text_type, color=color)
        ax_radar.fill(angles, values, alpha=0.1, color=color)

ax_radar.set_xticks(angles[:-1], categories)
ax_radar.set_ylim(0, 1)
ax_radar.set_title('Профили признаков атак по типам текстов')
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('attack_features_radar.png', dpi=150, bbox_inches='tight')
plt.show()

# ==============================
# 12. ВИЗУАЛИЗАЦИЯ 5: Матрица ошибок
# ==============================

true_labels = []
pred_labels = []

for r in detailed_results:
    if 'Человек' in r['type']:
        true = 'Человек'
    elif 'AI' in r['type']:
        true = 'AI обычный'
    else:
        true = 'Дипфейк-атака'
    
    if 'ЧЕЛОВЕКА' in r['verdict']:
        pred = 'Человек'
    elif 'ДИПФЕЙК' in r['verdict']:
        pred = 'Дипфейк-атака'
    else:
        pred = 'AI обычный'
    
    true_labels.append(true)
    pred_labels.append(pred)

labels_order = ['Человек', 'AI обычный', 'Дипфейк-атака']
cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order, ax=ax)
ax.set_xlabel('Предсказано')
ax.set_ylabel('Фактически')
ax.set_title('Матрица ошибок детектора дипфейк-атак')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ==============================
# 13. ИТОГОВЫЙ ОТЧЕТ
# ==============================

print("\n" + "="*70)
print("     ИТОГОВЫЙ ОТЧЕТ ПО ТЕСТИРОВАНИЮ")
print("="*70)

# Расчет метрик
tp = cm[2, 2]  # Дипфейк-атака -> Дипфейк-атака
fn = cm[2, 0] + cm[2, 1]  # Дипфейк-атака -> Человек или AI
fp = cm[0, 2] + cm[1, 2]  # Человек/AI -> Дипфейк-атака
tn = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]  # Все не-атаки

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"\nМетрики детекции дипфейк-атак:")
print(f"  • Attack Detection Rate (Recall): {recall:.3f} ({recall*100:.1f}%)")
print(f"  • Precision (атаки): {precision:.3f} ({precision*100:.1f}%)")
print(f"  • F1-Score: {f1:.3f}")
print(f"  • Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"  • False Alarm Rate: {fp/(fp+tn):.3f} ({fp/(fp+tn)*100:.1f}%)")

print("\n" + "="*70)
print("Сохраненные файлы:")
print("  • attack_probability_distribution.png - распределение вероятностей")
print("  • attack_features_radar.png - радарная диаграмма признаков")
print("  • confusion_matrix.png - матрица ошибок")
print("="*70)