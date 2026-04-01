# utils.py
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import warnings

warnings.filterwarnings('ignore')

# --- Глобальные переменные для моделей (загружаются один раз) ---
_SEMANTIC_MODEL = None
_PERPLEXITY_MODEL = None
_PERPLEXITY_TOKENIZER = None
_DEVICE = None


def get_device():
    """Определяет и возвращает устройство (GPU если доступно)"""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {_DEVICE}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return _DEVICE


def load_models():
    """Загружает модели один раз при первом вызове"""
    global _SEMANTIC_MODEL, _PERPLEXITY_MODEL, _PERPLEXITY_TOKENIZER

    if _SEMANTIC_MODEL is not None and _PERPLEXITY_MODEL is not None:
        return

    device = get_device()
    print("Загрузка моделей для признаков (один раз)...")

    try:
        print("  Загрузка SentenceTransformer...")
        _SEMANTIC_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        _SEMANTIC_MODEL = _SEMANTIC_MODEL.to(device)

        print("  Загрузка GPT-2...")
        _PERPLEXITY_TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
        _PERPLEXITY_MODEL = GPT2LMHeadModel.from_pretrained('gpt2')
        _PERPLEXITY_MODEL = _PERPLEXITY_MODEL.to(device)
        _PERPLEXITY_MODEL.eval()

        if _PERPLEXITY_TOKENIZER.pad_token is None:
            _PERPLEXITY_TOKENIZER.pad_token = _PERPLEXITY_TOKENIZER.eos_token

        print("Модели загружены успешно!")
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}")
        _SEMANTIC_MODEL = None
        _PERPLEXITY_MODEL = None
        _PERPLEXITY_TOKENIZER = None


# ==================== СТИЛОМЕТРИЯ ====================
def _extract_style_single(text):
    """Внутренняя функция для одного текста (без рекурсии)"""
    if not text or len(text.strip()) == 0:
        return np.array([0, 0, 0, 0, 0, 0])

    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    words = re.findall(r'\w+', text.lower())

    if len(words) == 0:
        return np.array([0, 0, 0, 0, 0, 0])

    # 1. Средняя длина предложения
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

    # 2. Лексическое разнообразие
    lexical_diversity = len(set(words)) / len(words)

    # 3. Доля знаков препинания
    punctuation = re.findall(r'[,.!?;:\"\']', text)
    punct_ratio = len(punctuation) / len(text) if len(text) > 0 else 0

    # 4. Средняя длина слова
    avg_word_len = np.mean([len(w) for w in words])

    # 5. Доля длинных слов (> 8 букв)
    long_words = sum(1 for w in words if len(w) > 8)
    long_word_ratio = long_words / len(words)

    # 6. Разнообразие длины предложений
    sent_lengths = [len(s.split()) for s in sentences] if sentences else [0]
    sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

    return np.array([avg_sent_len, lexical_diversity, punct_ratio,
                     avg_word_len, long_word_ratio, sent_len_std])


def extract_style_features(text):
    """Для одного текста"""
    return _extract_style_single(text)


def extract_style_batch(texts):
    """Для батча текстов"""
    features = []
    for text in texts:
        features.append(_extract_style_single(text))
    return np.array(features)


# ==================== СЕМАНТИКА ====================
def _sentence_split(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences


def _extract_semantic_single(text):
    load_models()

    if _SEMANTIC_MODEL is None:
        return np.zeros(6)

    try:
        sentences = _sentence_split(text)

        if len(sentences) == 0:
            return np.zeros(6)

        embeddings = _SEMANTIC_MODEL.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        if embeddings.is_cuda:
            embeddings = embeddings.cpu()

        embeddings = embeddings.numpy()

        # --- признаки эмбеддингов ---
        emb_mean = float(np.mean(embeddings))
        emb_std = float(np.std(embeddings))
        emb_norm = float(np.linalg.norm(embeddings) / 100)
        emb_max = float(np.max(embeddings))
        emb_min = float(np.min(embeddings))

        # --- семантическая связность предложений ---
        if len(embeddings) > 1:

            sims = []

            for i in range(len(embeddings) - 1):
                v1 = embeddings[i]
                v2 = embeddings[i+1]

                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                sims.append(sim)

            mean_sim = float(np.mean(sims))

        else:
            mean_sim = 0.5

        return np.array([
            emb_mean,
            emb_std,
            emb_norm,
            emb_max,
            emb_min,
            mean_sim
        ])

    except:
        return np.zeros(6)


def extract_semantic_features(text):
    return _extract_semantic_single(text)


def extract_semantic_batch(texts):

    load_models()

    if _SEMANTIC_MODEL is None:
        return np.zeros((len(texts), 6))

    features = []

    for text in texts:

        sentences = _sentence_split(text)

        if len(sentences) == 0:
            features.append(np.zeros(6))
            continue

        embeddings = _SEMANTIC_MODEL.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        emb_mean = float(np.mean(embeddings))
        emb_std = float(np.std(embeddings))
        emb_norm = float(np.linalg.norm(embeddings) / 100)
        emb_max = float(np.max(embeddings))
        emb_min = float(np.min(embeddings))

        if len(embeddings) > 1:
            sims = []

            for i in range(len(embeddings) - 1):
                v1 = embeddings[i]
                v2 = embeddings[i + 1]

                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                sims.append(sim)

            mean_sim = float(np.mean(sims))
        else:
            mean_sim = 0.5

        features.append([
            emb_mean,
            emb_std,
            emb_norm,
            emb_max,
            emb_min,
            mean_sim
        ])

    return np.array(features)

# ==================== ПЕРПЛЕКСИЯ ====================
def _extract_perplexity_single(text):

    load_models()

    if _PERPLEXITY_MODEL is None:
        return np.zeros(4)

    try:

        device = get_device()

        inputs = _PERPLEXITY_TOKENIZER(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        input_ids = inputs['input_ids'].to(device)

        with torch.no_grad():

            outputs = _PERPLEXITY_MODEL(input_ids)

            logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        losses = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )

        losses = losses.reshape(shift_labels.shape)

        token_losses = losses.mean(dim=0)

        perplexity = torch.exp(token_losses.mean()).item()

        log_perplexity = np.log(perplexity)

        probs = torch.softmax(shift_logits, dim=-1)

        entropy = (-probs * torch.log(probs + 1e-9)).sum(-1).mean().item()

        variance = token_losses.var().item()

        return np.array([
            perplexity / 200,
            log_perplexity / 10,
            entropy / 10,
            variance
        ])

    except:
        return np.zeros(4)


def extract_perplexity_features(text):
    return _extract_perplexity_single(text)


def extract_perplexity_batch(texts, batch_size=8):

    features = []

    for t in texts:
        features.append(_extract_perplexity_single(t))

    return np.array(features)

# ==================== СТАБИЛЬНОСТЬ ====================
def _extract_stability_single(text):
    """Внутренняя функция для одного текста"""
    words = re.findall(r'\w+', text)
    if len(words) < 20:
        return np.array([0.5, 0.5, 0.5])

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    # 1. Стабильность длины предложений
    if len(sentences) > 1:
        sent_lengths = [len(s.split()) for s in sentences]
        sent_len_var = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
        sent_stability = min(1.0, sent_len_var / 50.0)
    else:
        sent_stability = 0.5

    # 2. Разнообразие частей речи (через окончания)
    noun_endings = ['а', 'я', 'ь', 'ия', 'ие', 'ость', 'ение', 'ание']
    verb_endings = ['ть', 'ти', 'чь', 'ет', 'ит', 'ат', 'ят', 'ут', 'ют']
    adj_endings = ['ый', 'ий', 'ой', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие']

    endings_count = 0
    for word in words:
        word_lower = word.lower()
        for ending in noun_endings + verb_endings + adj_endings:
            if word_lower.endswith(ending):
                endings_count += 1
                break

    pos_richness = endings_count / len(words) if words else 0

    # 3. Повторяемость слов
    word_freq = {}
    for w in words:
        w_lower = w.lower()
        word_freq[w_lower] = word_freq.get(w_lower, 0) + 1

    avg_freq = np.mean(list(word_freq.values())) if word_freq else 1
    repeatability = min(1.0, avg_freq / 3.0)

    return np.array([sent_stability, pos_richness, repeatability])


def extract_stability_features(text):
    """Для одного текста"""
    return _extract_stability_single(text)


def extract_stability_batch(texts):
    """Для батча текстов"""
    features = []
    for text in texts:
        features.append(_extract_stability_single(text))
    return np.array(features)


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================
def extract_all_features(text):
    """
    Возвращает объединенный вектор признаков для одного текста.
    """
    style = extract_style_features(text)
    semantic = extract_semantic_features(text)
    perplexity = extract_perplexity_features(text)
    stability = extract_stability_features(text)

    return np.concatenate([style, semantic, perplexity, stability])


# ==================== ТЕСТИРОВАНИЕ ====================
if __name__ == "__main__":
    print("Тестирование функций...")
    test_text = "Это тестовый текст. Он нужен для проверки. Работает ли извлечение признаков?"

    print("\n1. Одиночные функции:")
    style = extract_style_features(test_text)
    semantic = extract_semantic_features(test_text)
    perplexity = extract_perplexity_features(test_text)
    stability = extract_stability_features(test_text)

    print(f"Стилометрия ({len(style)}): {style}")
    print(f"Семантика ({len(semantic)}): {semantic}")
    print(f"Перплексия ({len(perplexity)}): {perplexity}")
    print(f"Стабильность ({len(stability)}): {stability}")

    print("\n2. Батч-функции (3 текста):")
    test_texts = [
        "Первый текст. Короткий.",
        "Второй текст для проверки. Он немного длиннее. Три предложения.",
        "Третий текст. Тоже проверяем."
    ]

    style_batch = extract_style_batch(test_texts)
    semantic_batch = extract_semantic_batch(test_texts)
    perplexity_batch = extract_perplexity_batch(test_texts)
    stability_batch = extract_stability_batch(test_texts)

    print(f"Стилометрия батч: {style_batch.shape}")
    print(f"Семантика батч: {semantic_batch.shape}")
    print(f"Перплексия батч: {perplexity_batch.shape}")
    print(f"Стабильность батч: {stability_batch.shape}")