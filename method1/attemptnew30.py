# ==============================
# 1. ИМПОРТЫ
# ==============================
import pandas as pd
import numpy as np
import re
import torch
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from transformers import (
    DistilBertTokenizer, DistilBertModel,
    GPT2LMHeadModel, GPT2Tokenizer,
    BertTokenizer, BertForMaskedLM,
    MarianMTModel, MarianTokenizer
)

import nltk

nltk.download('punkt_tab', quiet=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# 2. ДАННЫЕ
# ==============================
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

# РЕКОМЕНДУЮ ДЛЯ ОТЛАДКИ (можешь убрать потом)
#train = train.sample(500)
#val = val.sample(200)
#test = test.sample(200)


# ==============================
# 3. СТИЛОМЕТРИЯ
# ==============================
def stylometric_features(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)

    num_sent = len(sentences)
    num_words = len(words)

    avg_sent_len = num_words / (num_sent + 1e-6)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    punctuation = len(re.findall(r'[.,!?;:]', text))
    commas = text.count(',')
    dots = text.count('.')
    questions = text.count('?')
    exclamations = text.count('!')

    unique_words = len(set(words))
    lexical_diversity = unique_words / (num_words + 1e-6)

    digits = len(re.findall(r'\d', text))
    uppercase = sum(1 for c in text if c.isupper())

    return [
        num_sent, num_words, avg_sent_len, avg_word_len,
        punctuation, commas, dots, questions, exclamations,
        unique_words, lexical_diversity, digits, uppercase,
        len(text), text.count('-'), text.count('"'),
        text.count("'"), text.count('('), text.count(')')
    ]


# ==============================
# 4. DistilBERT (SEMANTIC)
# ==============================
distil_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distil_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
distil_model.eval()


def get_bert_embedding(text):
    inputs = distil_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = distil_model(**inputs)

    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()


# ==============================
# 5. PERPLEXITY
# ==============================
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.eval()

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_mlm = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
bert_mlm.eval()


def gpt2_perplexity(text):
    encodings = gpt2_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = gpt2_model(**encodings, labels=encodings["input_ids"])

    loss = outputs.loss
    return torch.exp(loss).item()


def bert_perplexity(text):
    tokens = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    input_ids = tokens["input_ids"]

    with torch.no_grad():
        outputs = bert_mlm(**tokens, labels=input_ids)

    loss = outputs.loss
    return torch.exp(loss).item()


def perplexity_feature(text):
    return gpt2_perplexity(text) - bert_perplexity(text)


# ==============================
# 6. УСТОЙЧИВОСТЬ (translation)
# ==============================
translator_name = "Helsinki-NLP/opus-mt-en-de"
translator_tokenizer = MarianTokenizer.from_pretrained(translator_name)
translator_model = MarianMTModel.from_pretrained(translator_name).to(device)
translator_model.eval()

back_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
back_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en").to(device)
back_model.eval()


def back_translate(text):
    inputs = translator_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    translated = translator_model.generate(**inputs, max_length=512)
    de_text = translator_tokenizer.decode(translated[0], skip_special_tokens=True)

    inputs_back = back_tokenizer(
        de_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    back = back_model.generate(**inputs_back, max_length=512)

    return back_tokenizer.decode(back[0], skip_special_tokens=True)


def stability_feature(text, emb1):
    paraphrased = back_translate(text)
    emb2 = get_bert_embedding(paraphrased)
    return np.linalg.norm(emb1 - emb2)


# ==============================
# 7. FEATURE EXTRACTION (ИСПРАВЛЕННАЯ)
# ==============================
def extract_features(df):
    features = []

    print("Extracting features...")

    for idx, text in enumerate(tqdm(df["text"], desc="Processing texts")):
        try:
            # Стилометрия
            style = stylometric_features(text)

            # Семантические признаки (BERT)
            semantic = get_bert_embedding(text)

            # Perplexity
            perp = [perplexity_feature(text)]

            stability = [stability_feature(text, semantic)]

            # Конкатенация
            vector = np.concatenate([style, semantic, perp, stability])
            features.append(vector)

        except Exception as e:
            print(f"Error processing text {idx}: {e}")
            # В случае ошибки добавляем нулевой вектор
            features.append(np.zeros(768 + len(stylometric_features("test")) + 2))

    return np.array(features)


print("Extracting features...")
X_train = extract_features(train)
X_val = extract_features(val)
X_test = extract_features(test)

y_train = train["label"].values
y_val = val["label"].values
y_test = test["label"].values

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# ==============================
# 8. LIGHTGBM
# ==============================
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("\nTraining LightGBM...")
model.fit(X_train, y_train)


# ==============================
# 9. EVALUATION
# ==============================
def evaluate(X, y, name):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    print(f"\n{name}")
    print(f"ROC-AUC: {roc_auc_score(y, probs):.4f}")
    print(f"Accuracy: {accuracy_score(y, preds):.4f}")
    print(f"F1: {f1_score(y, preds):.4f}")
    print(f"Precision: {precision_score(y, preds):.4f}")
    print(f"Recall: {recall_score(y, preds):.4f}")


def plot_roc_curve(X, y, name):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    probs = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({name})")
    plt.legend()
    plt.show()


evaluate(X_train, y_train, "TRAIN")
evaluate(X_val, y_val, "VAL")
evaluate(X_test, y_test, "TEST")
plot_roc_curve(X_test, y_test, "TEST")