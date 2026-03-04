from typing import List, Tuple

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config_gpu import config
from perplexity_gpu import get_perplexity_calculator
from transformations_gpu import get_transformations
from utils_gpu import extract_stylometric_features_gpu, load_data_gpu, save_gpu_features


class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class SemanticPredictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.DEVICE
        self.model.eval()

    def predict_proba(self, text: str) -> float:
        encoding = self.tokenizer(
            text,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            if config.USE_AMP and self.device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
        return probs[0, 1].item()


def train_semantic_fold(
    train_texts: List[str],
    train_labels: List[int],
    n_epochs: int = 1
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(config.SEMANTIC_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.SEMANTIC_MODEL_NAME,
        num_labels=2
    ).to(config.DEVICE)

    dataset = SimpleTextDataset(train_texts, train_labels, tokenizer, config.MAX_LENGTH)
    loader = DataLoader(
        dataset,
        batch_size=config.SEMANTIC_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(config.DEVICE.type == 'cuda')
    )

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') if (config.USE_AMP and config.DEVICE.type == 'cuda') else None

    model.train()
    for _ in range(n_epochs):
        for batch in tqdm(loader, desc=f"Semantic train epoch {_ + 1}/{n_epochs}", leave=False):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)

            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

    model.eval()
    return model, tokenizer


def train_stylometric_fold(train_texts: List[str], train_labels: List[int]):
    X_train = []
    for text in train_texts:
        f = extract_stylometric_features_gpu(text)
        X_train.append([f[name] for name in config.STYLOMETRIC_FEATURES])
    X_train = np.array(X_train)
    y_train = np.array(train_labels)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def compute_hsse_vector(
    text: str,
    semantic_predictor: SemanticPredictor,
    sty_model,
    sty_scaler,
    perplexity_calc,
    transformator
) -> np.ndarray:
    if not text or len(text.strip()) < 10:
        return np.zeros(4, dtype=np.float32)

    # 1) semantic
    try:
        p_sem = semantic_predictor.predict_proba(text)
    except Exception:
        p_sem = 0.5

    # 2) stylometric
    try:
        style_features = extract_stylometric_features_gpu(text)
        style_vector = np.array([style_features[feat] for feat in config.STYLOMETRIC_FEATURES]).reshape(1, -1)
        style_vector_scaled = sty_scaler.transform(style_vector)
        p_sty = sty_model.predict_proba(style_vector_scaled)[0, 1]
    except Exception:
        p_sty = 0.5

    # 3) perplexity
    try:
        delta_ppl = perplexity_calc.calculate_perplexity_gap(text)
        delta_ppl_norm = np.clip(delta_ppl, -5, 5) / 10.0
    except Exception:
        delta_ppl_norm = 0.0

    # 4) stability
    try:
        transformed_texts = transformator.apply_transformations(text, config.N_TRANSFORMATIONS)
        all_probs = [p_sem]
        for transformed_text in transformed_texts:
            try:
                all_probs.append(semantic_predictor.predict_proba(transformed_text))
            except Exception:
                continue
        stability = np.var(all_probs) if len(all_probs) > 1 else 0.0
    except Exception:
        stability = 0.0

    return np.array([p_sem, p_sty, delta_ppl_norm, stability], dtype=np.float32)


def extract_hsse_features_oof(n_splits: int = 5, semantic_fold_epochs: int = 1):
    """
    Извлечение HSSE признаков без утечки:
    - train: out-of-fold признаки
    - val/test: признаки от моделей, обученных на полном train
    """
    print("=" * 70)
    print("🔒 HSSE OOF FEATURE EXTRACTION (NO LEAKAGE)")
    print("=" * 70)

    train_texts, train_labels = load_data_gpu(config.TRAIN_PATH)
    val_texts, val_labels = load_data_gpu(config.VAL_PATH)
    test_texts, test_labels = load_data_gpu(config.TEST_PATH)

    train_texts = [str(t) for t in train_texts]
    val_texts = [str(t) for t in val_texts]
    test_texts = [str(t) for t in test_texts]

    y_train = np.array(train_labels)

    perplexity_calc = get_perplexity_calculator()
    transformator = get_transformations()

    X_train_oof = np.zeros((len(train_texts), 4), dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (tr_idx, oof_idx) in enumerate(skf.split(train_texts, y_train), start=1):
        print(f"\n📁 Fold {fold}/{n_splits}")

        fold_train_texts = [train_texts[i] for i in tr_idx]
        fold_train_labels = [train_labels[i] for i in tr_idx]
        fold_oof_texts = [train_texts[i] for i in oof_idx]

        sem_model, sem_tokenizer = train_semantic_fold(
            fold_train_texts,
            fold_train_labels,
            n_epochs=semantic_fold_epochs
        )
        sem_predictor = SemanticPredictor(sem_model, sem_tokenizer)

        sty_model, sty_scaler = train_stylometric_fold(fold_train_texts, fold_train_labels)

        for local_i, text in enumerate(tqdm(fold_oof_texts, desc=f"OOF fold {fold}")):
            global_i = oof_idx[local_i]
            X_train_oof[global_i] = compute_hsse_vector(
                text,
                sem_predictor,
                sty_model,
                sty_scaler,
                perplexity_calc,
                transformator
            )

        del sem_model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'device-side assert triggered' in str(e):
                    print('⚠️ CUDA в ошибочном состоянии (device-side assert). Продолжаю без empty_cache.')
                else:
                    raise

    print("\n✅ OOF train features готовы")

    print("\n📦 Обучение full моделей для val/test...")
    full_sem_model, full_sem_tokenizer = train_semantic_fold(train_texts, train_labels, n_epochs=semantic_fold_epochs)
    full_sem_predictor = SemanticPredictor(full_sem_model, full_sem_tokenizer)
    full_sty_model, full_sty_scaler = train_stylometric_fold(train_texts, train_labels)

    X_val = np.zeros((len(val_texts), 4), dtype=np.float32)
    X_test = np.zeros((len(test_texts), 4), dtype=np.float32)

    for i, text in enumerate(tqdm(val_texts, desc="VAL features")):
        X_val[i] = compute_hsse_vector(
            text,
            full_sem_predictor,
            full_sty_model,
            full_sty_scaler,
            perplexity_calc,
            transformator
        )

    for i, text in enumerate(tqdm(test_texts, desc="TEST features")):
        X_test[i] = compute_hsse_vector(
            text,
            full_sem_predictor,
            full_sty_model,
            full_sty_scaler,
            perplexity_calc,
            transformator
        )

    save_gpu_features({
        'X': X_train_oof,
        'y': np.array(train_labels),
        'texts': train_texts,
        'feature_names': ['semantic_score', 'stylometric_score', 'perplexity_gap', 'stability_score'],
        'mode': 'oof_no_leakage'
    }, 'hsse_train')

    save_gpu_features({
        'X': X_val,
        'y': np.array(val_labels),
        'texts': val_texts,
        'feature_names': ['semantic_score', 'stylometric_score', 'perplexity_gap', 'stability_score'],
        'mode': 'full_train_inference'
    }, 'hsse_val')

    save_gpu_features({
        'X': X_test,
        'y': np.array(test_labels),
        'texts': test_texts,
        'feature_names': ['semantic_score', 'stylometric_score', 'perplexity_gap', 'stability_score'],
        'mode': 'full_train_inference'
    }, 'hsse_test')

    print("\n🎉 Готово: hsse_train/val/test сохранены без leakage на train.")

    print("\n💾 Сохраняю full базовые модели для инференса...")
    full_sem_model.save_pretrained(config.SEMANTIC_MODEL_PATH)
    full_sem_tokenizer.save_pretrained(config.SEMANTIC_MODEL_PATH)
    joblib.dump(full_sty_model, config.STYLOMETRIC_MODEL_PATH, compress=3)
    joblib.dump(full_sty_scaler, config.SCALER_PATH, compress=3)


if __name__ == '__main__':
    if not config.check_files_exist():
        raise SystemExit(1)
    extract_hsse_features_oof(n_splits=5, semantic_fold_epochs=1)
