import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW  # Импорт из torch.optim вместо transformers
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from config import Config
from utils import load_data


class TextDataset(Dataset):
    """Датасет для текстов."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    """Одна эпоха обучения."""
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []

    progress_bar = tqdm(data_loader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions)

    return avg_loss, acc, f1


def eval_model(model, data_loader, device):
    """Оценка модели."""
    model.eval()
    predictions = []
    actual_labels = []
    probabilities = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())

    acc = accuracy_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions)

    return acc, f1, predictions, probabilities


def train_semantic_model():
    """Основная функция обучения семантической модели."""
    print("Загрузка данных...")
    train_texts, train_labels = load_data(Config.TRAIN_PATH)
    val_texts, val_labels = load_data(Config.VAL_PATH)

    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")

    print("Загрузка токенизатора и модели...")
    tokenizer = AutoTokenizer.from_pretrained(Config.SEMANTIC_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.SEMANTIC_MODEL_NAME,
        num_labels=2
    ).to(Config.DEVICE)

    print("Создание датасетов...")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, Config.MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, Config.MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    # Оптимизатор
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Планировщик скорости обучения
    total_steps = len(train_loader) * Config.SEMANTIC_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    print("\nНачало обучения...")
    best_f1 = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_acc': [],
        'val_f1': []
    }

    for epoch in range(Config.SEMANTIC_EPOCHS):
        print(f"\nЭпоха {epoch + 1}/{Config.SEMANTIC_EPOCHS}")

        # Обучение
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, Config.DEVICE, scheduler
        )

        # Валидация
        val_acc, val_f1, _, _ = eval_model(model, val_loader, Config.DEVICE)

        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Сохранение лучшей модели
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(Config.SEMANTIC_MODEL_PATH)
            tokenizer.save_pretrained(Config.SEMANTIC_MODEL_PATH)
            print(f"Новая лучшая модель сохранена! F1: {val_f1:.4f}")

    print("\nОбучение завершено!")
    print(f"Лучшая модель сохранена в: {Config.SEMANTIC_MODEL_PATH}")

    # Визуализация истории обучения
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(Config.MODELS_DIR / "semantic_training_history.png")
    plt.show()


if __name__ == "__main__":
    train_semantic_model()