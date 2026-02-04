import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm
import time
import gc
from config_gpu import config
from utils_gpu import load_data_gpu, cleanup_gpu_memory


class GPUSemanticDataset(Dataset):
    """Датасет для семантической модели с оптимизацией для GPU."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Кэшируем токенизацию для ускорения
        print("Предварительная токенизация для GPU...")
        self.encodings = []

        for i in tqdm(range(len(texts)), desc="Токенизация"):
            encoding = tokenizer(
                texts[i],
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.encodings.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            })

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings[idx]['input_ids'],
            'attention_mask': self.encodings[idx]['attention_mask'],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_semantic_model_gpu():
    """Обучает семантическую модель на GPU."""
    print("=" * 70)
    print("🎯 HSSE - ОБУЧЕНИЕ СЕМАНТИЧЕСКОЙ МОДЕЛИ НА GPU")
    print("=" * 70)

    if not config.check_files_exist():
        return

    print(f"Устройство: {config.DEVICE}")
    print(f"Модель: {config.SEMANTIC_MODEL_NAME}")
    print(f"Batch size: {config.SEMANTIC_BATCH_SIZE}")
    print(f"Max length: {config.MAX_LENGTH}")
    print(f"Mixed Precision: {config.USE_AMP}")

    # Загрузка данных
    print("\n📥 Загрузка данных...")
    train_texts, train_labels = load_data_gpu(config.TRAIN_PATH)
    val_texts, val_labels = load_data_gpu(config.VAL_PATH)

    # Токенизатор и модель
    print(f"\n📦 Загрузка модели {config.SEMANTIC_MODEL_NAME} на GPU...")
    tokenizer = AutoTokenizer.from_pretrained(config.SEMANTIC_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.SEMANTIC_MODEL_NAME,
        num_labels=2
    ).to(config.DEVICE)

    # Датасеты
    print("\n🔧 Создание датасетов...")
    train_dataset = GPUSemanticDataset(train_texts, train_labels, tokenizer, config.MAX_LENGTH)
    val_dataset = GPUSemanticDataset(val_texts, val_labels, tokenizer, config.MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.SEMANTIC_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True  # Ускоряет передачу на GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.SEMANTIC_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Оптимизатор и планировщик
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.SEMANTIC_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # Mixed precision
    if config.USE_AMP:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    print("\n" + "=" * 50)
    print("🚀 НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 50)

    best_f1 = 0
    history = []
    start_time = time.time()

    for epoch in range(config.SEMANTIC_EPOCHS):
        epoch_start = time.time()

        # ========== ОБУЧЕНИЕ ==========
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []

        train_bar = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{config.SEMANTIC_EPOCHS} [Обучение]")
        for batch in train_bar:
            optimizer.zero_grad()

            # Перенос на GPU
            input_ids = batch['input_ids'].to(config.DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(config.DEVICE, non_blocking=True)
            labels = batch['label'].to(config.DEVICE, non_blocking=True)

            if config.USE_AMP and scaler is not None:
                # Mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Обычное обучение
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_true.extend(labels.cpu().tolist())

            # Обновляем прогресс-бар
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'gpu_mem': f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
            })

        # ========== ВАЛИДАЦИЯ ==========
        model.eval()
        val_preds = []
        val_true = []
        val_probs = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Эпоха {epoch + 1}/{config.SEMANTIC_EPOCHS} [Валидация]")
            for batch in val_bar:
                input_ids = batch['input_ids'].to(config.DEVICE, non_blocking=True)
                attention_mask = batch['attention_mask'].to(config.DEVICE, non_blocking=True)
                labels = batch['label'].to(config.DEVICE, non_blocking=True)

                if config.USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                preds = torch.argmax(outputs.logits, dim=1)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]

                val_preds.extend(preds.cpu().tolist())
                val_true.extend(labels.cpu().tolist())
                val_probs.extend(probs.cpu().tolist())

        # ========== МЕТРИКИ ==========
        train_acc = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds)
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds)

        epoch_time = time.time() - epoch_start

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'time': epoch_time
        })

        print(f"\n📊 Эпоха {epoch + 1}:")
        print(f"   Время: {epoch_time / 60:.1f} мин")
        print(f"   Loss: {train_loss / len(train_loader):.4f}")
        print(f"   Train Acc/F1: {train_acc:.4f}/{train_f1:.4f}")
        print(f"   Val Acc/F1: {val_acc:.4f}/{val_f1:.4f}")

        # Сохранение лучшей модели
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(config.SEMANTIC_MODEL_PATH)
            tokenizer.save_pretrained(config.SEMANTIC_MODEL_PATH)
            print(f"   💾 Сохранена лучшая модель! F1: {val_f1:.4f}")

        # Очистка памяти
        cleanup_gpu_memory()

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)
    print(f"Общее время: {total_time / 60:.1f} минут")
    print(f"Лучший F1: {best_f1:.4f}")
    print(f"Модель сохранена в: {config.SEMANTIC_MODEL_PATH}")

    # Сохраняем историю
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_path = config.MODELS_DIR / "semantic_training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"История обучения: {history_path}")

    return model, tokenizer


if __name__ == "__main__":
    train_semantic_model_gpu()