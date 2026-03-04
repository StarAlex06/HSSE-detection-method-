# HSSE-detection-method-

Метод детекции AI-текстов на основе 4 признаков:
1. Semantic score
2. Stylometric score
3. Perplexity gap
4. Stability score

## Важно: extraction без leakage

Чтобы train-признаки не были обучены на тех же самых объектах, используйте OOF-скрипт:

```bash
python extract_features_oof_gpu.py
```

Он:
- строит `hsse_train` через out-of-fold,
- строит `hsse_val`/`hsse_test` моделями, обученными на полном `train`,
- сохраняет артефакты в `models/`.
<<<<<<< ours
=======

## Запуск в одну команду

### Рекомендуемый режим (без утечек)

```bash
python run_full_hsse_gpu.py
```

По умолчанию этот запуск использует OOF-пайплайн:
1) `extract_features_oof_gpu.py`
2) `train_meta_gpu.py`
3) `evaluate_gpu.py`

### Классический режим

```bash
python run_full_hsse_gpu.py --classic
```

Этот режим запускает раздельное обучение базовых моделей и обычный `extract_features_gpu.py`.
>>>>>>> theirs
