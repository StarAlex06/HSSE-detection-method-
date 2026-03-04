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
