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
<<<<<<< ours
<<<<<<< ours
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs

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
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
=======
>>>>>>> theirs

## Google Colab: установка без конфликтов

1. Включите GPU: `Runtime -> Change runtime type -> GPU`.
2. Установите зависимости:

```bash
pip install -U pip
pip install -r requirements_gpu.txt
```

3. Перезапустите runtime (обязательно после установки).
4. Запустите пайплайн:

```bash
python run_full_hsse_gpu.py
```

Если видите warning про "You must restart the runtime", это нормально — просто перезапустите runtime и запустите команды снова.
<<<<<<< ours
>>>>>>> theirs
=======


## Если "зависло" в Colab на Step 1 (OOF)

Это обычно не зависание, а очень тяжелые вычисления признаков (перплексия + stability) на большом датасете.

Что уже оптимизировано в коде:
- back-translation по умолчанию отключен (`ENABLE_BACK_TRANSLATION=False`),
- исправлена загрузка MLM-модели для перплексии (`AutoModelForMaskedLM`),
- обновлены AMP-вызовы на современный API `torch.amp.*`,
- добавлен progress-bar по обучению семантической модели внутри fold.

Для ускоренного прогона можно временно отключить перплексию:
- в `config_gpu.py` установить `USE_PERPLEXITY_FEATURE = False`.

После отладки верните `USE_PERPLEXITY_FEATURE = True` для полного HSSE-вектора.
>>>>>>> theirs
