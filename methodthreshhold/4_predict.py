# 4_predict.py
import sys
import joblib
import json
import numpy as np
from utils import extract_style_features, extract_semantic_features, extract_perplexity_features, \
    extract_stability_features


def predict_text(text):
    """
    Функция для предсказания одного текста.
    Возвращает: (метка, вероятности по каждой модели, кто подал сигнал)
    """
    # Загрузка моделей
    model_style = joblib.load('model_style.pkl')
    model_semantic = joblib.load('model_semantic.pkl')
    model_perplexity = joblib.load('model_perplexity.pkl')
    model_stability = joblib.load('model_stability.pkl')

    scaler_style = joblib.load('scaler_style.pkl')
    scaler_semantic = joblib.load('scaler_semantic.pkl')
    scaler_perplexity = joblib.load('scaler_perplexity.pkl')
    scaler_stability = joblib.load('scaler_stability.pkl')

    with open('thresholds.json', 'r') as f:
        thresholds = json.load(f)

    # Извлечение признаков
    X_style = scaler_style.transform([extract_style_features(text)])
    X_semantic = scaler_semantic.transform([extract_semantic_features(text)])
    X_perplexity = scaler_perplexity.transform([extract_perplexity_features(text)])
    X_stability = scaler_stability.transform([extract_stability_features(text)])

    # Вероятности
    p_style = model_style.predict_proba(X_style)[0, 1]
    p_semantic = model_semantic.predict_proba(X_semantic)[0, 1]
    p_perplexity = model_perplexity.predict_proba(X_perplexity)[0, 1]
    p_stability = model_stability.predict_proba(X_stability)[0, 1]

    # Кто подал сигнал тревоги
    signals = {
        'style': p_style > thresholds['style'],
        'semantic': p_semantic > thresholds['semantic'],
        'perplexity': p_perplexity > thresholds['perplexity'],
        'stability': p_stability > thresholds['stability']
    }

    # Финальное решение
    is_ai = any(signals.values())

    return {
        'is_ai': is_ai,
        'probabilities': {
            'style': float(p_style),
            'semantic': float(p_semantic),
            'perplexity': float(p_perplexity),
            'stability': float(p_stability)
        },
        'thresholds': thresholds,
        'signals': signals
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Если передан аргумент - файл с текстом
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # Для Windows - ввод построчно, пустая строка = конец
        print("Введите текст для анализа (пустая строка + Enter для завершения):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        text = "\n".join(lines)

    result = predict_text(text)

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТ АНАЛИЗА")
    print("=" * 50)
    print(f"Текст {'🤖 ИИ' if result['is_ai'] else '👤 ЧЕЛОВЕК'}")
    print("\nВероятности (чем выше, тем больше похоже на ИИ):")
    for name, prob in result['probabilities'].items():
        signal = "⚠️" if result['signals'][name] else "✓"
        print(f"  {name.capitalize():12}: {prob:.3f}  {signal}")
    print("\n(⚠️ - признак превысил порог и подал сигнал)")