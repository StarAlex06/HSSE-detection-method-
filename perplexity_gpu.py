import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import BertForMaskedLM, BertTokenizerFast
from tqdm import tqdm
from config_gpu import config
from utils_gpu import cleanup_gpu_memory


class GPUPerplexityCalculator:
    """Вычисляет перплексию на GPU для HSSE метода."""

    def __init__(self):
        self.device = config.DEVICE

        print("🤖 Инициализация моделей для перплексии на GPU...")

        # Авторегрессионная модель (GPT-2)
        print(f"   Загрузка {config.AR_MODEL_NAME}...")
        self.ar_tokenizer = GPT2TokenizerFast.from_pretrained(config.AR_MODEL_NAME)
        self.ar_model = GPT2LMHeadModel.from_pretrained(config.AR_MODEL_NAME).to(self.device)
        self.ar_model.eval()

        # Маскированная модель (BERT)
        print(f"   Загрузка {config.MLM_MODEL_NAME}...")
        self.mlm_tokenizer = BertTokenizerFast.from_pretrained(config.MLM_MODEL_NAME)
        self.mlm_model = BertForMaskedLM.from_pretrained(config.MLM_MODEL_NAME).to(self.device)
        self.mlm_model.eval()

        print(f"✅ Модели загружены на {self.device}")

    def calculate_perplexity_ar(self, text: str) -> float:
        """Вычисляет перплексию с помощью авторегрессионной модели."""
        try:
            encodings = self.ar_tokenizer(text, return_tensors='pt')
            input_ids = encodings.input_ids.to(self.device)

            with torch.no_grad():
                if config.USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = self.ar_model(input_ids, labels=input_ids)
                else:
                    outputs = self.ar_model(input_ids, labels=input_ids)

            loss = outputs.loss
            perplexity = torch.exp(loss).cpu().item()

            return perplexity
        except Exception as e:
            print(f"Ошибка AR perplexity: {e}")
            return 0.0

    def calculate_perplexity_mlm(self, text: str) -> float:
        """Вычисляет перплексию с помощью маскированной модели."""
        try:
            # Токенизируем
            tokens = self.mlm_tokenizer.tokenize(text)
            if len(tokens) > 510:
                tokens = tokens[:510]

            # Добавляем специальные токены
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = self.mlm_tokenizer.convert_tokens_to_ids(tokens)
            input_tensor = torch.tensor([input_ids]).to(self.device)

            with torch.no_grad():
                if config.USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = self.mlm_model(input_tensor)
                else:
                    outputs = self.mlm_model(input_tensor)

                logits = outputs.logits

            # Вычисляем perplexity
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_tensor[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

            perplexity = torch.exp(loss).cpu().item()

            return perplexity
        except Exception as e:
            print(f"Ошибка MLM perplexity: {e}")
            return 0.0

    def calculate_perplexity_gap(self, text: str) -> float:
        """
        Вычисляет Perplexity Gap для HSSE метода.

        Δ_ppl(x) = log(PPL_AR(x)) - log(PPL_MLM(x))
        """
        ppl_ar = self.calculate_perplexity_ar(text)
        ppl_mlm = self.calculate_perplexity_mlm(text)

        # Защита от нулей
        if ppl_ar == 0 or ppl_mlm == 0:
            return 0.0

        # Вычисляем разницу в логарифмической шкале
        gap = np.log(ppl_ar) - np.log(ppl_mlm)

        # Нормализуем (эмпирические границы)
        gap = np.clip(gap, -5, 5)

        return float(gap)

    def batch_calculate_perplexity_gap(self, texts: list, batch_size: int = None) -> np.ndarray:
        """
        Вычисляет Perplexity Gap для батча текстов.

        Args:
            texts: список текстов
            batch_size: размер батча (None = автоматический)

        Returns:
            Массив с Perplexity Gap для каждого текста
        """
        if batch_size is None:
            batch_size = config.PERPLEXITY_BATCH_SIZE

        gaps = []

        print(f"📊 Вычисление Perplexity Gap для {len(texts)} текстов...")

        for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity Gap"):
            batch_texts = texts[i:i + batch_size]

            for text in batch_texts:
                try:
                    gap = self.calculate_perplexity_gap(str(text))
                    gaps.append(gap)
                except Exception as e:
                    print(f"Ошибка для текста {i}: {e}")
                    gaps.append(0.0)

            # Очищаем память после каждого батча
            cleanup_gpu_memory()

        return np.array(gaps)


# Синглтон для калькулятора
_perplexity_calculator = None


def get_perplexity_calculator():
    """Возвращает экземпляр калькулятора перплексии."""
    global _perplexity_calculator
    if _perplexity_calculator is None:
        _perplexity_calculator = GPUPerplexityCalculator()
    return _perplexity_calculator