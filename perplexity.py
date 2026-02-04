import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm
from config import Config


class PerplexityCalculator:
    """Калькулятор перплексии."""

    def __init__(self):
        self.device = Config.DEVICE

        # Авторегрессионная модель (GPT-2)
        print("Загрузка GPT-2 модели...")
        self.ar_tokenizer = GPT2Tokenizer.from_pretrained(Config.AR_MODEL_NAME)
        self.ar_model = GPT2LMHeadModel.from_pretrained(Config.AR_MODEL_NAME).to(self.device)
        self.ar_model.eval()

        # Маскированная модель (BERT)
        print("Загрузка BERT модели...")
        self.mlm_tokenizer = BertTokenizer.from_pretrained(Config.MLM_MODEL_NAME)
        self.mlm_model = BertForMaskedLM.from_pretrained(Config.MLM_MODEL_NAME).to(self.device)
        self.mlm_model.eval()

    def calculate_perplexity_ar(self, text, stride=512):
        """Вычисляет перплексию с помощью авторегрессионной модели."""
        encodings = self.ar_tokenizer(text, return_tensors='pt')
        max_length = self.ar_model.config.n_positions
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.ar_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc

            if end_loc == seq_len:
                break

        if len(nlls) == 0:
            return 0

        ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
        return ppl.item()

    def calculate_perplexity_mlm(self, text):
        """Вычисляет перплексию с помощью маскированной модели (приближенно)."""
        # Для BERT используем более простое приближение
        tokens = self.mlm_tokenizer.tokenize(text)

        if len(tokens) == 0:
            return 0

        # Ограничиваем длину
        max_len = 510  # 512 - 2 для [CLS] и [SEP]
        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        # Добавляем специальные токены
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.mlm_tokenizer.convert_tokens_to_ids(tokens)
        input_tensor = torch.tensor([input_ids]).to(self.device)

        with torch.no_grad():
            outputs = self.mlm_model(input_tensor)
            logits = outputs.logits

        # Вычисляем отрицательное логарифмическое правдоподобие
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_tensor[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        ppl = torch.exp(loss)
        return ppl.item()

    def calculate_perplexity_gap(self, text):
        """Вычисляет разницу в перплексии между AR и MLM моделями."""
        ppl_ar = self.calculate_perplexity_ar(text)
        ppl_mlm = self.calculate_perplexity_mlm(text)

        # Защита от деления на ноль и очень больших значений
        if ppl_ar == 0 or ppl_mlm == 0:
            return 0

        # Разница в логарифмической шкале более устойчива
        gap = np.log(ppl_ar) - np.log(ppl_mlm)
        return float(gap)

    def batch_calculate_gap(self, texts):
        """Вычисляет перплексию для батча текстов."""
        gaps = []
        for text in tqdm(texts, desc="Calculating perplexity"):
            try:
                gap = self.calculate_perplexity_gap(str(text))
                gaps.append(gap)
            except Exception as e:
                print(f"Error calculating perplexity: {e}")
                gaps.append(0)

        return np.array(gaps)


# Синглтон для калькулятора
_perplexity_calculator = None


def get_perplexity_calculator():
    """Возвращает экземпляр калькулятора перплексии."""
    global _perplexity_calculator
    if _perplexity_calculator is None:
        _perplexity_calculator = PerplexityCalculator()
    return _perplexity_calculator