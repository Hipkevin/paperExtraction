
import torch.nn as nn
from transformers import BertModel, BartForConditionalGeneration, AutoTokenizer, BartTokenizer

from .config import Config4cls, Config4gen

class BertClassification(nn.Module):
    def __init__(self, config: Config4cls):
        super(BertClassification, self).__init__()

        self.bert = BertModel.from_pretrained(config.ptm_name, cache_dir=config.ptm_path)

        self.classifier = nn.Linear(768, 3)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        x = self.bert(x, attention_mask=(x == 0)).pooler_output
        x = self.dropout(self.classifier(x))

        return x

class BartGeneration(nn.Module):
    def __init__(self, config: Config4gen):
        super(BartGeneration, self).__init__()

        self.bart = BartForConditionalGeneration.from_pretrained(config.ptm_name, cache_dir=config.ptm_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.ptm_name, cache_dir=config.ptm_path)

    def forward(self, x, y):
        out = self.bart(input_ids=x, attention_mask=(x == 0),
                        decoder_input_ids=y, decoder_attention_mask=(y == 0)).logits

        return out

    def generate(self, x):
        summary_ids = self.bart.generate(x, num_beams=2, min_length=0, max_length=20)

        res = self.tokenizer.batch_decode(summary_ids,
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]

        return res
