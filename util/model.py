import torch.nn as nn
from transformers import BertModel, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

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


class PTMGeneration(nn.Module):
    def __init__(self, config: Config4gen):
        super(PTMGeneration, self).__init__()

        self.PTM = AutoModelForSeq2SeqLM.from_pretrained(config.ptm_name, cache_dir=config.ptm_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.ptm_name, cache_dir=config.ptm_path)

    def forward(self, x, y):
        out = self.PTM(input_ids=x, attention_mask=(x == 0),
                       labels=y, decoder_attention_mask=(y == 0))

        logits = out.logits
        loss = out.loss

        return logits, loss