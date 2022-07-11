
import torch.nn as nn
from transformers import BertModel

class BertClassification(nn.Module):
    def __init__(self, config):
        super(BertClassification, self).__init__()

        self.bert = BertModel.from_pretrained(config.ptm_name, cache_dir=config.ptm_path)

        self.classifier = nn.Linear(768, 3)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        x = self.bert(x, attention_mask=(x == 0)).pooler_output
        x = self.dropout(self.classifier(x))

        return x