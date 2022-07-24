import torch.nn as nn
from transformers import BertModel, AutoTokenizer, AutoModelForSeq2SeqLM

from .config import Config4cls, Config4gen
from .wvTool import S2STokenizer


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

        self.classifier = nn.Linear(768, 3)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            encoder = self.PTM.get_encoder()
            out = encoder(x, attention_mask=(x == 0)).last_hidden_state[:, 0, :]

            return self.dropout(self.classifier(out))

        elif len(args) == 2:
            x, y = args[0], args[1]

            out = self.PTM(input_ids=x, attention_mask=(x == 0),
                           labels=y, decoder_attention_mask=(y == 0))

            logits = self.dropout(out.logits)
            loss = out.loss

            return logits, loss

        else:
            raise Exception("Param Error.")


class Seq2SeqModel(PTMGeneration):
    def __init__(self, config: Config4gen):
        super(Seq2SeqModel, self).__init__(config)

        self.embedding = nn.Embedding.from_pretrained(config.emb)
        self.tokenizer = S2STokenizer(config.vocab)

    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            encoder = self.PTM.get_encoder()
            out = encoder(x, attention_mask=(x == 0),
                          inputs_embeds=self.embedding).last_hidden_state[:, 0, :]

            return self.dropout(self.classifier(out))

        elif len(args) == 2:
            x, y = args[0], args[1]

            out = self.PTM(input_ids=x, attention_mask=(x == 0),
                           labels=y, decoder_attention_mask=(y == 0),
                           inputs_embeds=self.embedding,
                           decoder_inputs_embeds=self.embedding)

            logits = self.dropout(out.logits)
            loss = out.loss

            return logits, loss

        else:
            raise Exception("Param Error.")