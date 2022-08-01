from util.model import Seq2SeqModel
from util.config import Config4gen

import torch

if __name__ == '__main__':
    config = Config4gen()

    model = Seq2SeqModel(config)
    print(config.vocab.token2id)
    ids = model.tokenizer.encode(text='三大国际顶尖创业研究专业期刊（JBV、ETP和JSBM）为样本,利用科学学研究工具Sci2工具包对所采集的有关创业研究的文献进行引文耦合分析,'
                                      '绘制国际创业研究领域的知识图谱。通过对施引文献进行聚类分析,对该领域不同时间段的代表作者、重要文献及其主要研究主题进行梳理,'
                                      '深入解读国际创业研究文献引文耦合图谱的特点,总结国际创业研究的发展态势。', max_length=15, padding='max_length',
                                 add_special_tokens=True)
    print(ids)

    print(model.tokenizer.decode(ids))