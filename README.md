# 知识抽取：Problem-Solving Extraction

## 项目结构
    -data  # 原始数据
    -modles # PTM模型存放路径
    -util
        -dataTool.py  # 数据接口
        -model.py  # 模型定义
        -trainer.py  # 训练和测试接口

    config.py  # 实验参数配置
    run.py
    requirement.txt  # 项目依赖

## 实验结果

### 实验组1：
参数：

        self.num_classes = 3
        self.pad_size = 50

        # uer/bart-base-chinese-cluecorpussmall
        self.ptm_name = 'bert-base-chinese'

        self.epoch_size = 10
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.weight_decay = 1e-4
        self.dropout = 0.5
        self.cv = 0.15

        self.seed = 42


| Model | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: |
| Bert | 0.7259 | 0.8568 | 0.7258 | 0.7419 |
| Bart | 0.6272 | 0.6164 | 0.6272 | 0.6107 |

### 实验组2：
参数：

        self.content_pad_size = 150
        self.title_pad_size = 30

        self.ptm_name = 'uer/bart-base-chinese-cluecorpussmall'
        
        self.emb_size = 768
        self.emb = torch.tensor(np.load(f'wv_{self.emb_size}.npz')['embeddings'].astype('float32'))
        self.vocab = Vocab()
        self.vocab.load_from_pkl('vocab.pkl')
        self.tokenizer = S2STokenizer(self.vocab)

        self.epoch_size = 20
        self.batch_size = 64
        self.learning_rate = 2e-5
        self.weight_decay = 1e-4
        self.dropout = 0.5
        self.cv = 0.15

        self.T_0 = 6
        self.T_multi = 2

        self.num_beams = 6
        self.num_beam_groups = 3
        self.diversity_penalty = 0.5
        self.seed = 42


| Model | BP-BLEU | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Bart(wv)-standard | 2.50 | 15.42 | 2.99 | 1.33 | 0.69 |
| Bart(wv)-nonstandard | 1.96 | 15.65 | 1.91 | 0.99 | 0.51 |
| Bart(wv)+CLS-standard | 2.44 | 15.66 | 2.83 | 1.27 | 0.66 |
| Bart-standard(128-10ep) | 1.27 | 3.81 | 1.71 | 0.89 | 0.48 |
| Bart-nonstandard(128-10ep) | 0.09 | 0.25 | 0.12 | 0.06 | 0.03 |
| Bart+CLS-standard(128-10ep) | 1.42 | 3.85 | 1.96 | 1.02 | 0.53 |

### 实验组3：
参数：

        self.content_pad_size = 150
        self.title_pad_size = 30

        self.ptm_name = 'uer/bart-base-chinese-cluecorpussmall'
        
        self.emb_size = 768
        self.emb = torch.tensor(np.load(f'wv_{self.emb_size}.npz')['embeddings'].astype('float32'))
        self.vocab = Vocab()
        self.vocab.load_from_pkl('vocab.pkl')
        self.tokenizer = S2STokenizer(self.vocab)

        self.epoch_size = 20
        self.batch_size = 64
        self.learning_rate = 2e-5
        self.weight_decay = 1e-4
        self.dropout = 0.5
        self.cv = 0.15

        self.T_0 = 6
        self.T_multi = 2

        self.num_beams = 6
        self.num_beam_groups = 3
        self.diversity_penalty = 0.5
        self.seed = 42

将关键词加入分词词典，重新训练词向量

| Model (lambda=1) | BP-BLEU | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Bart(wv)-standard | 2.10 | 13.24 | 2.46 | 1.10 | 0.57 |
| Bart(wv)+CLS-standard | **2.19** | **14.27** | **2.60** | 1.13 | 0.58 |
| Bart(wv)+CLS(key)-standard | 2.18 | 13.64 | 2.57 | 1.14 | 0.59 |
| Bart(wv)+CLS-standard(extend-4-10-128) | 2.10 | 13.01 | 2.46 | 1.11 | 0.57 |
| Bart(wv)+CLS(key)-standard(extend-4-10-128) | 2.15 | 13.59 | 2.37 | **1.15** | **0.59** |

| Model (lambda=1) | BP-BLEU | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Bart(wv)-standard(190) | 2.10 | 13.24 | 2.46 | 1.10 | 0.57 |
| Bart(wv)+CLS-standard(190) | **2.28** | **15.54** | **2.61** | **1.17** | **0.60** |
| Bart(wv)+CLS(key)-standard(190) | 2.22 | 15.20 | 2.52 | 1.13 | 0.59 |

未将关键词加入分词词典，并训练词向量 <br>
lambda < 1 : loss = (1-lambda) * gen_loss + lambda * cls_loss <br>
lambda >= 1 : loss = gen_loss + lambda * cls_loss

| Model(Bart(wv)+CLS-standard) | BP-BLEU | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| lambda=5 | 2.60 | 16.31 | 3.05 | 1.39 | 0.72 |
| lambda=1 | 2.85 | 16.82 | 3.42 | **1.58** | 0.82 |
| lambda=0.8 | 2.71 | 16.51 | 3.22 | 1.47 | 0.76 |
| lambda=0.6 | 2.70 | 16.44 | 3.20 | 1.46 | 0.77 |
| lambda=0.5 | 2.44 | 15.85 | 2.74 | 1.29 | 0.67 |
| lambda=0.4 | **2.86** | **17.02** | **3.43** | **1.58** | **0.83** |
| lambda=0.2 | 2.75 | 16.83 | 3.29 | 1.49 | 0.78 |