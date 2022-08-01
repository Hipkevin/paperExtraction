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