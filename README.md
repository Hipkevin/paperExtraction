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

        self.num_classes = 3
        self.content_pad_size = 150
        self.title_pad_size = 30

        self.ptm_name = 'uer/bart-base-chinese-cluecorpussmall'
        self.ptm_path = 'models'

        self.epoch_size = 10
        self.batch_size = 64
        self.learning_rate = 2e-5
        self.weight_decay = 1e-4
        self.dropout = 0.5
        self.cv = 0.15

        self.T_0 = 6
        self.T_multi = 2

        self.num_beams = 2
        self.seed = 42


| Model | BLEU-1 | BLEU-2 | BLEU-3 | RougeL |
| :---: | :---: | :---: | :---: | :---: |
| Bart | 0.4814 | 0.0 | 0.0 | 0.0 |
| Bart+CLS | 0.2136 | 0.0 | 0.0 | 0.0 |