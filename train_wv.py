from util.wvTool import WVHandle, Vocab, S2STokenizer


if __name__ == '__main__':
    # with open('all_corpus.txt', 'r', encoding='utf8') as file:
    #     content = file.read().strip().split('\n')
    #
    # with open('corpus.txt', 'w', encoding='utf8') as file:
    #     for c in content:
    #         file.write(c.split('\t')[0] + '\n')

    vocab = Vocab()
    vocab.add_from_corpus('corpus.txt', min_freq=2, user_dict='keywords.txt')
    vocab.save_to_pkl('vocab.pkl')

    # tokenizer = S2STokenizer(vocab)
    # ids = tokenizer.encode('基于聚类模型的文本分类研究')
    # print(len(vocab))

    wv_handle = WVHandle(vocab)
    wv_handle.train_from_corpus('corpus.txt', wv_dim=768, save_path='wv_768.npz', seed=42)