from util.wvTool import WVHandle, Vocab


if __name__ == '__main__':
    # with open('pe_t2t_standard2.txt', 'r', encoding='utf8') as file:
    #     content = file.read().strip().split('\n')
    #
    # with open('corpus.txt', 'w', encoding='utf8') as file:
    #     for c in content:
    #         file.write(c.split('\t')[0] + '\n')

    vocab = Vocab()
    vocab.add_from_corpus('corpus.txt', max_size=10000, min_freq=1)
    vocab.save_to_pkl('vocab.pkl')

    wv_handle = WVHandle(vocab)
    wv_handle.train_from_corpus('corpus.txt', wv_dim=300, save_path='wv_300.npz', seed=42)