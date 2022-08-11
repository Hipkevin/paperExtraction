import pandas as pd
import os

from tqdm import tqdm

from util.dataTool import getStandard4gen, getStandard4gen_enhanced


def replace_name(s: str) -> str:
    replace_dict = {'【目的/意义】': '【目的】',
                    '【方法/过程】': '【方法】',
                    '【研究目的】': '【目的】',
                    '【研究方法】': '【方法】',
                    '【研究设计/方法】': '【方法】'}

    for k, v in replace_dict.items():
        s = s.replace(k, v)

    return s


def getStandard(file_name):
    path = 'data/all'

    data = pd.DataFrame()
    for file in tqdm(os.listdir(path), desc='Reading excel'):
        data = data.append(pd.read_excel(path + '/' + file))

    result = list()
    for idx, item in tqdm(data.iterrows(), desc='Extracting'):
        if str(item['title'])[0: 2] == '基于' and '综述' not in str(item['title']) and ('[' == str(item['abstract'])[0] or '【' == str(item['abstract'])[0]):
            result.append([item['title'],
                           item['abstract'].replace('[', '【').replace(']', '】').replace(' ', ''),
                           item['keywords']])

    result = pd.DataFrame(result, columns=['title', 'abstract', 'keywords'])
    result = result.sample(frac=1)

    result['abstract'] = result['abstract'].apply(replace_name)

    result.to_excel(file_name, index=False)


def write2txt(content_list, title_list, name):
    with open(name, 'w', encoding='utf8') as file:
        file.write('text_a' + '\t' + 'label' + '\n')

        for c, t in zip(content_list, title_list):
            file.write(str(c).strip() + '\t' + str(t).strip() + '\n')


if __name__ == '__main__':
    # getStandard('standard.xlsx')
    # getStandard('all_corpus.xlsx')

    c, t, keywords = getStandard4gen_enhanced('all_corpus.xlsx')

    res = list()
    for key_list in keywords:
        for k in key_list:
            res.append(k)

    with open('keywords.txt', 'w', encoding='utf8') as file:
        for k in set(res):
            file.write(str(k).strip() + '\n')

    # content, title = getStandard4gen('all_corpus.xlsx')
    # write2txt(content, title, 'all_corpus.txt')

    # cv = 0.15
    # test_index = int(len(content) * cv)
    #
    # # train
    # content_train = content[test_index:]
    # title_train = title[test_index:]
    # write2txt(content_train, title_train, 'p2_t2t_standard2_train.txt')
    #
    # # test
    # content_test = content[0: test_index]
    # title_test = title[0: test_index]
    # write2txt(content_test, title_test, 'p2_t2t_standard2_test.txt')