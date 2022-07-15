import pandas as pd
import os

from tqdm import tqdm

def replace_name(s: str) -> str:
    replace_dict = {'【目的/意义】': '【目的】',
                    '【方法/过程】': '【方法】',
                    '【研究目的】': '【目的】',
                    '【研究方法】': '【方法】',
                    '【研究设计/方法】': '【方法】'}

    for k, v in replace_dict.items():
        s = s.replace(k, v)

    return s


if __name__ == '__main__':
    path = 'data/all'

    data = pd.DataFrame()
    for file in tqdm(os.listdir(path), desc='Reading excel'):
        data = data.append(pd.read_excel(path + '/' + file))

    result = list()
    for idx, item in tqdm(data.iterrows(), desc='Extracting'):
        if '[' == str(item['abstract'])[0] or '【' == str(item['abstract'])[0]:
            result.append([item['title'],
                           item['abstract'].replace('[', '【').replace(']', '】').replace(' ', ''),
                           item['keywords']])

    result = pd.DataFrame(result, columns=['title', 'abstract', 'keywords'])

    result['abstract'] = result['abstract'].apply(replace_name)

    result.to_excel('standard2.xlsx', index=False)
