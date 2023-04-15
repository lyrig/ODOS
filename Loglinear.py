import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emoji

from tqdm import tqdm


#从训练集中学习概率P(word)，和词的数目
def Learn(load_dir:str):
    ret = {}
    context = pd.read_csv(load_dir, encoding='utf-8').to_dict('dict')
    text = context['text']
    length = len(text)
    loop = tqdm(range(length), desc='Analysing Data')
        
    for index in loop:
        wordlist = text[index].split(' ')
        for word in wordlist:
            ret.setdefault(word, 0)
            ret[word] += 1


    all = sum(list(ret.values()))
    for word in list(ret.keys()):
        ret[word] /= float(all)

    ret = dict(sorted(ret.items(), key=lambda x : x[1], reverse=True))
    print(type(ret))
    num = len(ret)
    return ret, num



'''
LogLinear的数据集
'''
class LoglinearSet(Dataset):
    def __init__(self, Load_dir:str, Worddict, dim:str='auto', category:str = 'threats') -> None:
        super().__init__()
        self.data = pd.read_csv(Load_dir, encoding='utf-8').dropna(axis=1).to_dict('dict')
        self.len = len(self.data['text'])
        self.wordfreq = Worddict
        self.wordnum = len(self.wordfreq)
        self.cata = category
        self.name = 'Loglinear'
        if dim == 'auto':
            self.dim = int(self.wordnum * 0.8)
        elif dim.isnumeric():
            self.dim = int(dim)
        else:
            raise('Warning, dim Error')
            exit(1)
    def tokenizer(self, X:str):
        #print(self.wordfreq)
        #print(type(self.wordfreq))
        dictionary = list(dict(self.wordfreq).keys())
        feature = []
        #print(X)
        ret = []
        text = list(X.split(' '))
        for index in range(self.dim):
            if text.count(dictionary[index]):
                feature.append(text.count(dictionary[index]))
            else:
                feature.append(0)
        #print(len(feature))
        return torch.tensor(feature, dtype=torch.float)

    def __len__(self):
        return self.len
    

    def __getitem__(self, index):
        if self.cata != 'None':
            return self.tokenizer(self.data['text'][index]), 0 if self.data['label_sexist'][index]=='not sexist' else 1, 1 if self.data['label_category'] == self.cata else 0
        else:
            return self.tokenizer(self.data['text'][index]), 0 if self.data['label_sexist'][index]=='not sexist' else 1, self.data['label_category']



'''
Loglinear 模型
'''
class LogLinearModel(nn.Module):
    def __init__(self, Worddict:dict, dim:str='auto', out_channel:int=2):
        super().__init__()
        self.wordfreq = Worddict
        self.wordnum = len(self.wordfreq)
        if dim == 'auto':
            self.dim = int(self.wordnum * 0.8)
        elif dim.isnumeric():
            self.dim = int(dim)
        else:
            raise('Warning, dim Error')
            exit(1)
        
        self.fc1 = nn.Linear(in_features=self.dim, out_features=out_channel)

    def tokenizer(self, X:str):
        #print(self.wordfreq)
        #print(type(self.wordfreq))
        dictionary = list(dict(self.wordfreq).keys())
        feature = []
        #print(X)
        ret = []
        for i in X:
            text = list(X.split(' '))
            for index in range(self.dim):
                feature.append(text.count(dictionary[index]))
        
        return torch.tensor(feature)

    def forward(self, X):
        out = self.fc1(X)
        out = F.softmax(out)

        return out

if __name__ == '__main__':
    dset = LoglinearSet('./data/edos_demo.csv')
    print(len(dset.data))
    print(dset.data)