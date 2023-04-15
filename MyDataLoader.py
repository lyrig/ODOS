from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer

import numpy as np
import sys, os, argparse, glob, pathlib, time
from tqdm import tqdm
from re import findall
import pandas as pd


class SexDataset(Dataset):
    def __init__(self, LoadPath:str, vector:bool=False, gpu:str = 'None', Model:str='bert-base-cased', category:str = 'threats') -> None:
        super().__init__()
        if findall('\.csv',LoadPath.split('/')[-1]) != 0: # 读取该csv文件
            #Index(['rewire_id', 'text', 'label_sexist', 'label_category', 'label_vector'],dtype='object')
            self.data = pd.read_csv(LoadPath, encoding='utf-8')
            self.vector = vector
            self.data = self.data.rename(columns={'Unnamed: 5':'input_ids', 'Unnamed: 6':'attention_mask', 'Unnamed: 7':'special_tokens_mask'})
            self.columns = self.data.columns
            self.shape = self.data.shape
            self.data = self.data.to_dict('dict')
            self.data['token_type_ids'] = {}
            self.cata = category
            token = BertTokenizer.from_pretrained(Model,unk_token="<unk>", use_fast=True)
            '''st = "Fuck that little shit. That downvote of his/her means he/she got hurt by your truth"
            print(len(st))
            tt = token.encode_plus(text=st,truncation=True, padding='max_length', max_length=5000 ,return_token_type_ids=True, return_special_tokens_mask=True, return_tensors='pt')
            print(len(tt['input_ids']))'''
            for i in range(self.shape[0]):
                try:
                    tmp = token.encode_plus(text=self.data['text'][i],truncation=True, padding='max_length', max_length=500 ,return_token_type_ids=True, return_special_tokens_mask=True, return_tensors='pt')
                    #print(i)
                    input_ids = tmp['input_ids'][0]
                    attention_mask = tmp['attention_mask'][0]
                    special_tokens_mask = tmp['special_tokens_mask'][0]
                    token_type_ids = tmp['token_type_ids'][0]
                    if gpu == 'None':
                        pass
                    elif gpu.isnumeric():
                        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
                        if torch.cuda.is_available():
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            input_ids = input_ids.to(device)
                            attention_mask = attention_mask.to(device)
                            token_type_ids = token_type_ids.to(device)
                    

                    self.data['input_ids'][i] = input_ids
                    self.data['attention_mask'][i] = attention_mask
                    self.data['special_tokens_mask'][i] = special_tokens_mask
                    self.data['token_type_ids'][i] = token_type_ids
                except:
                    print(self.data['text'][i])
                    exit(1)
            #print(len(self.data['text']))
            #print(tmp)
        else: #读取该文件夹下所有文件
            Filelist = os.listdir(LoadPath)

    
    def __getitem__(self, index: int):
        '''return input_ids, attention_mask, token_type_ids, sexist, category, vector'''
        if self.cata != 'None':
            if self.vector:
                return self.data['input_ids'][index], self.data['attention_mask'][index], self.data['token_type_ids'][index], 0 if self.data['label_sexist'][index] == 'not sexist' else 1, 1 if self.data['label_category'][index]==self.cata else 0, self.data['label_vector'][index], self.data['rewire_id'][index]
            else:
                return self.data['input_ids'][index], self.data['attention_mask'][index], self.data['token_type_ids'][index], 0 if self.data['label_sexist'][index] == 'not sexist' else 1, 1 if self.data['label_category'][index]==self.cata else 0, 0, self.data['rewire_id'][index]
        if self.cata == 'None':
            if self.vector:
                return self.data['input_ids'][index], self.data['attention_mask'][index], self.data['token_type_ids'][index], 0 if self.data['label_sexist'][index] == 'not sexist' else 1, self.data['label_category'][index], self.data['label_vector'][index], self.data['rewire_id'][index]
            else:
                return self.data['input_ids'][index], self.data['attention_mask'][index], self.data['token_type_ids'][index], 0 if self.data['label_sexist'][index] == 'not sexist' else 1, self.data['label_category'][index], 0, self.data['rewire_id'][index]

    def __len__(self):
        return self.shape[0]-1
    

    def show(self, index=None):
        if index == None:
            print(self.data.head(5))
            for col in self.columns:
                print(col +': ', end='')
                print(self.data[col][index])
        else:
            for col in self.columns:
                print(col +': ', end='')
                print(self.data[col][index])


if __name__ == "__main__":
    se = SexDataset('./data/edos_dev_p.csv')
    print(len(se))
    print(se.data.keys())