import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MyDataLoader import SexDataset

import numpy as np
import tqdm
import os

from transformers import BertModel, BertConfig

in_channel = 768
out_channel = 2
config = BertConfig(hidden_size=5000, num_hidden_layers=15, num_attention_heads=15)
pretrained = BertModel.from_pretrained('bert-large-uncased')
class Binary_classify(nn.Module):
    def __init__(self, in_channel, out_channel, config:str='bert-base-cased') -> None:
        super().__init__()
        if config == 'bert-large-uncased':
            in_channel = 1024
            self.name = 'Bert-large+fc'
        elif config == 'bert-base-cased':
            in_channel = 768
            self.name = 'Bert+fc'
        else:
            self.pretrained = BertModel(config)
        self.pretrained = BertModel.from_pretrained(config)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=out_channel)


    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(input_ids, attention_mask, token_type_ids)
        
        x1 = self.fc1(out.last_hidden_state[:, 0])
        x1 = F.softmax(x1, dim=1)

        return x1

if __name__ == '__main__':
    for param in pretrained.parameters():
        param.requires_grad_(False)
    dataset = SexDataset('./data/edos_demo.csv')
    #print(len(dataset))
    loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
    #print(len(loader))
    for i, (input_ids, attention_mask, token_type_mask, *kways) in enumerate(loader):
        out = pretrained(input_ids=input_ids, attention_mask = attention_mask, token_type_ids=token_type_mask)
        break
    #model = Binary_classify()
    
            
    '''
    torch.Size([16, 500, 768])
    torch.Size([16, 500])
    torch.Size([16, 500])
    torch.Size([16, 500])
    '''
    print(out.last_hidden_state.shape)
    print(input_ids.shape)
    print(attention_mask.shape)
    print(token_type_mask.shape)

