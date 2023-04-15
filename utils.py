import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AdamW

import numpy as np
import random
from model import Binary_classify
import MyDataLoader
import Loglinear
import os
from re import findall, sub
import emoji
from tqdm import tqdm


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


'''
train(Model:str, Train_dir:str, Test_dir:str, criterion, lr:float=5e-4, epochs:int=30, batch_size:int=128, save_dir:str='./model/', saved:str='auto', gpu:str='None')

Parameters:
     Model(str): Used to select what model to use with their datasets.
          examples: Model='Bert+fc'  or   Model='Loglinear'
          available: 'Bert+fc', 'Loglinear', 'Bert-large+fc'
     Train_dir(str): The path saving your Training data.
          examples: Train_dir = './data/edos_demo.csv'
     Test_dir(str): The path saving your Testing data.
          examples: Test_dir = './data/edos_demo.csv'
     criterion(loss function): Just the function to scoring your answer.
          examples: criterion = torch.nn.CrossEntropyLoss()
     lr(float): Learning rate.
          examples: lr = 1e-3
     batch_size(int): Set the batch_size of DataLoader.
          examples: batch_size = 128
     save_dir(str): The path to save trained model.
          examples: save_dir = './model/'
     saved(str): To determine how many epochs to save your model once.
          examples: saved = '1'
          available: 'auto', '[int]'
     gpu(str): To determine what gpu to use for training.
          examples: gpu = '1' or gpu = 'auto'
          available: 'None'(no use), 'auto'(automatically choose one to use), '[int]'
     tp(str): To determine what you want to classify labels or category.
          examples: tp = 'label'
          available: 'label', 'category'
     category(str): If you choose tp = 'label', then this determine which category to classify.
          examples: category = 'threats'
          available: 'threats', 'derogation', 'animosity', 'prejudiced discussions'

Return:
     Model(torch.nn.Module): The Trained model.
     History(dict): The training information.
          examples: History['error']
          available: 'error', 'accuracy', 'max_error', 'max_error_id'
     
'''
def train(Model:str, Train_dir:str, Test_dir:str, criterion, lr:float=5e-4, epochs:int=30, batch_size:int=128, save_dir:str='./model/', saved:str='auto', gpu:str='None', tp:str='label', category:str = 'threats'):

     #========= Dataset ==============
     if Model == 'Bert+fc':
          if tp == 'label':
               Train_set = MyDataLoader.SexDataset(Train_dir, vector=True)
               Test_set = MyDataLoader.SexDataset(Test_dir, vector=False)
          elif tp == 'category':
               Train_set = MyDataLoader.SexDataset(Train_dir, vector=True, category=category)
               Test_set = MyDataLoader.SexDataset(Test_dir, vector=False, category=category)
          
     elif Model == 'Loglinear':
          Worddict, length = Loglinear.Learn(load_dir=Train_dir)
          if tp == 'label':
               Train_set = Loglinear.LoglinearSet(Train_dir, Worddict)
               Test_set = Loglinear.LoglinearSet(Test_dir, Worddict)
          elif tp == 'category':
               Train_set = Loglinear.LoglinearSet(Train_dir, Worddict, category=category)
               Test_set = Loglinear.LoglinearSet(Test_dir, Worddict, category=category)
     
     elif Model == 'Bert-large+fc':
          if tp == 'label':
               Train_set = MyDataLoader.SexDataset(Train_dir, vector=True, Model='bert-large-uncased')
               Test_set = MyDataLoader.SexDataset(Test_dir, vector=False, Model='bert-large-uncased')
          elif tp == 'category':
               Train_set = MyDataLoader.SexDataset(Train_dir, vector=True, Model='bert-large-uncased', category=category)
               Test_set = MyDataLoader.SexDataset(Test_dir, vector=False, Model='bert-large-uncased', category=category)
     #================================
     Train_loader = DataLoader(dataset=Train_set, batch_size=batch_size, shuffle=True)
     Test_loader = DataLoader(dataset=Test_set, batch_size=batch_size)

     #======== Model ==================
     if Model == 'Bert+fc':
          model = Binary_classify()
     elif Model == 'Loglinear':
          model = Loglinear.LogLinearModel(Worddict=Worddict)
     elif Model == 'Bert-large+fc':
          model = Binary_classify(1024, 2, 'bert-large-uncased')
     #=================================


     #======== Optimizer =============
     if Model == 'Bert+fc':
          optimizer = AdamW(model.parameters(), lr=lr)
     elif Model == 'Loglinear':
          optimizer = optim.Adam(model.parameters(), lr = lr)
     elif Model == 'Bert-large+fc':
          optimizer = AdamW(model.parameters(), lr=lr)
     optimizer.zero_grad()
     #================================


     #======== GPU =====================
     if gpu.isnumeric():
          #gpu = int(gpu)
          os.environ['CUDA_VISIBLE_DEVICES'] = gpu
          if torch.cuda.is_available():
               device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
               model = model.to(device)
     elif gpu == 'None':
          pass
     elif gpu == 'auto':
          if torch.cuda.is_available():
               device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
               model = model.to(device)
          else:
               print('NO GPU TO USE.')
     #==================================

     if Model == 'Bert+fc' or Model == 'Bert-large+fc':
          last = 0
          History = {'error':[], 'accuracy':[], 'max_error':[], 'error_id':[]}
          for epoch in range(epochs):
               loop = tqdm(enumerate(Train_loader), total=len(Train_loader))
               for i, (input_ids, attention_mask, token_type_ids, sexist_label, sexist_category, sexist_vector, id) in loop:
                    out = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
                    if tp == 'label':
                         loss = criterion(out, sexist_label)
                    elif tp == 'category':
                         loss = criterion(out, sexist_category)
                    loss.backward()
                    

                    optimizer.step()
                    optimizer.zero_grad()

                    loop.set_description(f'Epoch [{epoch}/{epochs}]')
                    loop.set_postfix(loss = loss.item())

                    if i % 5 == 0:
                         out = out.argmax(axis=1)
                         if tp == 'label':
                              acc = (out == sexist_category).sum().item()/len(sexist_category)
                         elif tp == 'category':
                              acc = (out == sexist_category).sum().item()/len(sexist_category)
                         print('Acc:'+ str(acc))
                         # 调整学习率，感觉有可能过拟合了
                         if acc < last:
                              lr = lr * 0.1
                              last = acc


               with torch.no_grad():
                    correct_num = 0
                    Errors = 0

                    max_error = 0
                    error_id = 'None'
                    for i , (input_ids, attention_mask, token_type_ids, sexist_label, sexist_category, sexist_vector, id) in enumerate(Test_loader):
                         out = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
                         if tp == 'label':
                              loss = criterion(out, sexist_label)
                         elif tp == 'category':
                              loss = criterion(out, sexist_category)
                         if loss.item() > max_error:
                              max_error = loss.item()
                              error_id = i
                         Errors += loss.item()
                         out = out.argmax(axis=1)
                         if tp == 'label':
                              correct_num += (out == sexist_label).sum().item()
                         elif tp == 'category':
                              correct_num += (out == sexist_category).sum().item()
                    
                    acc = float(correct_num) / float(len(Test_set))
                    History['error'].append(Errors)
                    History['accuracy'].append(acc)
                    History['error_id'].append(error_id)
                    History['max_error'].append(max_error)
                    print("Epochs {},  Test ACC: {}".format(epoch, acc))
     elif Model == 'Loglinear':
          last = 0
          History = {'error':[], 'accuracy':[], 'max_error':[], 'error_id':[]}
          for epoch in range(epochs):
               loop = tqdm(enumerate(Train_loader), total=len(Train_loader))
               for i, (input, sexist_label, sexist_category) in loop:
                    out = model(input)
                    if tp == 'label':
                         loss = criterion(out, sexist_label)
                    elif tp == 'category':
                         loss = criterion(out, sexist_category)
                    loss.backward()
                    

                    optimizer.step()
                    optimizer.zero_grad()

                    loop.set_description(f'Epoch [{epoch}/{epochs}]')
                    loop.set_postfix(loss = loss.item())

                    if i % 5 == 0:
                         out = out.argmax(axis=1)
                         if tp == 'label':
                              acc = (out == sexist_label).sum().item()/len(sexist_label)
                         elif tp == 'category':
                              acc = (out == sexist_category).sum().item()/len(sexist_category)
                         print('Acc:'+ str(acc))
                         # 调整学习率，感觉有可能过拟合了
                         if acc < last:
                              lr = lr * 0.1
                              last = acc


               with torch.no_grad():
                    correct_num = 0
                    Errors = 0

                    max_error = 0
                    error_id = 'None'
                    for i , (input, sexist_label, sexist_category) in enumerate(Test_loader):
                         out = model(input)
                         if tp == 'label':
                              loss = criterion(out, sexist_label)
                         elif tp == 'category':
                              loss = criterion(out, sexist_category)
                         if loss.item() > max_error:
                              max_error = loss.item()
                              error_id = i
                         Errors += loss.item()
                         out = out.argmax(axis=1)
                         if tp == 'label':
                              correct_num += (out == sexist_label).sum().item()
                         elif tp == 'category':
                              correct_num += (out == sexist_category).sum().item()
                    acc = float(correct_num) / float(len(Test_set))
                    History['error'].append(Errors)
                    History['accuracy'].append(acc)
                    History['error_id'].append(error_id)
                    History['max_error'].append(max_error)
                    print("Epochs {},  Test ACC: {}".format(epoch, acc))
     # 保存
     if tp == 'label':
          try:
               torch.save(model, save_dir+'final.pth')
          except:
               os.mkdir(save_dir)
               torch.save(model, save_dir+'final.pth')
     elif tp == 'category':
          try:
               torch.save(model, save_dir+('final_{}.pth'.format(category)))
          except:
               os.mkdir(save_dir)
               torch.save(model, save_dir+('final_{}.pth'.format(category)))
     return Model, History


def validation(model, validation_dir, loss_function, tp:str='label'):
     if tp == 'label':
          Model = model.name
          #========= Dataset ==============
          if Model == 'Bert+fc':
               Validation_set = MyDataLoader.SexDataset(validation_dir, vector=True)
               
          elif Model == 'Loglinear':
               Validation_set = Loglinear.LoglinearSet(Validation_set, model.wordfreq)
          
          elif Model == 'Bert-large+fc':
               Validation_set = MyDataLoader.SexDataset(validation_dir, Model='bert-large-uncased',vector=True)

          loader = DataLoader(dataset=Validation_set, batch_size=128, shuffle=False)
          with torch.no_grad():
               loop = tqdm(enumerate(loader), total=len(loader))
               if Model == 'Loglinear':
                    correct_num = 0
                    for i, (input, sexist_label, sexist_category) in loop:
                         out = model(input)
                         out = out.argmax(axis=1)
                         correct_num += (out == sexist_label).sum().item()
                    acc = correct_num / len(Validation_set)
                    print(acc)
               else:
                    correct_num = 0
                    for i, (input_ids, attention_mask, token_type_ids, sexist_label, sexist_category, sexist_vector, id) in loop:
                         out = model(input_ids, attention_mask, token_type_ids)
                         out = out.argmax(axis=1)
                         correct_num += (out == sexist_label).sum().item()
                    acc = correct_num / len(Validation_set)
                    print(acc) 
     elif tp == 'category':
          if Model == 'Bert+fc':
               Validation_set = MyDataLoader.SexDataset(validation_dir, vector=True, category='None')
               
          elif Model == 'Loglinear':
               Validation_set = Loglinear.LoglinearSet(Validation_set, model.wordfreq, category='None')
          
          elif Model == 'Bert-large+fc':
               Validation_set = MyDataLoader.SexDataset(validation_dir, Model='bert-large-uncased',vector=True, category='None')
          loader = DataLoader(dataset=Validation_set, batch_size=128, shuffle=False)
          with torch.no_grad():
               loop = tqdm(enumerate(loader), total=len(loader))
               if Model == 'Loglinear':
                    correct_num = 0
                    for i, (input, sexist_label, sexist_category) in loop:
                         out = []
                         for mi in model:
                              out.append(mi(input).numpy())
                         output = np.concatenate(out, axis=1)
                         res = predict_category(output=output)
                         correct_num += (res == sexist_category).sum().item()
                    acc = correct_num / len(Validation_set)
                    print(acc)
               else:
                    correct_num = 0
                    for i, (input_ids, attention_mask, token_type_ids, sexist_label, sexist_category, sexist_vector, id) in loop:
                         out = []
                         for mi in model:
                              out.append(model(input_ids, attention_mask, token_type_ids).numpy())
                         output = np.concatenate(out, axis=1)
                         res = predict_category(output=output)
                         correct_num += (res == sexist_category).sum().item()
                    acc = correct_num / len(Validation_set)
                    print(acc) 

def predict_category(output:np.ndarray):
     ret = []
     keys = ['threats', 'derogation', 'animosity', ' prejudiced discussions']
     for result in output:
          tmp = []
          for i in range(4):
               if result[2*i] < result[2 * i + 1]:
                    tmp.append(result[4 * i + 1])
               else:
                    tmp.append(0)
          tt = 0
          tk = 'None'
          for ans in range(4):
               if tt < tmp[ans]:
                    tt = tmp[ans]
                    tk = keys[ans]
          ret.append(tk)
     return ret

def remove_upprintable_chars(s:str):
    """移除所有不可见字符"""
    return ''.join(x for x in s if x.isprintable())

def preprocess(loading_dir:str, save_dir:str=None):
     '''
     Para:
          loading_dir(str):   the loading path of data
          save_dir(str):     the saving path of preprocessed data(default=loading_dir)


          example:
               preprocess(loading_dir = './data/')
               or 
               preprocess(loading_dir = '~/tmp/datasets/', savd_dir = './data/')
          

     return:
          file_list(list):   a list contains the file names saved in the saving_dir


          examples:
               [saving_dir+filenames]
     '''
     ret = []
     if save_dir == None:
          save_dir = loading_dir
     file_list = os.listdir(loading_dir)
     print(file_list)
     for file in file_list:
          if findall('\.csv', file) == []:
               print('1')
               continue
          if findall('\_p\.csv', file) != []:
               print(findall('\_p\.csv', file))
               continue
          csv_in_path = loading_dir + file
          csv_temp_path = save_dir + file.split('.')[0] + '_p.csv'
          ret.append(csv_temp_path)
          with open(csv_in_path, 'rb') as csv_in:
               with open(csv_temp_path, "w", encoding="utf-8") as csv_temp:
                    for line in csv_in:
                         if not line:
                              break
                         else:
                              line = line.decode("utf-8", "ignore")
                              text = emoji.demojize(line)
                              result = sub('[^a-z^A-Z^0-9^\,^\:^\)^\(^\#^\)^\"^\'^\;^\.^\?^\_]+', ' ', text)
                              result = sub('ŕ', 'r', result)
                              result = remove_upprintable_chars(result)
                              #print(result)
                              csv_temp.write(str(result).rstrip() + '\n')

     return ret


if __name__ == '__main__':
     ret = preprocess('./data/')
     print(ret)