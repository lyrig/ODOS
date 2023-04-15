import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import setup_seed, train, validation

import tqdm
import os, argparse
import model
import matplotlib.pyplot as plt


import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('--train_path', type=str, default='./data/edos_train_p.csv')
parser.add_argument('--test_path', type=str, default='./data/edos_test_p.csv')
parser.add_argument('--validation_path', type=str, default='./data/edos_dev_p.csv')
parser.add_argument('--save_path', type=str, default='./model/')
parser.add_argument('--saved', type=str, default='30')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model', '-m', type=str, default='Bert+fc')
parser.add_argument('--loss_function', '-lf', type=str, default='criterion')
parser.add_argument('--gpu', type=str, default='auto')
parser.add_argument('--type', type=str, default='label')


args = parser.parse_args()


train_dir = args.train_path
test_dir = args.test_path
validation_dir = args.validation_path
lr = args.lr
epochs = args.epochs
save_path = args.save_path
batch_size = args.batch_size
lf = args.loss_function
loss = None
md = args.model
Model = None
sd = str(args.saved)
tp = args.type


setup_seed(args.seed)


#======== Loss Function ============
if lf == 'criterion':
    loss = nn.CrossEntropyLoss()
#===================================

#============ Start Train =============
Model, history = train(Model=md, Train_dir=train_dir, Test_dir=test_dir, criterion=loss, epochs=epochs, batch_size=batch_size, save_dir=save_path, saved=sd, lr=lr, tp = tp)
#======================================

print(history)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot([i for i in range(1, 31)], history['accuracy'])
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')


plt.subplot(2, 1, 2)
plt.plot([i for i in range(1, 31)], history['error'])
plt.title('error')
plt.xlabel('epochs')
plt.ylabel('error')

plt.show()
plt.savefig('./result.png')

#============ Validation ==============
History = validation(model=Model, validation_dir = validation_dir, loss_function = loss)
#======================================


print("End.")


