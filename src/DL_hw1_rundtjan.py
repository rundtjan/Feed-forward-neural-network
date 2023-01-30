"""
   Introduction to Deep Learning (LDA-T3114)
   Skeleton Code for Assignment 1: Sentiment Classification on a Feed-Forward Neural Network

   Hande Celikkanat & Miikka Silfverberg
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

#ATTENTION: If necessary, add the paths to your data_semeval.py and paths.py here:
#import sys
#sys.path.append('</path/to/below/modules>')
from data_semeval import *
from paths import data_dir


#--- hyperparameters ---

N_CLASSES = len(LABEL_INDICES)
N_EPOCHS = 200  
LEARNING_RATE = 0.06
BATCH_SIZE = 100
REPORT_EVERY = 1
IS_VERBOSE = False


def make_bow(tweet, indices):
    feature_ids = list(indices[tok] for tok in tweet['BODY'] if tok in indices)
    bow_vec = torch.zeros(len(indices))
    bow_vec[feature_ids] = 1
    return bow_vec.view(1, -1)

def generate_bow_representations(data):
    vocab = set(token for tweet in data['training'] for token in tweet['BODY'])
    vocab_size = len(vocab) 
    indices = {w:i for i, w in enumerate(vocab)}
  
    for split in ["training","development.input","development.gold",
                  "test.input","test.gold"]:
        for tweet in data[split]:
            tweet['BOW'] = make_bow(tweet,indices)

    return indices, vocab_size

# Convert string label to pytorch format.
def label_to_idx(label):
    return torch.LongTensor([LABEL_INDICES[label]])



#--- model ---

class FFNN(nn.Module):
    # Feel free to add whichever arguments you like here.
    def __init__(self, vocab_size, n_classes, extra_arg_1=None, extra_arg_2=None):
        super(FFNN, self).__init__()
        # WRITE CODE HER
        self.fc1 = nn.Linear(vocab_size, 60)
        self.fc2 = nn.Linear(60, n_classes)
        
    def forward(self, x, softmax):
        # WRITE CODE HERE
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        if softmax:
          x = F.log_softmax(x, dim=1)
        return x


#class FFNN_flat(nn.Module):
    # Feel free to add whichever arguments you like here.
#    def __init__(self, vocab_size, n_classes, extra_arg_1=None, extra_arg_2=None):
#        super(FFNN3, self).__init__()
        # WRITE CODE HERE
#        self.fc1 = nn.Linear(vocab_size, n_classes)
        
#    def forward(self, x, softmax):
        # WRITE CODE HERE
#        x = torch.sigmoid(self.fc1(x))
#        if softmax:
#          x = F.log_softmax(x, dim=1)
#        return x

#--- data loading ---
data = read_semeval_datasets(data_dir)
indices, vocab_size = generate_bow_representations(data)


#--- set up ---

# WRITE CODE HERE
model = FFNN(vocab_size, N_CLASSES) #add extra arguments here if you use
print(model)
loss_function = nn.NLLLoss()
loss_function2 = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

#result = model.forward(data['training'][0]['BOW'], True)
#print("My first result ", result)

#--- training ---
for epoch in range(N_EPOCHS):
    total_loss = 0
    # Generally speaking, it's a good idea to shuffle your
    # datasets once every epoch.
    random.shuffle(data['training'])    

    for i in range(int(len(data['training'])/BATCH_SIZE)):
        minibatch = data['training'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        mb = []
        target = []
        for item in minibatch:
          mb.append(item['BOW'])
          target.append(label_to_idx(item['SENTIMENT']))
        mb = torch.cat(mb, 0)
        target = torch.cat(target, 0)
        optimizer.zero_grad()
        result = model.forward(mb, True)
        loss = loss_function(result, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
                              
    if ((epoch+1) % REPORT_EVERY) == 0:
        print('epoch: %d, loss: %.4f' % (epoch+1, total_loss*BATCH_SIZE/len(data['training'])))



#--- test ---
correct = 0
with torch.no_grad():
    for tweet in data['test.gold']:
        gold_class = label_to_idx(tweet['SENTIMENT'])
        # WRITE CODE HERE
        # You can, but for the sake of this homework do not have to,
        # use batching for the test data.
        result = model(tweet['BOW'], True)
        predicted = torch.argmax(result).item()
        if gold_class == predicted:
          correct = correct+1
        if IS_VERBOSE:
            print('TEST DATA: %s, GOLD LABEL: %s, GOLD CLASS %d, OUTPUT: %d' % 
                 (' '.join(tweet['BODY'][:-1]), tweet['SENTIMENT'], gold_class, predicted))

    print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))

