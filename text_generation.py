# -*- coding: utf-8 -*-


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import re 
import pandas as pd
import argparse

def separate_sentiment(style):
    """
    separates a file with IMBd review dataset to positive and negative sentiment
    and cleans the text
    """
    
    df = pd.read_csv("./data/sentiment.csv")
    
    
    if style == "positive":
        positive_data= df.loc[df['label'] == "1"]
        positive_data.to_csv("positive.csv")
        with open("./data/clean_positive.txt", "w",encoding='utf-8') as outfile:
            with open("positive.csv", "r", encoding='utf-8') as f:
                for line in f.readlines():
                    result = re.sub(r'[0-9]+,', '', line).replace(",1","")
                    #result = re.sub(r"<br /><br />", "", res)
                    for r in result:
                        outfile.write(r)
                        
    elif style == "negative":
        neg = df.loc[df['label'] == "0"]
        neg.to_csv("negative.csv")

        with open("./data/clean_negative.txt", "w",encoding='utf-8') as outfile:
            with open("negative.csv", "r",encoding='utf-8') as f:
                for line in f.readlines():
                    res = re.sub(r'[0-9]+,', '', line).replace(",0","")
                    result = re.sub(r"<br /><br />", "", res)
                    for r in result:
                        outfile.write(r)
                        
                

def prepare_data(style):
    """
    Depending on the specified style, the correct encoding is passed onto the model

    """
    
    if style == "poem":
        with open("./data/inferno.txt", 'r', encoding='utf-8') as data:
            text = data.read()
        
        characters = tuple(set(text))
    
        idx_to_char = dict(enumerate(characters))
        char_to_idx = {ch: idx for idx, ch in idx_to_char.items()}
    
        encoding = np.array([char_to_idx[char] for char in text])   
        
        return encoding
    elif style == "positive":
        separate_sentiment(style)
        
        with open("./data/clean_positive.txt", 'r', encoding='utf-8') as data:
            text = data.read()
        
        characters = tuple(set(text))
    
        idx_to_char = dict(enumerate(characters))
        char_to_idx = {ch: idx for idx, ch in idx_to_char.items()}
    
        encoding = np.array([char_to_idx[char] for char in text])   
        
        return encoding
    
    elif style == "negative":
        separate_sentiment(style)
        with open("./data/clean_negative.txt", 'r', encoding='utf-8') as data:
            text = data.read()
        
        
        characters = tuple(set(text))
    
        idx_to_char = dict(enumerate(characters))
        char_to_idx = {ch: ii for ii, ch in idx_to_char.items()}
    
        encoding = np.array([char_to_idx[ch] for ch in text])   
        
        return encoding   



def embedding(array, y):
    "creates a one-hot vecotr of the given sequence"

    one_hot_vector = np.zeros((np.multiply(*array.shape), y), dtype=np.float32)
    one_hot_vector[np.arange(one_hot_vector.shape[0]), array.flatten()] = 1.
    one_hot_vector = one_hot_vector.reshape((*array.shape, y))

    return one_hot_vector



def generate_batches(array, batch_size, chunk_size):
    '''
    creates batches for the model from the given array

    '''

    total_batch_size = batch_size * chunk_size
    n_batches = len(array)//total_batch_size

    
    array = array[:n_batches * total_batch_size]
    array = array.reshape((batch_size, -1))

    for n in range(0, array.shape[1], chunk_size):

        
        x = array[:, n:n+chunk_size]
        y = np.zeros_like(x)

        try:
            y[:, :-1], y[:, -1] = x[:, 1:], array[:, n+chunk_size]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], array[:, 0]
        yield x, y
        
class LSTM(nn.Module):

    def __init__(self, characters, n_steps=100, hidden_size=256, layers=3,
                               dropout=0.8, lr=0.0001):
        super().__init__()
        
        self.drop_prob = dropout
        self.layers = layers
        self.hidden_size = hidden_size
        self.lr = lr

        
        self.chars = characters
        self.idx_to_char = dict(enumerate(self.chars))
        self.char_to_idx = {ch: idx for idx, ch in self.idx_to_char.items()}

        
        self.lstm = nn.LSTM(len(self.chars), hidden_size, layers,
                            dropout=dropout, batch_first=True)

        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, len(self.chars))

      
        self.init_weights()


    def forward(self, x, hidden_state):
        ''' forward pass of the model, returning the output and the hidden states
        uses dropout to prevent overfitting '''

     
        x, (hidden, cell) = self.lstm(x, hidden_state)

     
        output = self.dropout(x)
        output = output.view(output.size()[0]*output.size()[1], self.hidden_size)
        output = self.fc(output)

    
        return output, (hidden, cell)


    def predict(self, char, h=None, cuda=False, top_k=5):
        ''' 
        predicts the next character based on the previous character
        '''
        if cuda:
            self.cuda()
        else:
            self.cpu()

        if h is None:
            h = self.init_hidden(1)

        x = np.array([[self.char_to_idx[char]]])
        x = embedding(x, len(self.chars))

        inputs = torch.from_numpy(x)

        if cuda:
            inputs = inputs.cuda()

        h = tuple([each.data for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out, dim=1).data

        if cuda:
            p = p.cpu()


        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

        p = p.numpy().squeeze()

        char = np.random.choice(top_ch, p=p/p.sum())

        return self.idx_to_char[char], h

    def init_weights(self):
        ''' Initializes the weights '''

        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        return (weight.new(self.layers, n_seqs, self.hidden_size).zero_(),
                weight.new(self.layers, n_seqs, self.hidden_size).zero_())       
    
def save_model(model, name):
    
    checkpoint = {'hidden_size': model.hidden_size,
                  'layers': model.layers,
                  'state_dict': model.state_dict(),
                  'tokens': model.chars}

    with open(name, 'wb') as f:
        print("saving the model...")
        torch.save(checkpoint, f) 
        
def load_model(name):  

    with open(name, 'rb') as f:
        print("loading the model...")
        checkpoint = torch.load(f)    
    

    model = LSTM(checkpoint['tokens'], hidden_size=checkpoint['hidden_size'], layers=checkpoint['layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model         
      
    
def train(model, style, epochs=100, batch_size=10, chunk_size=50, lr=0.001, cuda=True, print_every=1000):
    ''' Trains the model, depending on the specified style, different data is chosen and
    different hidden states are initiated.

    '''
    if style == "poem":
        data = prepare_data( "poem")
    elif style == "positive":
        data = prepare_data("positive")
    elif style == "negative":
        data = prepare_data("positive")
        

    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data)*(1-0.1))
    data, val_data = data[:val_idx], data[val_idx:]

    if cuda:
        model.cuda()

    count = 0
    all_characters = len(model.chars)

    for epoch in range(epochs):

        h = model.init_hidden(batch_size)

        for x, y in generate_batches(data, batch_size, chunk_size):

            count += 1

            x = embedding(x, all_characters)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()


            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h = model.forward(inputs, h)
            if cuda:
                loss = criterion(output, targets.view(batch_size*chunk_size).type(torch.cuda.LongTensor))
            else:
                loss = criterion(output, targets.view(batch_size*chunk_size).type(torch.LongTensor))
                
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)

            opt.step()

            if count % print_every == 0:

            
                val_h = model.init_hidden(batch_size)
                val_losses = []

                for x, y in generate_batches(val_data, batch_size, chunk_size):

                    
                    x = embedding(x, all_characters)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)


                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = model.forward(inputs, val_h)
                    if cuda:
                        val_loss = criterion(output, targets.view(batch_size*chunk_size).type(torch.cuda.LongTensor))
                    else:
                        val_loss = criterion(output, targets.view(batch_size*chunk_size).type(torch.LongTensor))

                    val_losses.append(val_loss.item())
                    
                    perplexity = torch.exp(val_loss)
                    loss_perplexity=torch.exp(loss) 

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                      "Loss: {:.4f} ".format(loss.item()),
                      "Perplexity: {} ".format(loss_perplexity),
                      "Validation Loss: {:.4f} ".format(np.mean(val_losses)),
                      "Validation Perplexity: {} ".format(perplexity))

with open("./data/inferno.txt", 'r',encoding='utf-8') as f:
    text = f.read()
         
        
def generate_text(style, size=500, prime='The', top_k=5, cuda=False):
    """predicts the characters based on the previous characters based on the 
    specified style"""
    
    if style == "poem":
        net = load_model("./models/inferno_1000.net")
    elif style == "positive":
        
        net = load_model("./models/positive_100.net")
    elif style == "negative":
        
        net = load_model("./models/negative_60.net")
        
        
    if cuda:
        net.cuda()
    else:
        net.cpu()

    net.eval()
    
    
    chars = [ch for ch in prime]   
    h = net.init_hidden(1)
    
    for ch in prime:
        char, h = net.predict(ch, h, cuda=cuda, top_k=top_k)

    chars.append(char)
    
  
    for ii in range(size):
        
        char, h = net.predict(chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)

    return print(''.join(chars))

if __name__ == "__main__":
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('--style', type=str, default="poem")
    # argparser.add_argument('--cuda', type=bool, default=True)

    # argparser.add_argument('--top_k', type=int, default=5)
    # argparser.add_argument('--prime', type=str, default="the")
    
    # args = argparser.parse_args()
   
    

    chars = tuple(set(text))
    lstm_model = LSTM(chars, hidden_size=512, layers=2)
    
    #options for training: poem, positive, negative 
    #train(lstm_model, style="poem", epochs=60, batch_size=128, chunk_size=100, lr=0.0003, cuda=True, print_every=1000)
    
    #save_model(lstm_model, "test.net")
    #generate_text(args)
    
    #oprtions for style: poem, positive, negative 
    generate_text(style="poem", cuda=True, top_k=5, prime="The")   