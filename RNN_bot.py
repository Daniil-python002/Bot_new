#!/usr/bin/env python
# coding: utf-8

import numpy as np
import Sun_Parser as sparser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import datetime as dt
import telebot
a = dt.datetime.now()
print(a+dt.timedelta(days =1))

class SunDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file, sep=';', engine='python') 
        self.dataframe_normalized = (self.dataframe-self.dataframe.mean())/self.dataframe.std()
        self.data = self.dataframe_normalized.tail(7)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = {'data': self.data.iloc[idx][1:].values}
        return sample
    
class RNN1(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6,  output_size):
        super(RNN1, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.hidden_size5 = hidden_size5
        self.hidden_size6 = hidden_size6
        self.output_size = output_size
        
#        self.fc0 = nn.Linear(self.input_size, self.hidden_size1)
#        self.i1h = nn.Linear(self.input_size + self.hidden_size1, self.hidden_size1)
#        self.i1o = nn.Linear(self.input_size + self.hidden_size1, self.hidden_size1)        
#        self.i2h = nn.Linear(self.hidden_size1 + self.hidden_size2, self.hidden_size2)
#        self.i2o = nn.Linear(self.hidden_size1 + self.hidden_size2, self.hidden_size2)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, self.hidden_size3)
#        self.fc4 = nn.Linear(self.hidden_size3, self.hidden_size4)
#        self.fc5 = nn.Linear(self.hidden_size4, self.hidden_size5)
#        self.fc6 = nn.Linear(self.hidden_size5, self.hidden_size6)
        self.l1 = nn.Linear(self.hidden_size3, self.output_size)
        self.softmax = nn.Linear(self.output_size, self.output_size)
        
    def forward(self, input1):
      
 #        output = torch.tanh(self.fc0(input1))
       
#        combined1 = torch.cat((input1, hidden1), 1)
#        hidden1 = self.i1h(combined1)
#        output = self.i1o(combined1)
#        combined2 = torch.cat((output, hidden2), 1)
#        hidden2 = self.i2h(combined2)
#        output = self.i2o(combined2)
#        for l in range(random.randint(0,10)):
        output =torch.tanh(self.fc1(input1))
        output =torch.tanh(self.fc2(output))
        output =torch.tanh(self.fc3(output))
        output = self.l1(output)
        output = F.softmax(output, dim = -1)
        return output#, hidden1#, hidden2

# RNN 2      
class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4,   output_size):
        super(RNN2, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.output_size = output_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, self.hidden_size3)
        self.fc4 = nn.Linear(self.hidden_size3, self.hidden_size4)
        self.l1 = nn.Linear(self.hidden_size4, self.output_size)
        self.softmax = nn.Linear(self.output_size, self.output_size)
        
    def forward(self, input1):

        output =torch.tanh(self.fc1(input1))
        output =torch.tanh(self.fc2(output))
        output =torch.tanh(self.fc3(output))
        output =torch.tanh(self.fc4(output))
        output = self.l1(output)
        output = F.softmax(output, dim = -1)
        return output
#RNN5
class RNN5(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN5, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.l1 = nn.Linear(self.input_size, self.output_size)
        self.softmax = nn.Linear(self.output_size, self.output_size)
        
    def forward(self, input1):

        output = self.l1(input1)
        output = F.softmax(output, dim = -1)
        return output
#RNN7
class RNN7(nn.Module):
    def __init__(self, input_size, hidden_size1,  output_size):
        super(RNN7, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.output_size = output_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.l1 = nn.Linear(self.hidden_size1, self.output_size)
        self.softmax = nn.Linear(self.output_size, self.output_size)
        
    def forward(self, input1):
        output =torch.tanh(self.fc1(input1))
        output = self.l1(output)
        output = F.softmax(output, dim = -1)
        return output
#RNN6
class RNN6(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6,  output_size):
        super(RNN6, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.hidden_size5 = hidden_size5
        self.hidden_size6 = hidden_size6
        self.output_size = output_size
        
#        self.fc0 = nn.Linear(self.input_size, self.hidden_size1)
#        self.i1h = nn.Linear(self.input_size + self.hidden_size1, self.hidden_size1)
#        self.i1o = nn.Linear(self.input_size + self.hidden_size1, self.hidden_size1)        
#        self.i2h = nn.Linear(self.hidden_size1 + self.hidden_size2, self.hidden_size2)
#        self.i2o = nn.Linear(self.hidden_size1 + self.hidden_size2, self.hidden_size2)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
#        self.fc3 = nn.Linear(self.hidden_size2, self.hidden_size3)
#        self.fc4 = nn.Linear(self.hidden_size3, self.hidden_size4)
#        self.fc5 = nn.Linear(self.hidden_size4, self.hidden_size5)
#        self.fc6 = nn.Linear(self.hidden_size5, self.hidden_size6)
        self.l1 = nn.Linear(self.hidden_size2, self.output_size)
        self.softmax = nn.Linear(self.output_size, self.output_size)
        
    def forward(self, input1):
      
 #        output = torch.tanh(self.fc0(input1))
       
#        combined1 = torch.cat((input1, hidden1), 1)
#        hidden1 = self.i1h(combined1)
#        output = self.i1o(combined1)
#        combined2 = torch.cat((output, hidden2), 1)
#        hidden2 = self.i2h(combined2)
#        output = self.i2o(combined2)
#        for l in range(random.randint(0,10)):
        output =torch.tanh(self.fc1(input1))
        output =torch.tanh(self.fc2(output))
#        output =torch.tanh(self.fc3(output))
        output = self.l1(output)
        output = F.softmax(output, dim = -1)
        return output#, hidden1#, hidden2

bot =telebot.TeleBot('916588751:AAEShWbuUobID7jwAtqmnQV4-LJKzPcC5Rk')
keyboard = telebot.types.ReplyKeyboardMarkup(True, True, True)
keyboard.row('Дай прогноз')

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Хотите узнать прогноз?')

@bot.message_handler(content_types=['text'])
def send_text(message):
    decisions = ''
    if message.text.lower() == 'прогноз' or message.text.lower() == 'да':
        sparser.DataUpdate()
        geostorm_dataset = SunDataset(csv_file="dataset13-v3.csv")
        data_real = DataLoader(geostorm_dataset, shuffle = False, drop_last=False, batch_size = 7)

        rnn1 = RNN2(13, 940, 180, 940, 180, 5)
        rnn2 = RNN5(13, 5)
        rnn3 = RNN7(13, 397, 5)
        rnn4 = RNN6(13, 327, 374, 0, 0, 0, 0, 5)
        rnn5 = RNN1(13, 79, 123, 11, 0, 0, 0, 5)

        rnn1.load_state_dict(torch.load("model\\best model\\rnn5.2.2"))
        rnn2.load_state_dict(torch.load("model\\best model\\rnn5.5.2"))
        rnn3.load_state_dict(torch.load("model\\best model\\rnn5.7.2"))
        rnn4.load_state_dict(torch.load("model\\best model\\rnn5.6.2"))
        rnn5.load_state_dict(torch.load("model\\best model\\rnn5.1.2"))                                

        accuracy1 = [0.7377238590410168, 0.35359116022099446, 0.0, 0.0, 0.0]
        accuracy2 = [0.3125361062969382, 0.20994475138121546, 0.1, 0.15384615384615385, 0.0] 
        accuracy3 = [0.45291738879260546, 0.5580110497237569, 0.0, 0.0, 0.0] 
        accuracy4 = [0.3212016175621028, 0.7237569060773481, 0.0, 0.0, 0.0]
        accuracy5 = [0.005777007510109763, 0.9613259668508287, 0.0, 0.0, 0.0] 
        results=[]
        for batch_idx, data in enumerate(data_real):
    
            data = torch.FloatTensor(data['data'].float())
        
            net_out1 = rnn1(data)
            pred1 = net_out1.data.max(1)[1]
            
            net_out2 = rnn2(data)
            pred2 = net_out2.data.max(1)[1]

            net_out3 = rnn3(data)
            pred3 = net_out3.data.max(1)[1]
            
            net_out4 = rnn4(data)
            pred4 = net_out4.data.max(1)[1]
                                        
            net_out5 = rnn5(data)
            pred5 = net_out5.data.max(1)[1]
            
            k = 0
            print(net_out4)
            print(net_out5)
            for i in range(len(pred1)):
                predict1 = pred1[i]+1 * accuracy1[pred1[i]]
                predict2 = pred2[i]+1 * accuracy2[pred2[i]]
                predict3 = pred3[i]+1 * accuracy3[pred3[i]]
                predict4 = np.log(pred4[i]+1) * accuracy4[pred4[i]]
                predict5 = np.log(pred5[i]+1) * accuracy5[pred5[i]]
                sum_weights = accuracy1[pred1[i]]*0+accuracy2[pred2[i]]*0+accuracy3[pred3[i]]*0+accuracy4[pred4[i]]+accuracy5[pred5[i]]
                
                all_decision = round((predict1.item()*0+predict2.item()*0+\
                                             predict3.item()*0+predict4.item()+predict5.item())/sum_weights)-1
                print(all_decision)
                if all_decision == 0:
                    all_decision = 'Бури не ожидаются'
                else:
                    all_decision = 'Ожидается буря класса G' + str(all_decision)
                decisions+= str(dt.datetime.now()+dt.timedelta(days=(i+1)))[:10]+" | " + str(all_decision)+"\n"
        bot.send_message(message.chat.id, decisions)
bot.polling(none_stop = True, interval=0)


# In[ ]:




