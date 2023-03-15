from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import torch
import math
import time
import sys
import json
import numpy as np
import torch.nn as nn


class MyDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, tokenizer, max_length):
    self.sentences=[]
    self.label=[]
    self.tokenizer = tokenizer
    self.max_length = max_length
    for i in dataset:
      sentence = '' + i[0][0] + ' ' + i[1][0] + ' ' + i[2][0] + ' '  + i[3][0]
      labell = [i[0][1],i[1][1],i[2][1],i[3][1]]
      self.sentences.append(sentence)
      self.label.append(labell)

  def __len__(self):
    return len(self.label)

  def __getitem__(self, index):
    sent = self.sentences[index]

    # Tokenize the pair of sentences to get token ids, attention masks and token type ids
    inputs = self.tokenizer.encode_plus(
      sent,
      None,
      add_special_tokens=True,
      max_length=self.max_length,
      pad_to_max_length=True,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
    )

    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs["token_type_ids"]

    return {
      'ids': torch.tensor(ids, dtype=torch.long),
      'mask': torch.tensor(mask, dtype=torch.long),
      'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
      'targets': torch.tensor(self.label[index], dtype=torch.long)
    } 
  

class BERT(torch.nn.Module):
  
  def __init__(self):
    super(BERT, self).__init__()
    self.bert_model = BertModel.from_pretrained("bert-base-uncased")
    #self.fc1 = nn.Linear(768,768)
    self.out = nn.Linear(768, 4)
    self.dropout = nn.Dropout(0.5)
  
  def forward(self, ids, mask, token_type_ids):
    _, out= self.bert_model(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
    out = self.dropout(out)
    #out= self.fc1(out)  
    #out=torch.relu(out) 
    out=self.out(out)
    return out




def eval(model, dataloader):
  model.eval()
  wrong_total=0
  total = 0
  for index, data in enumerate(dataloader):
    ids = data['ids'].cuda()
    token_type_ids=data['token_type_ids'].cuda()
    mask= data['mask'].cuda()
    label=data['targets']
    total+=label.size(dim=0)

    with torch.no_grad():
      output=model(ids=ids, mask=mask, token_type_ids=token_type_ids)
      act = nn.Softmax(dim=1)
      output = act(output)
      preds = output.detach().cpu()
      preds = torch.argmax(preds, dim=1)
      label = torch.argmax(label, dim=1)
      diff = preds - label
      wrong = diff.count_nonzero()
      wrong_total += wrong

  #print(total)
  #print(wrong_total)
  return 1-(wrong_total / total)


def train_model(train_dataloader, model, epoch, loss_fn, optimizer, valid_dataloader):
  model.train()
  loss_lst=[]

  for epo in range(epoch):
    print(epo)
    error = 0
    for index, data in enumerate(train_dataloader):
      ids = data['ids'].cuda()
      token_type_ids=data['token_type_ids'].cuda()
      mask= data['mask'].cuda()
      label=data['targets'].cuda()

      optimizer.zero_grad()

      output=model(ids=ids, mask=mask, token_type_ids=token_type_ids)
      label = label.type_as(output)
      loss=loss_fn(output,label)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      los_cpu = loss.detach().cpu()
      error += los_cpu
      
    print(f"Epoch {epo}/{epoch}  - Train accuracy: {eval(model,train_dataloader).item():.2f}  - Validation accuracy: {eval(model,valid_dataloader).item():.2f}")
  

    loss_lst.append(error)


def main():


    torch.manual_seed(0)
    answers = ['A','B','C','D']

    train = []
    test = []
    valid = []

    file_name = '/content/drive/MyDrive/ColabNotebooks/hw#2/train_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        train.append(obs)
        
        '''
        print(obs)
        print(' ')
        print(result['question']['stem'])
        print(' ',result['question']['choices'][0]['label'],result['question']['choices'][0]['text'])
        print(' ',result['question']['choices'][1]['label'],result['question']['choices'][1]['text'])
        print(' ',result['question']['choices'][2]['label'],result['question']['choices'][2]['text'])
        print(' ',result['question']['choices'][3]['label'],result['question']['choices'][3]['text'])
        print('  Fact: ',result['fact1'])
        print('  Answer: ',result['answerKey'])
        print('  ')
        '''
                
    file_name = '/content/drive/MyDrive/ColabNotebooks/hw#2/dev_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        valid.append(obs)
        
    file_name = '/content/drive/MyDrive/ColabNotebooks/hw#2/test_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        test.append(obs)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    linear = torch.rand(768,2)

    train_dataset = MyDataset(train, tokenizer, max_length = 170)
    valid_dataset = MyDataset(test, tokenizer, max_length = 170)
    test_dataset = MyDataset(valid, tokenizer, max_length = 170)

    train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32)
    valid_dataloader=torch.utils.data.DataLoader(dataset=valid_dataset,batch_size=32)
    test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=32)

    model = BERT()
    model = model.cuda()
    print(f"Before fine tuning, we test the model's accuracy for zero shot.   Train accuracy:  {eval(model,train_dataloader).item():.2f}, Valid accuracy:  {eval(model,valid_dataloader).item():.2f}, Test accuracy:  {eval(model,test_dataloader).item():.2f}")


    opt= optim.Adam(model.parameters(),lr= 3e-5)
    loss_f = torch.nn.CrossEntropyLoss()
    for param in model.bert_model.parameters():
        param.requires_grad = True

    train_model(train_dataloader, model, 6, loss_f, opt, valid_dataloader)

    print(f"After training for 6 epochs,  Train accuracy:  {eval(model,train_dataloader).item():.2f}, Valid accuracy:  {eval(model,valid_dataloader).item():.2f}, Test accuracy:  {eval(model,test_dataloader).item():.2f}")

    #torch.save(model.state_dict(), "/content/drive/MyDrive/ColabNotebooks/hw#2/bert_cls.pth")
    #torch.save(model.state_dict(), "/content/drive/MyDrive/ColabNotebooks/hw#2/bert_cls.pt")



if __name__ == "__main__":
    main()