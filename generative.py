import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
import math
import time
import sys
import json
import numpy as np
import torch.nn as nn

PAD_SIZE = 128

class GPT2Dataset(torch.utils.data.Dataset):

  def __init__(self, input, labels, raw_labels, max_length, tokenizer):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []
    self.label_ids = []
    self.input_txt = []
    self.labels = []
    self.raw_labels = []

    for txt in input:
      encodings_dict = self.tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")
      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
      self.input_txt.append(txt)
    
    for txt in labels:
      encodings_dict = self.tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")
      self.label_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.labels.append(txt)
    
    for txt in raw_labels:
      self.raw_labels.append(txt)
    
  def __len__(self):
    return len(self.input_txt)

  def __getitem__(self, idx):
    item = {}
    item['input_ids'] = self.input_ids[idx]
    item['attn_masks'] = self.attn_masks[idx]
    item['label_ids'] = self.label_ids[idx]
    return item

def load_data_gpt(file_name): 
  input = []
  labels = []
  raw_labels = []
  with open(file_name) as json_file:
      json_list = list(json_file)
      for i in range(len(json_list)):
          json_str = json_list[i]
          result = json.loads(json_str)
          
          text = result['fact1'] + ' [SEP] ' + result['question']['stem']
          for j in range(4):
              text += ' [SEP] ' + result['question']['choices'][j]['label'] + ' ' + result['question']['choices'][j]['text']
          input.append(text)
          labels.append(text + " " + result['answerKey'])
          raw_labels.append(result['answerKey'])
          

  return input, labels, raw_labels

def eval_model(model, tokenizer, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(test_loader.dataset)):
            input_txt_curr = test_loader.dataset.input_txt[i]
            inputs = tokenizer(input_txt_curr, return_tensors="pt").to(device)
            # Generate new tokens
            output_ids = model.generate(**inputs, max_length=len(inputs['input_ids'][0])+1, pad_token_id=tokenizer.pad_token_id)

            # Decode the new tokens
            output_full = tokenizer.decode(output_ids[0],skip_special_tokens=False)
            generated_ans = output_full.split(' ')[-1]
            correct_ans = test_loader.dataset.raw_labels[i]
            if generated_ans == correct_ans:
              correct += 1
    print("Accuracy: ", (correct/len(test_loader.dataset)))

def train(model, optimizer, tokenizer, train_loader, valid_loader, epochs, device):
    logs = {
    'train loss':[],
    'val loss':[],
    }
    for epoch in range(epochs):
        print("start training epoch no.", epoch+1)
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            model.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attn_masks = batch['attn_masks'].to(device)
            label_ids = batch['label_ids'].to(device)
            outputs = model(input_ids,
                  labels=label_ids, 
                  attention_mask = attn_masks,
                  token_type_ids=None
                )
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        # print(scheduler.get_lr())

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            print("start validating epoch no.", epoch+1)
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attn_masks"].to(device)
                labels = batch["label_ids"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=None)
                loss = outputs[0]
                valid_loss += loss.item()
        

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        logs['train loss'].append(train_loss)
        logs['val loss'].append(valid_loss)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Validation Loss = {valid_loss:.4f}")
        eval_model(model, tokenizer, valid_loader, device)
    return model, logs

if __name__ == "__main__":
    train_input, train_labels, train_raw_labels = load_data_gpt('train_complete.jsonl')
    test_input, test_labels, test_raw_labels = load_data_gpt('test_complete.jsonl')
    val_input, val_labels, val_raw_labels = load_data_gpt('dev_complete.jsonl')

    torch.cuda.empty_cache()
    # Load pre-trained GPT-2 model and tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|pad|>') 
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataset = GPT2Dataset(train_input, train_labels, train_raw_labels, PAD_SIZE, tokenizer)
    test_dataset = GPT2Dataset(test_input, test_labels, test_raw_labels, PAD_SIZE, tokenizer)
    val_dataset = GPT2Dataset(val_input, val_labels, val_raw_labels, PAD_SIZE, tokenizer)
    
    batch_size = 32

    train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            batch_size = batch_size # Trains with this batch size.
        )
    test_dataloader = DataLoader(
                test_dataset,  # The training samples.
                batch_size = batch_size # Trains with this batch size.
            )
    val_dataloader = DataLoader(
                val_dataset,  # The training samples.
                batch_size = batch_size # Trains with this batch size.
            )

    epochs = 15
    learning_rate = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps = epsilon)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)

    torch.cuda.empty_cache()
    model, logs = train(model, optimizer, tokenizer, train_dataloader, val_dataloader, epochs, device)
    eval_model(model, tokenizer, test_dataloader, device)