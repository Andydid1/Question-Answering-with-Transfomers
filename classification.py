from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
import torch
import math
import time
import sys
import json
import numpy as np
import matplotlib.pyplot as plt



#Data Parsing
def load_data(file_name): 
  input = []
  labels = []
  torch.manual_seed(0)
  answers = ['A','B','C','D']
  with open(file_name) as json_file:
      json_list = list(json_file)
      for i in range(len(json_list)):
          json_str = json_list[i]
          result = json.loads(json_str)
          
          base = result['fact1'] + ' [SEP] ' + result['question']['stem']
          ans = answers.index(result['answerKey'])
          
          for j in range(4):
              text = base + " " + result['question']['choices'][j]['text'] + ' [SEP]'
              if j == ans:
                  label = 1
              else:
                  label = 0
              input.append(text)
              labels.append(label)
  return input, labels

# Define the model
class BertClassifier(torch.nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased', output_attentions=False, output_hidden_states=False)
        self.classification_layer = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_layer(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classification_layer(pooled_output)
        return logits


#main
def main():
    torch.manual_seed(0)
    answers = ['A','B','C','D']

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load data
    train_input, train_labels = load_data('train_complete.jsonl')
    test_input, test_labels = load_data('test_complete.jsonl')
    val_input, val_labels = load_data('dev_complete.jsonl')

    # Define function to encode data using the tokenizer
    def encode_data(input_list, label_list):
        input_ids = []
        attention_masks = []
        labels = []
        for text, label in zip(input_list, label_list):
            encoded_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=256,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
                return_token_type_ids=False,

            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(label)
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
        return dataset

    # Encode data using the tokenizer
    train_dataset = encode_data(train_input, train_labels)
    test_dataset = encode_data(test_input, test_labels)
    val_dataset = encode_data(val_input, val_labels)

    # Define dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=32
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=32
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=32
    )

    # Instantiate BertClassifier
    model = BertClassifier().to(device)

    # Define optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,
    )
    # Train Model
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        train_correct_questions = 0
        train_total_questions = 0
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_masks, labels = batch  # unpack all three tensors
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_masks)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.squeeze(), labels.float().squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # compute accuracy
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).int()
            correct_predictions = predictions == labels.int().unsqueeze(1)
            for j in range(0, len(predictions), 4):
                if all(correct_predictions[j:j+4]):
                    train_correct_questions += 1
                train_total_questions += 1
        train_loss /= len(train_dataloader)
        train_accuracy = train_correct_questions / train_total_questions
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

      # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_correct_questions = 0
        val_total_questions = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_masks, labels = batch
                logits = model(input_ids, attention_mask=attention_masks)
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(logits.squeeze(), labels.float().squeeze())
                val_loss += loss.item()
                # compute accuracy
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).int()
                correct_predictions = predictions == labels.int().unsqueeze(1)
                for j in range(0, len(predictions), 4):
                    if all(correct_predictions[j:j+4]):
                        val_correct_questions += 1
                    val_total_questions += 1
        val_loss /= len(test_dataloader)
        val_accuracy = val_correct_questions / val_total_questions
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

    # Print metrics for current epoch 
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f} - Train accuracy: {train_accuracy:.4f} - Validation loss: {val_loss:.4f} - Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_correct_questions = 0  # count of correctly predicted questions
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_masks, labels = batch
            logits = model(input_ids, attention_mask=attention_masks)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.squeeze(), labels.float().squeeze())
            test_loss += loss.item()
            # compute accuracy
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).int()
            correct_predictions = predictions == labels.int().unsqueeze(1)
            for j in range(0, len(predictions), 4):
                if all(correct_predictions[j:j+4]):
                    test_correct_questions += 1
    test_loss /= len(test_dataloader)
    test_accuracy = test_correct_questions / (len(test_labels) / 4) # divide by 4 because each question has 4 options

    # Print test metrics
    print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_accuracy:.4f}")

    # Plot learning curve
    plt.plot(range(1, num_epochs+1), train_losses, label='Training loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
