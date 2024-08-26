import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from main import tokenize, bag_of_word
from model import NeuralNet

# Đọc file intents.json
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['intent']
    tags.append(tag)
    for pat in intent['patterns']:
        word = tokenize(pat)
        all_words.extend(word)
        xy.append((word, tag))

ignore_words = ['?', '!', '.', ',']
x_train = []
y_train = []
for (pat_sen, tag) in xy:
    bag = bag_of_word(pat_sen, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)  # CrossEntropyLoss

x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatData(Dataset):
    def __init__(self):
        self.n_sample = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_sample

if __name__ == "__main__":
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(x_train[0])
    learning_rate = 0.001
    num_epochs = 1000

    dataset = ChatData()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device).long()  # Chuyển đổi labels sang LongTensor

            # Forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward và optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')

    print(f'final loss, loss = {loss.item():.4f}')
    data_dic = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }
    FILE = "data.pth"
    torch.save(data_dic, FILE)
    print(f'Training complete. File save to {FILE}')


