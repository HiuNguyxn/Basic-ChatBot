import random
import json
import torch
import numpy as np
from model import NeuralNet
from main import bag_of_word, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON file
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Load pre-trained model and data
FILE = 'data.pth'
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Prepare intents dictionary for easy access
responses_dict = {}
for intent_data in intents['intents']:
    tag = intent_data['intent']
    responses_dict[tag] = intent_data['responses']

bot_name = 'Hieu'


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_word(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        # Access the response using the 'tag' from the responses_dict
        if tag in responses_dict:
            return random.choice(responses_dict[tag])

    return "Xin lỗi tôi không hiểu ..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("Bạn: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(f"{bot_name}: {resp}")
