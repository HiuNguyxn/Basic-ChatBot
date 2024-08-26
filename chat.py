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
print("Cùng chat nào! gõ 'quit' để thoát ")
while True:
    sentence = input('Bạn: ')
    if sentence.lower() == 'quit':
        break

    # Tokenize and prepare input
    sentence_tokens = tokenize(sentence)
    X = bag_of_word(sentence_tokens, all_words)
    X = np.array(X).reshape(1, -1)  # Ensure X is 2D
    X = torch.from_numpy(X).float().to(device)

    # Get model output
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate probabilities
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        # Respond based on the predicted intent
        if tag in responses_dict:
            response = random.choice(responses_dict[tag])
            print(f"{bot_name}: {response}")
        else:
            print(f"{bot_name}: Tôi không có phản hồi cho ý định này.")
    else:
        print(f"{bot_name}: Tôi không hiểu câu bạn vừa hỏi ...")
