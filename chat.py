import random
import json
import torch
import tkinter as tk
from tkinter import scrolledtext
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."

def send():
    msg = entry.get()
    if msg:
        chat_area.configure(state='normal')
        chat_area.insert(tk.END, "You: " + msg + '\n')
        response = get_response(msg)
        chat_area.insert(tk.END, "Bot: " + response + '\n')
        chat_area.configure(state='disabled')
        chat_area.yview(tk.END)
        entry.delete(0, tk.END)

# Create the main window
root = tk.Tk()
root.title("Chatbot")

# Create a frame for the text area and scrollbar
frame = tk.Frame(root)
scrollbar = tk.Scrollbar(frame)
chat_area = scrolledtext.ScrolledText(frame, wrap='word', state='disabled', yscrollcommand=scrollbar.set)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
scrollbar.config(command=chat_area.yview)
chat_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Create a text entry box
entry = tk.Entry(root, font=("Helvetica", 14))
entry.pack(pady=10, padx=10, fill=tk.X)

# Create a send button
send_button = tk.Button(root, text="Send", command=send)
send_button.pack(pady=10)

# Bind the enter key to send message
root.bind('<Return>', lambda event: send())

# Start the GUI event loop
root.mainloop()

