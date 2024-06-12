# Chatbot for Online Bookstore

This repository contains the code to develop a chatbot for an online bookstore that assists users with various tasks such as providing product information, processing payment queries, delivering details, suggesting books, and even telling jokes.

## Installation

### PyTorch and Dependencies

To install PyTorch, please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

You also need to install `nltk`:

```bash
pip install nltk
```
## Usage
Training the Model
Run the training script:

```bash
python train.py
```

This will generate a data.pth file.

## Interacting with the Chatbot
Run the chat script:

```bash
python chat.py
```

## Features
The chatbot can handle the following types of queries:

Greetings
Product information
Payment queries
Delivery details
Book recommendations
Jokes

### Example Questions
"What kinds of items are there?"
"Do you accept Mastercard?"
"How long does delivery take?"
"Can you recommend a fiction book?"
"Tell me a joke!"

## Architecture Overview
The chatbot application consists of the following components:

### Intents File:
 A JSON file (intents.json) containing predefined tags, patterns, and responses.

### NLTK Utils:
 A Python module (nltk_utils.py) for tokenizing, stemming, and generating bag-of-words representations.

### Neural Network Model:
 A PyTorch-based neural network model defined in model.py for intent classification.

### Training Script:
 A training script (train.py) to preprocess data, train the neural network model, and save the trained model.

### Chat Script:
 A chat script (chat.py) to load the trained model and interact with the user through the command line.

### Flask Web App:
 A web application (app.py) using Flask to provide an API for the chatbot.

## Design Decisions
### Intent Classification:
 The chatbot uses a neural network-based approach to map user inputs to predefined intents.

### Data Preprocessing:
 Tokenization and stemming are employed to preprocess user inputs. The bag-of-words model is used to represent sentences as fixed-size vectors.

### Model Architecture:
 A simple multi-layer neural network with one hidden layer is used for intent classification. The network uses ReLU activation functions and CrossEntropyLoss for optimization.

### Confidence Threshold:
 A confidence threshold of 75% is set to ensure the model only responds when it is reasonably confident about the intent.

### Scalability:
 The Flask web application allows for easy deployment and scalability, enabling the chatbot to be accessed via an API.

## Implementation Details

### Intents File (intents.json):
Contains various tags representing different intents.
Each tag has associated patterns (example user inputs) and responses (bot replies).

### NLTK Utils (nltk_utils.py):
Functions for tokenizing sentences, stemming words, and creating bag-of-words representations.

### Neural Network Model (model.py):
Defines a NeuralNet class extending torch.nn.Module.
The network consists of an input layer, a hidden layer, and an output layer.

### Training Script (train.py):
Loads and preprocesses data from intents.json.
Tokenizes and stems words to create a vocabulary.
Converts sentences to bag-of-words vectors.
Trains the neural network using these vectors and saves the trained model.

### Chat Script (chat.py):
Loads the trained model and intents file.
Processes user input and predicts the intent using the model.
Responds based on the predicted intent with a confidence threshold.

### Flask Web App (app.py):
Exposes a /chat endpoint that accepts POST requests.
Processes the incoming message and responds with the chatbot's reply.

## Conclusion
This chatbot application uses a neural network for intent classification, trained with tokenized and stemmed input data converted into a bag-of-words representation. The chatbot is accessible via a Flask web application, making it easy to deploy and use as an API. Expanding the training data and refining the model can further enhance the chatbot's ability to handle a broader range of user queries effectively.






