# Fake News Detection using LSTM

This project implements a Deep Learning model to classify news articles as fake or real using an LSTM (Long Short-Term Memory) neural network.
The model predicts the probability of a news article being fake and achieves decent performance on validation data.
# 

## Project Overview
---
Fake news can have significant social and political impacts. This project uses NLP and deep learning to classify news articles as fake or real. It leverages:

- Text preprocessing (cleaning, stopword removal)
- Tokenization & padding
- Pre-trained GloVe embeddings for semantic understanding
- LSTM neural network for sequential learning
# ---

## Dataset
---

The dataset used is a CSV file containing the following columns:

- `id` : Unique identifier for each news article
- `title` : Title of the news article
- `author` : Author of the article
- `text` : Full text of the news article
- `label` : Target variable (0 for real news, 1 for fake news)
# ---

## Data Preprocessing
---

Steps applied to text data:

1. Clean data Format
2. Remove stopwords using NLTK

Tokenization and padding:

- Tokenizer fitted on the training data
- Sequences padded/truncated to tokens
# ---

## Model Architecture
---

The model is a Sequential LSTM network:

- Embedding layer: Pre-trained GloVe embeddings
- LSTM: 128 units with dropout for regularization
- Dense layers: Fully connected layers for final prediction
- Output layer: Sigmoid activation for binary classification
- Evaluation
- MSE (probabilities): 0.1809
- RMSE (probabilities): 0.4253
# ---

## Author
---
[Thrupthi Bhat](https://github.com/tofuux)
