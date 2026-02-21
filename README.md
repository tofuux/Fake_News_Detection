# Fake News Detection using LSTM

This project implements a **Deep Learning–based Fake News Detection system** to classify news articles as **fake or real** using an **LSTM (Long Short-Term Memory)** neural network.

The model predicts the probability of a news article being fake and demonstrates reliable performance on validation data through sequence-based learning.

---

## Project Overview

Fake news poses serious social, political, and economic challenges. This project leverages **Natural Language Processing (NLP)** and **Deep Learning** techniques to automatically detect misinformation in news articles.

Key components of the system include:

- Text preprocessing (cleaning, normalization, stopword removal)
- Tokenization and sequence padding
- Pre-trained **GloVe embeddings** for semantic representation
- LSTM neural network for sequential text learning

---

## Dataset

The dataset used is a CSV file containing labeled news articles with the following columns:

- `id` : Unique identifier for each news article  
- `title` : Title of the news article  
- `author` : Author of the article  
- `text` : Full text content of the news article  
- `label` : Target variable  
  - `0` → Real news  
  - `1` → Fake news  

---

## Data Preprocessing

The following preprocessing steps are applied to the textual data:

1. Cleaning text format (lowercasing, punctuation removal)
2. Removing stopwords using **NLTK**
3. Tokenization using Keras Tokenizer
4. Padding and truncation of sequences to a fixed length

These steps ensure that the textual data is suitable for input into the LSTM model.

---

## Model Architecture

The model is implemented as a **Sequential LSTM neural network** with the following architecture:

- **Embedding Layer**  
  - Uses pre-trained **GloVe word embeddings** for semantic understanding

- **LSTM Layer**  
  - 128 units  
  - Dropout applied for regularization

- **Dense Layers**  
  - Fully connected layers for feature transformation

- **Output Layer**  
  - Sigmoid activation function  
  - Binary classification (Fake / Real)

---

## Evaluation

The model is evaluated on validation data using error-based metrics:

- **Mean Squared Error (MSE)**: 0.1809  
- **Root Mean Squared Error (RMSE)**: 0.4253  

These metrics indicate the model’s effectiveness in predicting the probability of fake news.

---

## Technologies Used

- Python  
- Jupyter Notebook  
- Pandas  
- NumPy  
- NLTK  
- TensorFlow / Keras  
- Pre-trained GloVe Embeddings  

---

## Future Enhancements

- Improve accuracy using larger and more diverse datasets
- Experiment with advanced architectures (BiLSTM, GRU, Transformer models)
- Integrate BERT or other contextual embeddings
- Deploy the model as a web-based application
- Add real-time news verification support

---

## Author

**Thrupthi Bhat**  
GitHub: https://github.com/tofuux

---

## License

This project is intended for academic and educational purposes.
