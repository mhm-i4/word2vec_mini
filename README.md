# Word2Vec from Scratch (Skip-Gram)

This project implements the Word2Vec Skip-Gram model from scratch using NumPy.

## Features

* Vocabulary creation
* Skip-gram training data generation
* Forward pass (embedding → softmax)
* Backpropagation (manual gradients)
* Word similarity (cosine similarity)
* Visualization using PCA and t-SNE

## Example Output

* Words with similar context cluster together
* Example: "learning", "algorithms", "neural" appear closer

## Tech Stack

* Python
* NumPy
* Matplotlib
* Scikit-learn

## How to Run

```bash
pip install numpy matplotlib scikit-learn
python word2vec_mini.py
```
