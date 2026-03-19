import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from numpy.linalg import norm

def softmax(u):
    u = u - np.max(u)
    exp_u = np.exp(u)
    return exp_u / np.sum(exp_u)

# ------------------ CORPUS ------------------
corpus = [
    "i love coding and building projects",
    "machine learning is fun and powerful",
    "deep learning uses neural networks",
    "coding in python is fun",
    "i enjoy learning new algorithms",
]

# Repeat corpus to strengthen patterns
corpus = corpus * 20

# Remove stopwords
stopwords = {"i", "is", "and", "in", "the", "a"}

corpus_clean = []
for sentence in corpus:
    words = [w for w in sentence.lower().split() if w not in stopwords]
    corpus_clean.append(" ".join(words))

corpus = corpus_clean

# ------------------ VOCAB ------------------
word_index = {}
index_word = {}

vocabulary = []
for sentence in corpus:
    for word in sentence.split():
        if word not in vocabulary:
            vocabulary.append(word)

for i, word in enumerate(vocabulary):
    word_index[word] = i
    index_word[i] = word

# ------------------ TRAINING DATA ------------------
window_size = 2
training_data = []

for sentence in corpus:
    words = sentence.split()
    for i in range(len(words)):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                target = word_index[words[i]]
                context = word_index[words[j]]
                training_data.append((target, context))

V = len(vocabulary)
d = 5   # smaller embedding

epoch = 1000
lr = 0.01

# ------------------ MODEL ------------------
W1 = np.random.randn(V, d) * 0.01
W2 = np.random.randn(d, V) * 0.01

for e in range(epoch):
    if e % 500 == 0:
        print("Epoch:", e)
    np.random.shuffle(training_data)
    for target, context in training_data:
        h = W1[target]
        u = h @ W2
        p = softmax(u)

        error = p.copy()
        error[context] -= 1

        dw1 = W2 @ error
        dw2 = np.outer(h, error)

        W1[target] -= lr * dw1
        W2 -= lr * dw2

# ------------------ SIMILARITY ------------------
def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def most_similar(word, top_k=5):
    target_vec = W1[word_index[word]]
    sims = []

    for w, idx in word_index.items():
        if w != word:
            sim = cosine(target_vec, W1[idx])
            sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

print("learning:", most_similar("learning"))
print("coding:", most_similar("coding"))

# ------------------ VISUALIZATION ------------------
embeddings = W1
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
reduced = tsne.fit_transform(embeddings)


plt.figure()

for i, word in index_word.items():
    x, y = reduced[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, word)

plt.title("Word Embeddings Visualization")
plt.show()