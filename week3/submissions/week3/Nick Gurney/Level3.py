# Imports 
# Tokenization
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')

# Needed for cosine similarity analysis
from numpy import dot
from numpy.linalg import norm

# Model 
import torch 
from torch import nn
import torch.optim as optim 

# Visualisation
import matplotlib.cm as cm 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE 
import seaborn as sns 

# Save model 
import pickle
'''
    Implementation of a Negative sampling Skip Gram model to try and measure output 
    based on similarity between the prompt and what is generated
'''

def data_cleaning(text_file):
    with open(text_file,"r") as f:
        text = f.read().replace("\n", " ")
    
    
    return get_tokens_from_text(text)


def tokens_to_index(tokens):
    vocab = sorted(set(tokens))
    word2index = {word: index for index, word in enumerate(vocab)}
    index2word = {index:word for index,word in enumerate(vocab)}
    return vocab, word2index, index2word



def generate_pairs(tokens, window_size):
    pairs = []
    for i in range(window_size, len(tokens)-window_size):
        target = tokens[i]
        pair = tokens[i-window_size:i+window_size+1]
        pair.remove(target)
        for word in pair:
            pairs.append((target,word))
    return pairs
    
def create_training_data(vec,word2index):
    X_train = []
    y_train = []

    for target,other in vec:
        X_train.append(word2index[target])
        y_train.append(word2index[other])
    
    X_train = torch.LongTensor(X_train)
    y_train = torch.LongTensor(y_train)

    return X_train, y_train

class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embed_size)
        self.context_embeddings = nn.Embedding(vocab_size, embed_size)


        initrange = 0.5 / embed_size
        self.target_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.uniform_(-0, 0)  

    def forward(self, target_ids, context_ids, neg_ids):
        v = self.target_embeddings(target_ids)

       
        u_pos = self.context_embeddings(context_ids)

        
        u_neg = self.context_embeddings(neg_ids)

        return v, u_pos, u_neg

def sample_negative_words(vocab_size, batch_size, k, true_context_ids):
    neg_samples = np.random.choice(vocab_size, size=(batch_size, k))
    
    # ensure no accidental positives
    for i in range(batch_size):
        for j in range(k):
            if neg_samples[i, j] == true_context_ids[i]:
                neg_samples[i, j] = (neg_samples[i, j] + 1) % vocab_size

    return torch.LongTensor(neg_samples)

def neg_sampling_loss(v, u_pos, u_neg):
    # positive score
    pos_score = torch.mul(v, u_pos).sum(dim=1)   # dot product
    pos_loss = torch.log(torch.sigmoid(pos_score))

    # negative score
    neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze()  # (batch,k)
    neg_loss = torch.log(torch.sigmoid(-neg_score)).sum(dim=1)

    return -torch.mean(pos_loss + neg_loss)


def train_neg_sampling(model, pairs, word2index, embed=100, epochs=5, lr=0.001, k=5, batch_size=32):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    vocab_size = len(word2index)

    X = [word2index[p[0]] for p in pairs]
    y = [word2index[p[1]] for p in pairs]

    X = torch.LongTensor(X)
    y = torch.LongTensor(y)

    for epoch in range(epochs):
        epoch_loss = 0
        
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            neg_ids = sample_negative_words(vocab_size, len(batch_x), k, batch_y)

            v, u_pos, u_neg = model(batch_x, batch_y, neg_ids)

            loss = neg_sampling_loss(v, u_pos, u_neg)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")
    return model 
    
def visualise_words_embeddings(model,epoch_number,vocab, word2index):
    words_to_visualise = vocab

    word_vectors = model.target_embeddings.weight.data 

    indices = [word2index[word] for word in words_to_visualise]
    word_vectors = model.target_embeddings.weight.data[indices]

    tsne = TSNE(n_components = 2,perplexity = 30)
    word_vectors_2d = tsne.fit_transform(word_vectors)

    colors = sns.husl_palette(n_colors = len(words_to_visualise))

    plt.figure(figsize = (12,12))

    for i, word in enumerate(words_to_visualise):
        plt.scatter(word_vectors_2d[:,0], word_vectors_2d[:,1], c = colors)
        plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

    plt.savefig(f'./submissions/week3/Nick Gurney/word_embeddings_epoch_{epoch_number}.png')
    


def cosine_similarity(a, b):
    if norm(a)==0 or norm(b)==0:
        return 0
    return dot(a, b) / (norm(a) * norm(b))


def sentence_embedding(sentence, word2index, model):
    after_filtering = get_tokens_from_text(sentence)
    vectors = [model.target_embeddings.weight.data[word2index[i]] for i in after_filtering if i in word2index]

    if len(vectors)  == 0:
        return np.zeros(len(word2index))
    
    return np.mean(vectors, axis = 0)

def get_sentences(text_file):
    with open(text_file,"r") as f:
        text = f.readlines()
    prompts = []
    generations = []
    flag = 0

    for line in range(len(text)):
        if "Prompt" in text[line]: 
            i = line
            lines = []
            out = ""
            while True:
                if "Generation:" in text[i] or i>=len(text):
                    for x in lines:
                        out = out + x
                    prompts.append(out)
                    break
                else:
                    lines.append(text[i])
                    i += 1

        if "Generation:" in text[line]: 
            i = line
            lines = []
            out = ""
            while True:
                if  i >= len(text) or "Prompt" in text[i]:
                    for x in lines:
                        out = out + x
                    generations.append(out)
                    break
                else:
                    lines.append(text[i])
                    i += 1
    return prompts,generations

def load_model():
    with open("./submissions/week3/Nick Gurney/SkipGramModel.model", "rb") as f:
        model = pickle.load(f)
    return model 

def save_model(model):
    with open("./submissions/week3/Nick Gurney/SkipGramModel.model","wb") as f:
        pickle.dump(model,f)

def get_tokens_from_text(sentence):
    text = sentence.lower().replace("\n", " ")

    text = text.replace("prompt", "").replace("generation", "")

    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))

    after_filtering = []

    for token in tokens:
        if not token in stop_words and token.isnumeric() == False and len(token) < 13:
            after_filtering.append(token)
    return after_filtering

def get_creativity_score(prompt, generation):
    repeated_words = 0
    prompt_tokens = get_tokens_from_text(prompt)
    generation_tokens = get_tokens_from_text(generation)

    for token in generation_tokens:
        if token in prompt_tokens:
            repeated_words += 1
    
    return ((len(generation_tokens) - repeated_words) / len(generation_tokens)) * 100
    

tf = "./submissions/week3/Nick Gurney/results.txt"
tokens = data_cleaning(tf)
vocab,word2index, index2word = tokens_to_index(tokens)

pairs = generate_pairs(tokens,3)
try:
    model = load_model()
except:
    tokens = data_cleaning(tf)
    vocab,word2index, index2word = tokens_to_index(tokens)

    pairs = generate_pairs(tokens,3)

    X_train, y_train = create_training_data(pairs,word2index)


    model = SkipGramNegSampling(len(vocab), 50)
    visualise_words_embeddings(model, "Before Training", vocab, word2index)
    model = train_neg_sampling(model, pairs, word2index, embed=50, epochs=50, lr = 0.001, k = 5, batch_size=16)
    visualise_words_embeddings(model, "After Training", vocab, word2index)
    save_model(model)


def get_relevance_score(similarity):
    return (similarity + 1) * 50


prompts, generations = get_sentences(tf)
# print(generations)

max_score = 0
max_prompt, max_generation = "", ""
min_score = 1000000000
min_prompt, min_generation = "", ""

for x,y in zip(prompts,generations):
    v1 = sentence_embedding(x,word2index,model)
    v2 = sentence_embedding(y, word2index, model)
    relevance_score = get_relevance_score(cosine_similarity(v1,v2))
    creativity_score = get_creativity_score(x, y)
    score = relevance_score * 0.8 + creativity_score * 0.2
    if score > max_score:
        max_score = score 
        max_prompt, max_generation = x, y 
    if score < min_score:
        min_score = score 
        min_prompt, min_generation = x, y
    print(get_relevance_score(cosine_similarity(v1,v2)))

print(f"\nMax Score: {max_score}")
print(max_prompt)
print(max_generation)

print(f"\nMin Score: {min_score}")
print(min_prompt)
print(min_generation)

v1 = sentence_embedding(prompts[1],word2index,model)
v2 = sentence_embedding(generations[4], word2index, model)

print(get_relevance_score(cosine_similarity(v1,v2)))