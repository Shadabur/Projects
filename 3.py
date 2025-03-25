from gensim.models import Word2Vec 
from nltk.tokenize import word_tokenize 
import nltk
nltk.download('punkt_tab')
  # Download the necessary tokenizer

# Download necessary NLTK resources nltk.download('punkt') # For tokenization
# Sample corpus (you can load your own domain-specific text)
corpus = "The law of physics governs everything in the universe. Newton's laws are fundamental."

# Tokenize the text
tokens = word_tokenize(corpus.lower())

# Train a Word2Vec model
model = Word2Vec([tokens], min_count=1, vector_size=100, window=5)

# Save the trained model model.save("custom_word2vec.model")

# Check similar words for 'law'
similar_words = model.wv.most_similar('law', topn=5) 
print(similar_words)
