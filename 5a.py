# from gensim.downloader import load
# import random

# # Load the pre-trained GloVe model
# glove_model = load("glove-wiki-gigaword-50")

# # Function to construct a meaningful paragraph
# def create_paragraph(topic_word, similar_words):  
#   paragraph = f"The topic of {topic_word} is fascinating, often linked to terms like"
#   random.shuffle(similar_words) # Shuffle to add variety

#   for word in similar_words:
#     paragraph += str(word) + ", "
    
#   paragraph = paragraph.rstrip(", ") + "."
#   return paragraph

# topic_word = "hacking"  
# similar_words_with_scores = glove_model.most_similar(topic_word, topn=5)  
# similar_words = [word for word, similarity_score in similar_words_with_scores]  
# paragraph = create_paragraph(topic_word, similar_words)
# print(paragraph)

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from transformers import pipeline

nltk.download('punkt')

def train_w2v(corpus):
    return Word2Vec([word_tokenize(corpus.lower())], min_count=1, vector_size=100, window=5)

def generate_text(prompt):
    return pipeline("text-generation", model="gpt2")(prompt, max_length=100, truncation=True)[0]['generated_text']

if __name__ == "__main__":
    model = train_w2v("The law of physics governs everything in the universe. Newton's laws are fundamental.")
    print("Similar words:", model.wv.most_similar("law", topn=5))
    print(generate_text("Discuss the importance of Newton's laws, motion, and force in modern science."))
