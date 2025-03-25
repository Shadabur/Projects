from gensim.downloader import load
import random

# Load the pre-trained GloVe model
glove_model = load("glove-wiki-gigaword-50")

# Function to construct a meaningful paragraph
def create_paragraph(topic_word, similar_words):  
  paragraph = f"The topic of {topic_word} is fascinating, often linked to terms like"
  random.shuffle(similar_words) # Shuffle to add variety

  for word in similar_words:
    paragraph += str(word) + ", "
    
  paragraph = paragraph.rstrip(", ") + "."
  return paragraph

topic_word = "hacking"  
similar_words_with_scores = glove_model.most_similar(topic_word, topn=5)  
similar_words = [word for word, similarity_score in similar_words_with_scores]  
paragraph = create_paragraph(topic_word, similar_words)
print(paragraph)
