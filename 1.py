

from gensim.models import Word2Vec, KeyedVectors

# Download a small pre-trained word2vec model
import gensim.downloader as api
print("Loading model...")
w = api.load("word2vec-google-news-300")  # This is a lightweight alternative

# Perform word arithmetic: king - man + woman
r = w['king'] - w['man'] + w['woman']
s = w.similar_by_vector(r, topn=5)

# Print the top 5 similar words
print("\nTop 5 similar words:")
for word, score in s:
    print(f"{word}: {score}")




# from gensim.models import Word2Vec, KeyedVectors

# # Download a small pre-trained word2vec model
# import gensim.downloader as api
# word_vectors = api.load("word2vec-google-news-300")  # This is a lightweight alternative

# # Perform word arithmetic: king - man + woman
# result_vector = word_vectors['king'] - word_vectors['man'] + word_vectors['woman']
# similar_words = word_vectors.similar_by_vector(result_vector, topn=5)

# # Print the top 5 similar words
# print("\nTop 5 similar words:")
# for word, score in similar_words:
#     print(f"{word}: {score}")



# from gensim.models import KeyedVectors 
 
# # Load pre-trained GloVe vectors (100-dimensional) 
# from gensim.downloader import load 
# word_vectors = load('word2vec-google-news-300')  # Automatically downloads the model 
 
# # Example 1: Animal relationship (kitten → cat, puppy → dog) 
# result = word_vectors.most_similar(positive=['kitten', 'dog'], negative=['cat'], topn=1) 
# print("Result of 'kitten - cat + dog':", result[0][0])  # Expected output: 'puppy' or a related word 
 
# # Example 2: Fruit relationship (orange → fruit, mango → tropical fruit) 
# result = word_vectors.most_similar(positive=['orange', 'tropical'], negative=['fruit'], topn=1) 
# print("Result of 'orange - fruit + tropical':", result[0][0]) 