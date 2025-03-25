# from gensim.downloader import load
# import torch
# from transformers import pipeline

# # Load pre-trained word embeddings (GloVe)
# embedding_model = load("glove-wiki-gigaword-50")  # GloVe model with 50 dimensions
# torch.manual_seed(42)

# # Define contextually relevant word enrichment
# def enrich_prompt(original_prompt):
#   enriched_prompt = ""  # Start with the original prompt
#   words = original_prompt.split()  # Split the prompt into words
  
#   for word in words:
#     similar_words = embedding_model.most_similar(word, topn=3)
#     enriched_words = []
    
#     for similar_word, _ in similar_words:
#       enriched_words.append(similar_word)
      
#     enriched_prompt += " " + " ".join(enriched_words)
#   return enriched_prompt

# # Example prompt to be enriched
# original_prompt = "lung cancer"
# enriched_prompt = enrich_prompt(original_prompt)

# # Display the results
# print("Original Prompt:", original_prompt)
# print("Enriched Prompt:", enriched_prompt)

# text_generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

# original_response = text_generator(
#     original_prompt,
#     max_length=200,
#     num_return_sequences=1,
#     no_repeat_ngram_size=2,
#     top_p=0.95,
#     temperature=0.7)
# print("Prompt response\n", original_response[0]["generated_text"])

# enriched_response = text_generator(
#     enriched_prompt,
#     max_length=200,
#     num_return_sequences=1,
#     no_repeat_ngram_size=2,
#     top_p=0.95,
#     temperature=0.7)
# print("Enriched prompt response\n", enriched_response[0]["generated_text"])

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from transformers import pipeline

nltk.download('punkt')

# Sample text
corpus = "The law of physics governs everything in the universe. Newton's laws are fundamental."
tokens = word_tokenize(corpus.lower())

# Train Word2Vec model
model = Word2Vec([tokens], min_count=1, vector_size=100, window=5)

# Get similar words
stopwords = {'the', 'of', 'and', 'in', 'to', 'a', 'is', 'for', 'on', 'that', 'are', 'with', '.', ',', "'s", 'it', 'as'}
filtered_words = [w for w, _ in model.wv.most_similar("law", topn=10) if w not in stopwords]

# Create enriched prompt
enriched_prompt = f"Discuss the importance of {', '.join(filtered_words[:3])} in modern science."

# Generate text using GPT-2
generator = pipeline("text-generation", model="gpt2")
generated_text = generator(enriched_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

print(generated_text)
