import matplotlib
matplotlib.use('TkAgg') # Switch backend to TkAgg

from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import numpy as np

words = ['football', 'basketball', 'cricket', 'technology', 'computer', 'robot', 'AI', 'cloud', 'python', 'data']

word_vectors = {word: np.random.rand(100) for word in words} # Each word has a 100-dimensional vector

pca = PCA(n_components=2)
pca_result = pca.fit_transform([word_vectors[word] for word in words])

plt.scatter(pca_result[:, 0], pca_result[:, 1])

for i, word in enumerate(words):
 plt.annotate(word, (pca_result[i, 0], pca_result[i, 1]))
 plt.title('Word Embedding Visualization with PCA') 
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.decomposition import PCA

# # Sample words with random 100D vectors (Replace with actual embeddings)
# words = ['football', 'basketball', 'cricket', 'technology', 'computer', 'robot', 'AI', 'cloud', 'python', 'data']
# np.random.seed(42)  # Ensure reproducibility
# word_vectors = np.random.rand(len(words), 100)

# # Apply PCA to reduce to 2D
# pca_result = PCA(n_components=2).fit_transform(word_vectors)

# # Plot
# plt.figure(figsize=(8, 8))
# plt.scatter(pca_result[:, 0], pca_result[:, 1])
# for i, word in enumerate(words):
#     plt.annotate(word, pca_result[i], fontsize=10)
# plt.title('Word Embedding Visualization with PCA')
# plt.show()
