from Data.DataExtract import load_wiki_with_limit_vocab
import numpy as np

indexed_sentences, word2index = load_wiki_with_limit_vocab(n_vocab=20000)
vocab_size = len(word2index)

# 参数
window_size = 5
learning_rate = 0.025
final_learning_rate = 0.0001
epochs = 20
D = 50	# word embedding size

# 线性递减学习率
learning_rate_delta = (learning_rate - final_learning_rate) / epochs

W1 = np.random.randn(vocab_size, D) / np.sqrt(D)
W2 = np.random.randn(D, vocab_size) / np.sqrt(vocab_size)
