from Data.DataExtract import load_brown_with_limit_vocab
from Data.DataTransform import softmax
from NLP.Common.Util import get_bigram_prob
import numpy as np
import random
import matplotlib.pyplot as plt

# 一个词预测下一个词

indexed_sentences, word2index = load_brown_with_limit_vocab()
# 单词数
V = len(word2index)
start_index = word2index['START']
end_index = word2index['END']

bigram_probs = get_bigram_prob(indexed_sentences, V, start_index, end_index, smooth=0.1)

D = 100
W1 = np.random.randn(V, D) / np.sqrt(D)
W2 = np.random.randn(D, V) / np.sqrt(V)

lr = 0.001
losses = []
epoches = 1
for epoch in range(epoches):
	random.shuffle(indexed_sentences)

	for sentence in indexed_sentences:
		sentence = word2index['START'] + sentence + word2index['END']
		n = len(sentence)
		prev_sentence = np.zeros(shape=(n-1, V))
		next_sentence = np.zeros(shape=(n-1, V))
		prev_sentence[np.arrange(n-1), sentence[:n-1]] = 1
		next_sentence[np.arrange(n-1), sentence[1:]] = 1

		hidden = np.tanh(prev_sentence.dot(W1))
		predictions = softmax(hidden.dot(W2))

		# 更新权重
		W2 = W2  - lr * hidden.T.dot(predictions - next_sentence)
		dhidden = (predictions - next_sentence).dot(W2.T) * (1 - hidden * hidden)
		W1 = W1 - lr * prev_sentence.T.dot(hidden)

		# 交叉熵, 句子的样本均值（对比时，去除句子不定长影响）
		loss = -np.sum(next_sentence * np.log(predictions)) / n - 1
		losses.append(loss)

W_bigram = np.log(bigram_probs)

#
plt.subplot(1, 2, 1)
plt.title("Neural Network Model")
plt.imshow(np.tanh(W1).dot(W2))
plt.subplot(1, 2, 2)
plt.title("Bigram Probs")
plt.imshow(W_bigram)
plt.show()

