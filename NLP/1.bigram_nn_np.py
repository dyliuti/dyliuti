from Data.DataExtract import load_brown_with_limit_vocab
from Data.DataTransform import softmax, smoothed_loss
from NLP.Common.Util import get_bigram_prob
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

# 一个词预测下一个词

indexed_sentences, word2index = load_brown_with_limit_vocab(2000)
# 单词数
V = len(word2index)
start_index = word2index['START']
end_index = word2index['END']

bigram_probs = get_bigram_prob(indexed_sentences, V, start_index, end_index, smooth=0.1)

D = 100
W1 = np.random.randn(V, D) / np.sqrt(D)
W2 = np.random.randn(D, V) / np.sqrt(V)
W_bigram = np.log(bigram_probs)

lr = 0.001
losses = []
bigram_losses = []
epoches = 1
t0 = datetime.now()
for epoch in range(epoches):
	random.shuffle(indexed_sentences)

	i = 0
	for sentence in indexed_sentences:
		sentence = [word2index['START']] + sentence + [word2index['END']]
		n = len(sentence)
		# 1.one-hot 导致稀疏 前向传播效率低
		prev_sentence = np.zeros(shape=(n-1, V))
		next_sentence = np.zeros(shape=(n-1, V))
		prev_sentence[np.arange(n-1), sentence[:n-1]] = 1
		next_sentence[np.arange(n-1), sentence[1:]] = 1

		# 前向传播
		hidden = np.tanh(prev_sentence.dot(W1))
		predictions = softmax(hidden.dot(W2))
		# 更新权重
		W2 = W2  - lr * hidden.T.dot(predictions - next_sentence)
		dhidden = (predictions - next_sentence).dot(W2.T) * (1 - hidden * hidden)
		W1 = W1 - lr * prev_sentence.T.dot(hidden)

		# 交叉熵, 句子的样本均值（对比时，去除句子不定长影响）
		# 2.计算时也有许多无效运算
		loss = -np.sum(next_sentence * np.log(predictions)) / n - 1
		losses.append(loss)

		if epoch == 0:
			bigram_predictions = softmax(prev_sentence.dot(W_bigram))
			bigram_loss = -np.sum(next_sentence * np.log(bigram_predictions)) / (n - 1)
			bigram_losses.append(bigram_loss)

		if i % 100 == 0:
			print("epoch:", epoch, "sentence: %s/%s" % (i, len(sentence)), "loss:", loss)
		i += 1

print("Elapsed time training:", datetime.now() - t0)
# W_bigram = np.log(bigram_probs)
avg_bigram_loss = np.mean(bigram_losses)
print("avg_bigram_loss:", avg_bigram_loss)
plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')
plt.plot(smoothed_loss(losses))
plt.plot(losses)
plt.show()

# softmax is the opposite of the log()
plt.subplot(1, 2, 1)
plt.title("Neural Network Model")
plt.imshow(np.tanh(W1).dot(W2))
plt.subplot(1, 2, 2)
plt.title("Bigram Probs")
plt.imshow(W_bigram)
plt.show()


# 加速
D = 100
W1 = np.random.randn(V, D) / np.sqrt(D)
W2 = np.random.randn(D, V) / np.sqrt(V)
W_bigram = np.log(bigram_probs)

lr = 0.001
losses = []
bigram_losses = []
epoches = 1
t0 = datetime.now()
for epoch in range(epoches):
	random.shuffle(indexed_sentences)

	i = 0
	for sentence in indexed_sentences:
		sentence = [word2index['START']] + sentence + [word2index['END']]
		n = len(sentence)
		# 从one-hot NxV -> N
		prev_sentence = sentence[:n-1]
		next_sentence = sentence[1:]

		# 前向传播 1.利用稀疏矩阵运算特性 之前：nxV VxD -> nxD 现在：VxD中寻址选n个
		hidden = np.tanh(W1[prev_sentence])
		predictions = softmax(hidden.dot(W2))

		# 交叉熵, 句子的样本均值（对比时，去除句子不定长影响）
		# predictions被下面引用了，会变，这里提前了
		# 2.寻址计算交叉熵 之前：nxV - nxV 现在 nxV中寻址probality True项
		loss = -np.sum(np.log(predictions[np.arange(n - 1), next_sentence])) / (n - 1)
		losses.append(loss)

		# 更新权重
		doutput = predictions
		# 3.寻址更新doutput(y-y_true,next_sentence为y_true，每行只有一个值为1) 替代 preditions - one_hot
		doutput[np.arange(n - 1), next_sentence] -= 1
		W2 = W2  - lr * hidden.T.dot(doutput)
		dhidden = doutput.dot(W2.T) * (1 - hidden * hidden)
		np.subtract.at(W1, prev_sentence, lr * dhidden) # 区别于substract  a[indices] -= b  indics:n b:nxD

		# i = 0
		# for w in inputs: # 不包括开始结束标记'END'
		#   W1[w] = W1[w] - lr * dhidden[i]		# W1[x]作为输入取代x.dot(W1)
		#   i += 1

		# onehot_inputs = np.zeros((n - 1, V))
		# onehot_inputs[np.arange(n - 1), prev_sentence] = 1
		# W1 = W1 - lr * onehot_inputs.T.dot(dhidden) #  常规：onehot_inputs需要转置，但此时输入是一维list

		if epoch == 0:
			bigram_predictions = softmax(W_bigram[prev_sentence])
			bigram_loss = -np.sum(np.log(bigram_predictions[np.arange(n - 1), next_sentence])) / (n - 1)
			bigram_losses.append(bigram_loss)

		if i % 100 == 0:
			print("epoch:", epoch, "sentence: %s/%s" % (i, len(sentence)), "loss:", loss)
		i += 1

print("Elapsed time training:", datetime.now() - t0)
# W_bigram = np.log(bigram_probs)
avg_bigram_loss = np.mean(bigram_losses)
print("avg_bigram_loss:", avg_bigram_loss)
plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')
plt.plot(smoothed_loss(losses))
plt.plot(losses)
plt.show()

# softmax is the opposite of the log()
plt.subplot(1, 2, 1)
plt.title("Neural Network Model")
plt.imshow(np.tanh(W1).dot(W2))
plt.subplot(1, 2, 2)
plt.title("Bigram Probs")
plt.imshow(W_bigram)
plt.show()


