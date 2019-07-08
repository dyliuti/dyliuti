import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def init_weight(Mi, Mo):
	return np.random.randn(Mi, Mo) * np.sqrt(Mi + Mo)

def get_bigram_prob(sentences, vocab_size, start_index, end_index, smooth=1):
	# 极大似然估计->贝叶斯估计
	bigram = np.ones((vocab_size, vocab_size)) * smooth
	for sentence in sentences:
		for i in range(len(sentence)):
			if i == 0:
				bigram[start_index, sentence[i]] += 1
			elif i == len(sentence) - 1:
				bigram[sentence[i], end_index] += 1
			else:
				bigram[sentence[i-1], sentence[i]] += 1
	# P(Y|X) = #(X, Y) / #(X)
	bigram_prob = bigram / np.sum(bigram, axis=1, keepdims=True)
	return bigram_prob


def get_score(bigram_probs, sentence, start_index, end_index):
	score = 0
	for i in range(len(sentence)):
		if i == 0:
			score += np.log(bigram_probs[start_index, sentence[i]])
		elif i == len(bigram_probs[0] - 1):
			score += np.log(bigram_probs[sentence[i], end_index])
		else:
			score += np.log(bigram_probs[sentence[i - 1], sentence[i]])

	# 标准化score
	return score / (len(sentence) + 1)


def analogy(word1, neg1, word2, neg2, word2index, index2word, word_embedding):
	V, D = word_embedding.shape
	print("testing: %s - %s = %s - %s" % (word1, neg1, word2, neg2))
	for word in (word1, neg1, word2, neg2):
		if word not in word2index:
			print('%s 不在word2index中。' % word)
			return

	p1 = word_embedding[word2index[word1]]
	n1 = word_embedding[word2index[neg1]]
	p2 = word_embedding[word2index[word2]]
	n2 = word_embedding[word2index[neg2]]

	vec = p1 - n1 + n2

	for dist in ('euclidean', 'cosine'):
		distances = pairwise_distances(vec.reshape(1, D), word_embedding, metric=dist).reshape(V)
		index = distances.argsort()[:6]

		# 已用的词汇
		used_word = [word2index[word] for word in (word1, neg1, neg2)]
		best_index = -1
		for i in index:
			if i not in used_word:
				best_index = i
				break
		best_word = index2word[best_index]
		print("got: %s - %s = %s - %s" % (word1, neg1, best_word, neg2))
		print("closest 6:")
		for i in index:
			print(index2word[i], distances[i])

		distance1 = pairwise_distances(vec.reshape(1, D), p2.reshape(1, D), metric=dist)
		distance2 = pairwise_distances(vec.reshape(1, D), word_embedding[best_index].reshape(1, D), metric=dist)
		print("closest match by '%s'." % dist, "word: %s." % word2, "distance: ", distance1[0][0])
		print("closest match by '%s'." % dist, "word: %s." % best_word, "distance: ", distance2[0][0])


def test_model(word2index, W1, W2):
	index2word = {i: w for w, i in word2index.items()}
	# 也可以试 We = W2.T
	for word_embedding in (W1, (W1 + W2.T) / 2):
		print("**********")

		analogy('king', 'man', 'queen', 'woman', word2index, index2word, word_embedding)
		analogy('king', 'prince', 'queen', 'princess', word2index, index2word, word_embedding)
		analogy('miami', 'florida', 'dallas', 'texas', word2index, index2word, word_embedding)
		analogy('einstein', 'scientist', 'picasso', 'painter', word2index, index2word, word_embedding)
		analogy('japan', 'sushi', 'germany', 'bratwurst', word2index, index2word, word_embedding)
		analogy('man', 'woman', 'he', 'she', word2index, index2word, word_embedding)

def all_parity_pairs(nbit):
	# total number of samples (Ntotal) will be a multiple of 100
	# why did I make it this way? I don't remember.
	N = 2 ** nbit
	remainder = 100 - (N % 100)
	Ntotal = N + remainder
	X = np.zeros((Ntotal, nbit))
	Y = np.zeros(Ntotal)
	for ii in range(Ntotal):
		i = ii % N
		# now generate the ith sample
		for j in range(nbit):
			if i % (2**(j+1)) != 0:
				i -= 2**j
				X[ii,j] = 1
		Y[ii] = X[ii].sum() % 2
	return X, Y

def all_parity_pairs_with_sequence_labels(nbit):
	X, Y = all_parity_pairs(nbit)
	N, t = X.shape

	# we want every time step to have a label
	Y_t = np.zeros(X.shape, dtype=np.int32) # Y target
	for n in range(N):
		ones_count = 0
		for i in range(t):
			if X[n,i] == 1:
				ones_count += 1
			if ones_count % 2 == 1:
				Y_t[n,i] = 1

	X = X.reshape(N, t, 1).astype(np.float32)
	return X, Y_t
