from Data.DataExtract import load_chunking
from Markov.hmm_discrete_model import HMM
import numpy as np
from sklearn.metrics import f1_score

def accuracy(tags, states):
	# 输入是 lists of lists
	n_correct = 0
	n_total = 0
	for tag, state in zip(tags, states):
		n_correct += np.sum(tag == state)
		n_total += len(state)
	return float(n_correct) / n_total

def total_f1_score(tags, states):
	# 输入是 lists of lists
	tags = np.concatenate(tags)
	states = np.concatenate(states)
	return f1_score(tags, states, average=None).mean()

# X: 词汇  Y: tag 词性
X_train, Y_train, X_test, Y_test, word2index = load_chunking(split_sequence=True)
# +1是因为序号从0开始
M = max(max(y) for y in Y_train) + 1
V = len(word2index) + 1		# +1 是因为 emit中类别从0开始，不然emit中下标会越界

smoothing = 0.1
pi = np.zeros(M)
trans_mat = np.ones(shape=(M, M)) * smoothing
# 计算首个词性的频数与词性->下个词性转换频数
for tags in Y_train:
	pi[tags[0]] += 1
	for i in range(len(tags) - 1):
		trans_mat[tags[i], tags[i+1]] += 1
# 得到pi与trans_mat概率
pi /= pi.sum()
trans_mat /= trans_mat.sum(axis=1, keepdims=True)

# 计算每个tag->每个词的频数
emit_mat = np.ones(shape=(M, V)) * smoothing
for sequence, tags in zip(X_train, Y_train):
	for word_index, tag in zip(sequence, tags):
		emit_mat[tag, word_index] += 1
emit_mat /= emit_mat.sum(axis=1, keepdims=True)

hmm = HMM(M)
hmm.pi = pi
hmm.trans_mat = trans_mat
hmm.emit_mat = emit_mat

# 解问题2：给定序列下，得到最有可能的状态
states_train = []
for sequence in X_train:
	states = hmm.get_state_sequence(sequence)
	states_train.append(states)

states_test = []
for sequence in X_test:
	states = hmm.get_state_sequence(sequence)
	states_test.append(states)

print("train accuracy:", accuracy(Y_train, states_train))
print("test accuracy:", accuracy(Y_test, states_test))
print("train f1:", total_f1_score(Y_train, states_train))
print("test f1:", total_f1_score(Y_test, states_test))


