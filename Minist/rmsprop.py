import numpy as np
from datetime import datetime
import DataExtract, DataTransform
from NeuralNetwork import forward, derivative_b1, derivative_b2, derivative_w1, derivative_w2, cost, error_rate
from sklearn.utils import shuffle

X_train, X_test, Y_train, Y_test =  DataExtract.load_minist_csv()
class_num = 10
Y_train_onehot = DataTransform.y2one_hot(Y_train, class_num=class_num)
Y_test_onehot  = DataTransform.y2one_hot(Y_test, class_num=class_num)


N, D = X_train.shape
M = 512

# batch gradient
W1 = np.random.randn(D, M) / np.sqrt(M)
b1 = np.zeros(M)
W2 = np.random.randn(M, class_num) / np.sqrt(class_num)
b2 = np.zeros(class_num)

# 2. batch with momentum
lr = 0.0001	# learning rate
reg = 0.01	# regularization
batch_size = 300
n_batch = N // batch_size
# RMSProp
cache_W2 = 1
cache_b2 = 1
cache_W1 = 1
cache_b1 = 1
decay_rate = 0.999
eps = 1e-10
losses_rms = []
t0 = datetime.now()

for i in range(50):
	tmpX, tmpY = shuffle(X_train, Y_train_onehot)
	for j in range(n_batch):
		X_batch = tmpX[j * batch_size : (j*batch_size + batch_size), :]
		Y_batch = tmpY[j * batch_size : (j*batch_size + batch_size), :]
		Z, p_y = forward(X_batch, W1, b1, W2, b2)

		# 计算导数
		gW2 = derivative_w2(Z, p_y, Y_batch) + reg * W2
		gb2 = derivative_b2(p_y, Y_batch) + reg * b2
		gW1 = derivative_w1(X_batch, Z, p_y, Y_batch, W2) + reg * W1
		gb1 = derivative_b1(Z, p_y, Y_batch, W2) + reg * b1

		cache_W2 = decay_rate * cache_W2 + (1 - decay_rate) * gW2 * gW2
		cache_b2 = decay_rate * cache_b2 + (1 - decay_rate) * gb2 * gb2
		cache_W1 = decay_rate * cache_W1 + (1 - decay_rate) * gW1 * gW1
		cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1 * gb1

		# 更新权重
		W2 -= lr * gW2 / (np.sqrt(cache_W2) + eps)
		b2 -= lr * gb2 / (np.sqrt(cache_b2) + eps)
		W1 -= lr * gW1 / (np.sqrt(cache_W1) + eps)
		b1 -= lr * gb1 / (np.sqrt(cache_b1) + eps)

		# 测试集上计算loss
		_, p_y_test = forward(X_test, W1, b1, W2, b2)
		batch_loss = cost(p_y_test, Y_test_onehot)
		losses_rms.append(batch_loss)
	error = error_rate(p_y_test, Y_test)	# 这里是Y_test，维度1
	if i % 10 == 0:
		print("Cost at iteration %d: %.6f" % (i, batch_loss))
		print("Error rate: ", error)
# 验证集验证
_, p_y  = forward(X_test, W1, b1, W2, b2)
print("\nFinal Error rate: ", error_rate(p_y, Y_test))
print("Elapsted time for RMSProp: ", datetime.now() - t0)

import matplotlib.pyplot as plt
plt.plot(losses_rms, label='RMSProp')




