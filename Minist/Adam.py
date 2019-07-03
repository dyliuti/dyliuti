import numpy as np
from datetime import datetime
from Data import DataExtract, DataTransform
from Minist.Common.Util import forward, derivative_b1, derivative_b2, derivative_w1, derivative_w2, cost, error_rate
from sklearn.utils import shuffle

X_train, X_test, Y_train, Y_test =  DataExtract.load_minist_csv()
class_num = 10
Y_train_onehot = DataTransform.y2one_hot(Y_train, class_num=class_num)
Y_test_onehot  = DataTransform.y2one_hot(Y_test, class_num=class_num)


N, D = X_train.shape
M = 512

# para
W1 = np.random.randn(D, M) / np.sqrt(M)
b1 = np.zeros(M)
W2 = np.random.randn(M, class_num) / np.sqrt(class_num)
b2 = np.zeros(class_num)

W1_0 = W1.copy()
b1_0 = b1.copy()
W2_0 = W2.copy()
b2_0 = b2.copy()

# 共同参数
lr = 0.0001	# learning rate
reg = 0.01	# regularization
epochs = 100
batch_size = 300
n_batch = N // batch_size

# 1. Adam
# 1st moment 像 momentum
decay_rate1 = 0.9
mW1 = 0
mb1 = 0
mW2 = 0
mb2 = 0

# 2st moment 像 RMSProp
decay_rate2 = 0.999
eps = 1e-10
vW1 = 0
vb1 = 0
vW2 = 0
vb2 = 0

t = 1
losses_rms = []
t0 = datetime.now()
for i in range(epochs):
	tmpX, tmpY = shuffle(X_train, Y_train_onehot)
	for j in range(n_batch):
		X_batch = tmpX[j * batch_size : (j*batch_size + batch_size), :]
		Y_batch = tmpY[j * batch_size : (j*batch_size + batch_size), :]
		Z, p_y = forward(X_batch, W1, b1, W2, b2)

		# 计算导数
		gW1 = derivative_w1(X_batch, Z, p_y, Y_batch, W2) + reg * W1
		gb1 = derivative_b1(Z, p_y, Y_batch, W2) + reg * b1
		gW2 = derivative_w2(Z, p_y, Y_batch) + reg * W2
		gb2 = derivative_b2(p_y, Y_batch) + reg * b2

		# 指数衰减累加一阶导
		mW1 = decay_rate1 * mW1 + (1 - decay_rate1) * gW1
		mb1 = decay_rate1 * mb1 + (1 - decay_rate1) * gb1
		mW2 = decay_rate1 * mW2 + (1 - decay_rate1) * gW2
		mb2 = decay_rate1 * mb2 + (1 - decay_rate1) * gb2

		# 指数衰减累加二阶导
		vW1 = decay_rate2 * vW1 + (1 - decay_rate2) * gW1 * gW1
		vb1 = decay_rate2 * vb1 + (1 - decay_rate2) * gb1 * gb1
		vW2 = decay_rate2 * vW2 + (1 - decay_rate2) * gW2 * gW2
		vb2 = decay_rate2 * vb2 + (1 - decay_rate2) * gb2 * gb2

		# bias correction 去除低通滤波
		correction1 = 1 - decay_rate1 ** t
		hat_mW1 = mW1 / correction1
		hat_mb1 = mb1 / correction1
		hat_mW2 = mW2 / correction1
		hat_mb2 = mb2 / correction1

		correction2 = 1 - decay_rate2 ** t
		hat_vW1 = vW1 / correction2
		hat_vb1 = vb1 / correction2
		hat_vW2 = vW2 / correction2
		hat_vb2 = vb2 / correction2

		# 更新 t 与权重
		W1 -= lr * gW1 / (np.sqrt(hat_vW1) + eps)
		b1 -= lr * gb1 / (np.sqrt(hat_vb1) + eps)
		W2 -= lr * gW2 / (np.sqrt(hat_vW2) + eps)
		b2 -= lr * gb2 / (np.sqrt(hat_vb2) + eps)

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
print("Elapsted time for Adam: ", datetime.now() - t0)

import matplotlib.pyplot as plt
plt.plot(losses_rms, label='Adam')



# 2. RMSProp with momentum
W1 = W1_0.copy()
b1 = b1_0.copy()
W2 = W2_0.copy()
b2 = b2_0.copy()

# RMSProp
decay_rate = 0.999
eps = 1e-10
cache_W2 = 1
cache_b2 = 1
cache_W1 = 1
cache_b1 = 1

# momentum
mu = 0.9
dW1 = 0
db1 = 0
dW2 = 0
db2 = 0

losses_rms = []
t0 = datetime.now()
for i in range(epochs):
	tmpX, tmpY = shuffle(X_train, Y_train_onehot)
	for j in range(n_batch):
		X_batch = tmpX[j * batch_size : (j*batch_size + batch_size), :]
		Y_batch = tmpY[j * batch_size : (j*batch_size + batch_size), :]
		Z, p_y = forward(X_batch, W1, b1, W2, b2)

		# 计算导数
		gW1 = derivative_w1(X_batch, Z, p_y, Y_batch, W2) + reg * W1
		gb1 = derivative_b1(Z, p_y, Y_batch, W2) + reg * b1
		gW2 = derivative_w2(Z, p_y, Y_batch) + reg * W2
		gb2 = derivative_b2(p_y, Y_batch) + reg * b2

		cache_W1 = decay_rate * cache_W1 + (1 - decay_rate) * gW1 * gW1
		cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1 * gb1
		cache_W2 = decay_rate * cache_W2 + (1 - decay_rate) * gW2 * gW2
		cache_b2 = decay_rate * cache_b2 + (1 - decay_rate) * gb2 * gb2

		dW1 = mu * dW1 + (1 - mu) * lr * gW1 / (np.sqrt(cache_W1) + eps)
		db1 = mu * db1 + (1 - mu) * lr * gb1 / (np.sqrt(cache_b1) + eps)
		dW2 = mu * dW2 + (1 - mu) * lr * gW2 / (np.sqrt(cache_W2) + eps)
		db2 = mu * db2 + (1 - mu) * lr * gb2 / (np.sqrt(cache_b2) + eps)

		# 更新权重
		W1 -= dW1
		b1 -= db1
		W2 -= dW2
		b2 -= db2

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
print("Elapsted time for RMSProp with momentum: ", datetime.now() - t0)

import matplotlib.pyplot as plt
plt.plot(losses_rms, label='M-RMSProp')