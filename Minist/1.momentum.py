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

# batch gradient
W1 = np.random.randn(D, M) / np.sqrt(M)
b1 = np.zeros(M)
W2 = np.random.randn(M, class_num) / np.sqrt(class_num)
b2 = np.zeros(class_num)

# save initial weights
W1_0 = W1.copy()
b1_0 = b1.copy()
W2_0 = W2.copy()
b2_0 = b2.copy()

lr = 0.0001	# learning rate
reg = 0.01	# regularization
losses_batch = []
t0 = datetime.now()
batch_size = 300
n_batch = N // batch_size

# 1.batch
for i in range(50):
	tmpX, tmpY = shuffle(X_train, Y_train_onehot)
	for j in range(n_batch):
		X_batch = tmpX[j * batch_size : (j*batch_size + batch_size), :]
		Y_batch = tmpY[j * batch_size : (j*batch_size + batch_size), :]
		Z, p_y = forward(X_batch, W1, b1, W2, b2)

		# 更新权重
		W2 -= lr * (derivative_w2(Z, p_y, Y_batch) + reg * W2)
		b2 -= lr * (derivative_b2(p_y, Y_batch) + reg * b2)
		W1 -= lr * (derivative_w1(X_batch, Z, p_y, Y_batch, W2) + reg * W1)
		b1 -= lr * (derivative_b1(Z, p_y, Y_batch, W2) + reg * b1)
		# 测试集上计算loss
		_, p_y_test = forward(X_test, W1, b1, W2, b2)
		batch_loss = cost(p_y_test, Y_test_onehot)
		losses_batch.append(batch_loss)
	error = error_rate(p_y_test, Y_test)	# 这里是Y_test，维度1
	if i % 10 == 0:
		print("Cost at iteration %d: %.6f" % (i, batch_loss))
		print("Error rate: ", error)
# 验证集验证
_, p_y  = forward(X_test, W1, b1, W2, b2)
print("\nFinal Error rate: ", error_rate(p_y, Y_test))
print("Elapsted time for batch GD: ", datetime.now() - t0)


# 2. batch with momentum
W1 = W1_0.copy()
b1 = b1_0.copy()
W2 = W2_0.copy()
b2 = b2_0.copy()
mu = 0.9
dW2 = 0
db2 = 0
dW1 = 0
db1 = 0
losses_momentum = []
t0 = datetime.now()

for i in range(50):
	tmpX, tmpY = shuffle(X_train, Y_train_onehot)
	for j in range(n_batch):
		X_batch = tmpX[j * batch_size : (j*batch_size + batch_size), :]
		Y_batch = tmpY[j * batch_size : (j*batch_size + batch_size), :]
		Z, p_y = forward(X_batch, W1, b1, W2, b2)

		# 更新权重
		gW2 = derivative_w2(Z, p_y, Y_batch) + reg * W2
		gb2 = derivative_b2(p_y, Y_batch) + reg * b2
		gW1 = derivative_w1(X_batch, Z, p_y, Y_batch, W2) + reg * W1
		gb1 = derivative_b1(Z, p_y, Y_batch, W2) + reg * b1

		dW2 = mu * dW2 - lr * gW2
		db2 = mu * db2 - lr * gb2
		dW1 = mu * dW1 - lr * gW1
		db1 = mu * db1 - lr * gb1

		# 更新权重
		W2 += dW2
		b2 += db2
		W1 += dW1
		b1 += db1

		# 测试集上计算loss
		_, p_y_test = forward(X_test, W1, b1, W2, b2)
		batch_loss = cost(p_y_test, Y_test_onehot)
		losses_momentum.append(batch_loss)
	error = error_rate(p_y_test, Y_test)	# 这里是Y_test，维度1
	if i % 10 == 0:
		print("Cost at iteration %d: %.6f" % (i, batch_loss))
		print("Error rate: ", error)
# 验证集验证
_, p_y  = forward(X_test, W1, b1, W2, b2)
print("\nFinal Error rate: ", error_rate(p_y, Y_test))
print("Elapsted time for regular momentum: ", datetime.now() - t0)


# 3. batch with nesterov momentum
W1 = W1_0.copy()
b1 = b1_0.copy()
W2 = W2_0.copy()
b2 = b2_0.copy()
mu = 0.9
vW2 = 0
vb2 = 0
vW1 = 0
vb1 = 0
losses_nesterov = []
t0 = datetime.now()

for i in range(50):
	tmpX, tmpY = shuffle(X_train, Y_train_onehot)
	for j in range(n_batch):
		X_batch = tmpX[j * batch_size : (j*batch_size + batch_size), :]
		Y_batch = tmpY[j * batch_size : (j*batch_size + batch_size), :]
		Z, p_y = forward(X_batch, W1, b1, W2, b2)

		# 更新权重
		gW2 = derivative_w2(Z, p_y, Y_batch) + reg * W2
		gb2 = derivative_b2(p_y, Y_batch) + reg * b2
		gW1 = derivative_w1(X_batch, Z, p_y, Y_batch, W2) + reg * W1
		gb1 = derivative_b1(Z, p_y, Y_batch, W2) + reg * b1

		vW2 = mu * vW2 - lr * gW2
		vb2 = mu * vb2 - lr * gb2
		vW1 = mu * vW1 - lr * gW1
		vb1 = mu * vb1 - lr * gb1

		# 更新权重， v更新过，为当前的
		W2 += mu * vW2 - lr * gW2
		b2 += mu * vb2 - lr * gb2
		W1 += mu * vW1 - lr * gW1
		b1 += mu * vb1 - lr * gb1

		# 测试集上计算loss
		_, p_y_test = forward(X_test, W1, b1, W2, b2)
		batch_loss = cost(p_y_test, Y_test_onehot)
		losses_nesterov.append(batch_loss)
	error = error_rate(p_y_test, Y_test)	# 这里是Y_test，维度1
	if i % 10 == 0:
		print("Cost at iteration %d: %.6f" % (i, batch_loss))
		print("Error rate: ", error)
# 验证集验证
_, p_y  = forward(X_test, W1, b1, W2, b2)
print("\nFinal Error rate: ", error_rate(p_y, Y_test))
print("Elapsted time for nesterov momentum: ", datetime.now() - t0)


import matplotlib.pyplot as plt
plt.plot(losses_batch, label="batch")
plt.plot(losses_momentum, label="momentum")
plt.plot(losses_nesterov, label="nesterov")
plt.legend()
plt.show()

