import numpy as np
from datetime import datetime
from Data import DataExtract, DataTransform
from Minist.Common.Util import forward, gradW, gradb, cost, error_rate
from sklearn.utils import shuffle

X_train, X_test, Y_train, Y_test =  DataExtract.load_minist_csv()
class_num = 10
Y_train_onehot = DataTransform.y2one_hot(Y_train, class_num=class_num)
Y_test_onehot  = DataTransform.y2one_hot(Y_test, class_num=class_num)


N, D = X_train.shape
W = np.random.randn(D, class_num) / np.sqrt(class_num)   # 不能太小 不然z为0 log有问题
b = np.zeros(class_num)	# 为10而非D，有意义的

lr = 0.0001	# learning rate
reg = 0.1	# regularization  动量 0.9
loss_list = []
# Full gradient
t0 = datetime.now()
for i in range(50):
	p_y = forward(X=X_train, W=W, b=b)
	# 训练一次，更新参数
	W -= lr * (gradW(p_y, Y_train_onehot, X_train) + reg * W)
	b -= lr * (gradb(p_y, Y_train_onehot) + reg * b)
	# 用训练好的参数预测一次
	p_y_test = forward(X=X_test, W=W, b=b)
	loss = cost(p_y_test, Y_test_onehot)
	loss_list.append(loss)
	error = error_rate(p_y_test, Y_test)
	if i % 10 == 0:
		print("Cost at iteration %d: %.6f" % (i, loss))
		print("Error rate: ", error)

# 验证集验证
p_y  = forward(X_test, W, b)
print("\nFinal Error rate: ", error_rate(p_y, Y_test))
print("Elapsted time for Full Gradient: ", datetime.now() - t0)


# stochastic gradient
W = np.random.randn(D, class_num) / np.sqrt(class_num)
b = np.zeros(10)
lr = 0.0001	# learning rate
reg = 0.1	# regularization
loss_list = []
t0 = datetime.now()
for i in range(50):
	# 随机选取样本
	tmpX, tmpY = shuffle(X_train, Y_train_onehot)	# 训练时Y_train_onehot 维度class_num
	for n in range(min(N, 500)):
		# 单个样本
		x = tmpX[n, :].reshape(1, D)
		y = tmpY[n, :].reshape(1, class_num)  # class_num 对应 Y_train_onehot 维度
		# 预测得到概率值
		p_y = forward(x, W, b)
		# 更新权重
		W -= lr * gradW(p_y, y, x)
		b -= lr * gradb(p_y, y)
		# 得到当前权重预测的错误率
		p_y_test = forward(X_test, W, b)
		stochastic_loss = cost(p_y_test, Y_test_onehot)
		loss_list.append(stochastic_loss)
	error = error_rate(p_y_test, Y_test)  # 这里是Y_test，维度1
	if i % 10 == 0:
		print("Cost at iteration %d: %.6f" % (i, stochastic_loss))
		print("Error rate: ", error)

# 验证集验证
p_y  = forward(X_test, W, b)
print("\nFinal Error rate: ", error_rate(p_y, Y_test))
print("Elapsted time for SGD: ", datetime.now() - t0)


# batch gradient
W = np.random.randn(D, class_num) / np.sqrt(class_num)
b = np.zeros(class_num)
lr = 0.0001	# learning rate
reg = 0.01	# regularization
loss_list = []
t0 = datetime.now()
batch_size = 300
n_batch = N // batch_size
for i in range(50):
	tmpX, tmpY = shuffle(X_train, Y_train_onehot)
	for j in range(n_batch):
		X_batch = tmpX[j * batch_size : (j*batch_size + batch_size), :]
		Y_batch = tmpY[j * batch_size : (j*batch_size + batch_size), :]
		p_y = forward(X_batch, W, b)
		# 更新权重
		W -= lr * (gradW(p_y, Y_batch, X_batch) + reg * W)
		b -= lr * (gradb(p_y, Y_batch) + reg * b)
		# 测试集上计算loss
		p_y_test = forward(X_test, W, b)
		batch_loss = cost(p_y_test, Y_test_onehot)
		loss_list.append(batch_loss)
	error = error_rate(p_y_test, Y_test)	# 这里是Y_test，维度1
	if i % 10 == 0:
		print("Cost at iteration %d: %.6f" % (i, batch_loss))
		print("Error rate: ", error)
# 验证集验证
p_y  = forward(X_test, W, b)
print("\nFinal Error rate: ", error_rate(p_y, Y_test))
print("Elapsted time for batch GD: ", datetime.now() - t0)


