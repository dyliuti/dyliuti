import numpy as np
import tensorflow as tf
from Data import DataTransform
from Minist.Common.Util import gradb, gradW, cost, error_rate, derivative_b1, derivative_b2, derivative_w1, derivative_w2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def init_weight(M1, M2):
	return np.random.randn(M1, M2) * np.sqrt(2.0 / M1)

def init_weight_and_bias(M1, M2):
	return np.random.randn(M1, M2) / np.sqrt(M2), np.zeros(M2)


class LogisticModel(object):
	def __init__(self):
		pass

	def fit(self, X, Y, test_size=0.2, learning_rate=1e-6, reg=0, epochs=1000, show_fig=False):
		N, D = X.shape
		class_num = len(set(Y))
		self.W = np.random.randn(D, class_num) / np.sqrt(D)
		self.b = np.random.randn(class_num)

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
		Y_train_onehot = DataTransform.y2one_hot(Y_train, class_num=class_num)
		Y_test_onehot = DataTransform.y2one_hot(Y_test, class_num=class_num)

		costs = []
		for i in range(epochs):
			p_y = self.forward(X=X_train)
			# 训练一次，更新参数
			self.W -= learning_rate * (gradW(p_y, Y_train_onehot, X_train) + reg * self.W)
			self.b -= learning_rate * (gradb(p_y, Y_train_onehot) + reg * self.b)
			# 用训练好的参数预测一次
			p_y_test = self.forward(X=X_test)
			loss = cost(p_y_test, Y_test_onehot)
			costs.append(loss)
			error = error_rate(p_y_test, Y_test)
			if i % 10 == 0:
				print("Cost at iteration %d: %.6f" % (i, loss))
				print("Error rate: ", error)

		if show_fig:
			plt.plot(costs)
			plt.show()

	def forward(self, X):
		z = X.dot(self.W) + self.b
		a = z - np.max(z)  # 排除上溢（a最大为0，expz不会上溢）、下溢(分母必有一项1，不会因z都很小分母小)
		expz = np.exp(a)
		p_y = expz / expz.sum(axis=1, keepdims=True)  # 交叉熵计算需要log p_y p_y接近0也不太好
		return p_y

	def predict(self, X):
		p_y = self.forward(X)
		return np.argmax(p_y, axis=1)

	def score(self, X, Y):
		p_y_index = self.predict(X)
		return 1 - error_rate(p_y_index, Y)


# numpy ANN + momentum + l1 regularizatrion
class ANNModel(object):
	def __init__(self, hidden_layer_units=512):
		self.units = hidden_layer_units
		pass

	def fit(self, X, Y, batch_size=300, test_size=0.2, learning_rate=1e-5, mu = 0.9, reg=0, epochs=100, show_fig=False):
		N, D = X.shape
		class_num = len(set(Y))
		self.W1 = np.random.randn(D, self.units) / np.sqrt(D)
		self.b1 = np.random.randn(self.units)
		self.W2 = np.random.randn(self.units, class_num) / np.sqrt(self.units)
		self.b2 = np.random.randn(class_num)

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
		Y_train_onehot = DataTransform.y2one_hot(Y_train, class_num=class_num)
		Y_test_onehot = DataTransform.y2one_hot(Y_test, class_num=class_num)

		dW2 = 0
		db2 = 0
		dW1 = 0
		db1 = 0
		costs = []
		for i in range(50):
			tmpX, tmpY = shuffle(X_train, Y_train_onehot)
			for j in range(N//batch_size):
				X_batch = tmpX[j * batch_size: (j * batch_size + batch_size), :]
				Y_batch = tmpY[j * batch_size: (j * batch_size + batch_size), :]
				Z, p_y = self.forward(X_batch)

				# 更新权重
				gW2 = derivative_w2(Z, p_y, Y_batch) + reg * self.W2
				gb2 = derivative_b2(p_y, Y_batch) + reg * self.b2
				gW1 = derivative_w1(X_batch, Z, p_y, Y_batch, self.W2) + reg * self.W1
				gb1 = derivative_b1(Z, p_y, Y_batch, self.W2) + reg * self.b1

				dW2 = mu * dW2 - learning_rate * gW2
				db2 = mu * db2 - learning_rate * gb2
				dW1 = mu * dW1 - learning_rate * gW1
				db1 = mu * db1 - learning_rate * gb1

				# 更新权重
				self.W2 += dW2
				self.b2 += db2
				self.W1 += dW1
				self.b1 += db1

				# 测试集上计算loss
				_, p_y_test = self.forward(X_test)
				batch_loss = cost(p_y_test, Y_test_onehot)
				costs.append(batch_loss)
			error = error_rate(p_y_test, Y_test)  # 这里是Y_test，维度1
			if i % 10 == 0:
				print("Cost at iteration %d: %.6f" % (i, batch_loss))
				print("Error rate: ", error)

		if show_fig:
			plt.plot(costs)
			plt.show()
		# 验证集验证
		_, p_y = self.forward(X_test)
		print("\nFinal Error rate: ", error_rate(p_y, Y_test))

	def forward(self, X):
		Z = X.dot(self.W1) + self.b1
		Z[Z < 0] = 0

		A = Z.dot(self.W2) + self.b2
		expA = np.exp(A)
		Y = expA / expA.sum(axis=1, keepdims=True)
		return Z, Y

	def predict(self, X):
		_, p_y = self.forward(X)
		return np.argmax(p_y, axis=1)

	def score(self, X, Y):
		p_y_index = self.predict(X)
		return 1 - np.mean(p_y_index != Y)

class HiddenLayerBatchNorm(object):
	def __init__(self, M1, M2, f):
		self.M1 = M1
		self.M2 = M2
		self.f = f

		# 无参数B
		W = init_weight(M1, M2).astype(np.float32)
		gamma = np.ones(M2).astype(np.float32)
		beta = np.zeros(M2).astype(np.float32)

		self.W = tf.Variable(W)
		self.gamma = tf.Variable(gamma)
		self.beta = tf.Variable(beta)

		self.running_mean = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)
		self.running_var = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)

	def forward(self, X, is_training, decay=0.9):
		res = tf.matmul(X, self.W)
		if is_training:
			mean, var = tf.nn.moments(res, axes=[0]) # 不是axis
			update_running_mean = tf.assign(
				self.running_mean,
				self.running_mean * decay + mean * (1 - decay)
			)
			update_running_var = tf.assign(
				self.running_var,
				self.running_var * decay + var * (1 - decay)
			)
			# 计算normalization后的输出值，前提是上面两个tensor先被执行了(用control_dependencies保障)
			with tf.control_dependencies(control_inputs=[update_running_mean, update_running_var]):
				out = tf.nn.batch_normalization(
					res,
					mean,
					var,
					self.beta,
					self.gamma,
					variance_epsilon=1e-4
				)
		else:
			out = tf.nn.batch_normalization(
				res,
				self.running_mean,
				self.running_var,
				self.beta,
				self.gamma,
				variance_epsilon=1e-4
			)

		return self.f(out)

class HiddenLayer(object):
	def __init__(self, M1, M2):
		self.M1 = M1
		self.M2 = M2
		W = np.random.randn(M1, M2) / np.sqrt(M2)
		b = np.random.randn(M2)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))
		self.params = [self.W, self.b]

	def forward(self, X):
		return tf.nn.relu(tf.matmul(X, self.W) + self.b)
		
# tensorflow ANN + batch norm
class ANN(object):
	def __init__(self, hidden_layer_units):
		self.hidden_layer_units = hidden_layer_units
		session = tf.InteractiveSession()
		self.set_session(session)

	def set_session(self, session):
		self.sess = session

	def fit(self, X, Y, activation=tf.nn.relu, batch_size=300, test_size=0.2, learning_rate=1e-3, decay=0.9, momentum=0.9, reg=1e-3, epochs=10, show_fig=True):
		N, D = X.shape
		class_num = len(set(Y))
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
		Y_train_onehot = DataTransform.y2one_hot(Y_train, class_num=class_num)
		Y_test_onehot = DataTransform.y2one_hot(Y_test, class_num=class_num)

		# 中间层做batch norm
		M1 = D
		self.hidden_layers = []
		for M2 in self.hidden_layer_units:
			h = HiddenLayerBatchNorm(M1, M2, activation)
			self.hidden_layers.append(h)
			M1 = M2
		# 最后一层，获取logit
		W, b = init_weight_and_bias(M1, class_num)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

		# batch normalization已经包含了一定的正则，不对参数进行正则化处理

		inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
		outputs = tf.placeholder(tf.float32, shape=(None, class_num), name='outputs')
		logits = self.forward(inputs, is_training=True)
		test_logits = self.forward(inputs, is_training=False)
		# cost
		# regular_l2_cost = reg * sum(tf.nn.l2_loss(p) for p in self.params)
		# entropy shape is the same as labels
		entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputs, logits=logits)
		cost_ = tf.reduce_mean(entropy)

		# train_optimize = tf.train.AdamOptimizer(learning_rate)
		# train_optimize = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9)
		train_optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum, epsilon=1e-8)
		# train_optimize = tf.train.GradientDescentOptimizer(learning_rate)
		train_step = train_optimize.minimize(cost_)
		pred_index = tf.argmax(test_logits, axis=1)

		tflosses = []
		init = tf.global_variables_initializer()
		batch_num = N // batch_size
		self.sess.run(init)
		for epoch in range(epochs):
			tmpX, tmpY = shuffle(X_train, Y_train_onehot)
			for i in range(batch_num - 30):		# 用完整的batch_num gamma缩放参数会nan
				X_batch = tmpX[i * batch_size: (i + 1) * batch_size, :]
				Y_batch = tmpY[i * batch_size: (i + 1) * batch_size, :]
				# 迭代一次，更新参数
				self.sess.run(train_step, feed_dict={inputs: X_batch, outputs: Y_batch})
				# 损失
				tfloss = self.sess.run(cost_, feed_dict={inputs: X_train, outputs: Y_train_onehot})
				tflosses.append(tfloss)
				if i % 10 == 0:
					# 错误率
					p_y_train_idx = self.sess.run(pred_index, feed_dict={inputs: X_train})
					acc = np.mean(p_y_train_idx == Y_train)
					print('epoch: %d, batch_num: %d, accuracy rate: %f' % (epoch, i, acc))
					self.print_param()
		# make prediction
		if show_fig:
			plt.plot(tflosses)
			plt.show()
		p_y_test_idx = self.sess.run(pred_index, feed_dict={inputs: X_test})
		error = np.mean(p_y_test_idx != Y_test)
		print("tensorflow test error rate %f." % error)

	def forward(self, X, is_training):
		Z = X
		# batch normalization layer 前向传播
		for h in self.hidden_layers:
			Z = h.forward(Z, is_training)
		return tf.matmul(Z, self.W) + self.b

	def predict(self, X):
		p_y = self.forward(X)
		return tf.argmax(p_y, axis=1)

	def score(self, X, Y):
		p_y = self.predict(X)
		return np.mean(p_y == Y)

	def print_param(self):
		param = [[layer.W, layer.beta, layer.gamma] for layer in self.hidden_layers]
		params0 = [param[0][0]] + [param[0][1]] + [param[0][2]]
		params1 = [param[1][0]] + [param[1][1]] + [param[1][2]]
		params2 = [self.W] + [self.b]
		n_params0 = self.sess.run(params0)
		n_params1 = self.sess.run(params1)
		n_params2 = self.sess.run(params2)
		np.savetxt('param0.txt', n_params0, fmt='%s', newline='\n\n')
		np.savetxt('param1.txt', n_params1, fmt='%s', newline='\n\n')
		np.savetxt('param2.txt', n_params2, fmt='%s', newline='\n\n')


# tensorflow ANN + l2 regularization
class ANN_without_batch_normalization(object):
	def __init__(self, hidden_layer_units):
		self.hidden_layer_units = hidden_layer_units
		session = tf.InteractiveSession()
		self.set_session(session)

	def set_session(self, session):
		self.sess = session

	def fit(self, X, Y, activation=tf.nn.relu, batch_size=300, test_size=0.2, learning_rate=1e-5, decay=0.99,
			momentum=0.9, reg=1e-3, epochs=50, show_fig=False):
		N, D = X.shape
		class_num = len(set(Y))
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
		Y_train_onehot = DataTransform.y2one_hot(Y_train, class_num=class_num)
		Y_test_onehot = DataTransform.y2one_hot(Y_test, class_num=class_num)

		# 初始化变量与操作
		M1 = D
		self.hidden_layers = []
		for M2 in self.hidden_layer_units:
			h = HiddenLayer(M1, M2)
			self.hidden_layers.append(h)
			M1 = M2
		W, b = init_weight_and_bias(M1, class_num)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

		# 正则化项
		self.params = []
		for h in self.hidden_layers:
			self.params += h.params

		inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
		outputs = tf.placeholder(tf.float32, shape=(None, class_num), name='outputs')
		logits = self.forward(inputs)

		# cost 正则化
		regular_l2_cost = reg * sum(tf.nn.l2_loss(p) for p in self.params)
		# entropy shape is the same as labels
		entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputs, logits=logits)
		cost_ = tf.reduce_mean(entropy) + regular_l2_cost

		# train_optimize = tf.train.AdamOptimizer(learning_rate)
		# train_optimize = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9)
		train_optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum,
												   epsilon=1e-8)
		# train_optimize = tf.train.GradientDescentOptimizer(learning_rate)
		train_step = train_optimize.minimize(cost_)
		pred_index = tf.argmax(logits, axis=1)

		tflosses = []
		init = tf.global_variables_initializer()
		self.sess.run(init)
		for epoch in range(epochs):
			tmpX, tmpY = shuffle(X_train, Y_train_onehot)
			for i in range(N // batch_size):
				X_batch = tmpX[i * batch_size: (i + 1) * batch_size, :]
				Y_batch = tmpY[i * batch_size: (i + 1) * batch_size, :]
				# 迭代一次，更新参数
				self.sess.run(train_step, feed_dict={inputs: X_batch, outputs: Y_batch})
				# 损失
				tfloss = self.sess.run(cost_, feed_dict={inputs: X_train, outputs: Y_train_onehot})
				tflosses.append(tfloss)
				if i % 10 == 0:
					# 错误率
					p_y_train_idx = self.sess.run(pred_index, feed_dict={inputs: X_train})
					acc = np.mean(p_y_train_idx == Y_train)
					print('epoch: %d, batch_num: %d, accuracy rate: %f' % (epoch, i, acc))
		# make prediction
		if show_fig:
			plt.plot(tfloss)
			plt.show()
		p_y_test_idx = self.sess.run(pred_index, feed_dict={inputs: X_test})
		error = np.mean(p_y_test_idx != Y_test)
		print("tensorflow test error rate %f." % error)

	def forward(self, X):
		Z = X
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return tf.matmul(Z, self.W) + self.b

	def predict(self, X):
		p_y = self.forward(X)
		return tf.argmax(p_y, axis=1)