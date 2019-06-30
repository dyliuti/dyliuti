import numpy as np
from datetime import datetime
import DataExtract
import DataTransform
from NeuralNetwork import error_rate
from sklearn.utils import shuffle

# keras test error rate 0.036071.
# Elapsted time for keras rmsprop:  0:00:25.107859
# tensorflow test error rate 0.025238.
# Elapsted time for tensorflow rmsprop:  0:08:09.270659

X_train, X_test, Y_train, Y_test =  DataExtract.load_minist_csv()
class_num = 10
Y_train_onehot = DataTransform.y2one_hot(Y_train, class_num=class_num)
Y_test_onehot  = DataTransform.y2one_hot(Y_test, class_num=class_num)


N, D = X_train.shape
M = 512
batch_size = 300
epochs = 50

# keras
from keras.models import Sequential
from keras.layers import Dense
# input: N,D  W1: D,M  W2: M,class_num
model = Sequential()
model.add(Dense(units=M, input_shape=(D, ), activation='relu'))   # input_dim = D， units为output个数
model.add(Dense(units=class_num))

weights0 = model.layers[0].get_weights()
W1 = weights0[0].copy()
b1 = weights0[1].copy()
weights1 = model.layers[1].get_weights()
W2 = weights1[0].copy()
b2 = weights1[1].copy()
config = model.get_config()

t0 = datetime.now()
model.compile(
	loss='mean_squared_error',
	metrics=['accuracy'],
	optimizer='rmsprop'
)

r = model.fit(X_train, Y_train_onehot, epochs=epochs, batch_size=batch_size)

p_y_test = model.predict(X_test)
error = error_rate(p_y_test, Y_test)
print("keras test error rate %f." % error)
print("Elapsted time for keras rmsprop: ", datetime.now() - t0)
print(r.history.keys())


# tensorflow
import tensorflow as tf
inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
outputs = tf.placeholder(tf.float32, shape=(None, class_num), name='outputs')

tfW1 = tf.Variable(W1)
tfb1 = tf.Variable(b1)
tfW2 = tf.Variable(W2)
tfb2 = tf.Variable(b2)

# opreation
logits = tf.matmul(tf.nn.relu(tf.matmul(inputs, tfW1) + tfb1), tfW2) + tfb2

# loss
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputs, logits=logits)
loss = tf.reduce_sum(entropy)
# loss = tf.reduce_mean(tf.square(pred - outputs))
train_optimize = tf.train.RMSPropOptimizer(learning_rate=1e-4, decay=0.99, momentum=0.9, epsilon=1e-8)
train_step = train_optimize.minimize(loss)
pred_index = tf.argmax(logits, axis=1)

tflosses = []
init = tf.global_variables_initializer()
t0 = datetime.now()
sess = tf.Session()
sess.run(init)
for epoch in range(epochs):
	tmpX, tmpY = shuffle(X_train, Y_train_onehot)
	for i in range(N // batch_size):
		X_batch = tmpX[i * batch_size: (i + 1) * batch_size, :]
		Y_batch = tmpY[i * batch_size: (i + 1) * batch_size, :]
		# 迭代一次，更新参数
		sess.run(train_step, feed_dict={inputs: X_batch, outputs: Y_batch})
		# 损失
		tfloss = sess.run(loss, feed_dict={inputs: X_train, outputs: Y_train_onehot})
		tflosses.append(tfloss)
		if i % 10 == 0:
			# 错误率
			p_y_train_idx = sess.run(pred_index, feed_dict={inputs: X_train})
			error = np.mean(p_y_train_idx != Y_train)
			print('epoch: %d, batch_num: %d, loss: %f' % (epoch, i, error))
# make prediction
p_y_test_idx = sess.run(pred_index, feed_dict={inputs: X_test})
error = np.mean(p_y_test_idx != Y_test)
print("tensorflow test error rate %f." % error)
print("Elapsted time for tensorflow rmsprop: ", datetime.now() - t0)


# theano
import theano
import theano.tensor as T

theano.config.floatX='float32'
# 2.定义变量与操作
inputs = T.matrix(name='inputs')
outputs = T.matrix(name='outputs')
thW1 = theano.shared(W1, name='W1')
thb1 = theano.shared(b1, name='b1')
thW2 = theano.shared(W2, name='W2')
thb2 = theano.shared(b2, name='b2')

# opreation
thZ = T.nnet.relu(inputs.dot(thW1) + thb1)
logits = thZ.dot(thW2) + thb2
# loss
pred_outputs = T.nnet.softmax(logits)
# cost function 交叉熵+正则
reg = 0.01
lr = 0.0004
cost = -(outputs * T.log(pred_outputs)).sum() + reg * ((thW1*thW1).sum() + (thb1*thb1).sum() + (thW2*thW2).sum() + (thb2*thb2).sum())
pred_index = T.argmax(pred_outputs, axis=1)

mu = 0.9
dW2 = 0
db2 = 0
dW1 = 0
db1 = 0
# 3. 在训练集上训练
dW2 = mu * dW2 - lr * T.grad(cost, thW2)
db2 = mu * db2 - lr * T.grad(cost, thb2)
dW1 = mu * dW1 - lr * T.grad(cost, thW1)
db1 = mu * db1 - lr * T.grad(cost, thb1)

update_W1 = (thW1 - lr * T.grad(cost, thW1))
update_b1 = thb1 - lr * T.grad(cost, thb1)
update_W2 = thW2 - lr * T.grad(cost, thW2)
update_b2 = thb2 - lr * T.grad(cost, thb2)

# thW1与update_W1类型要相同
train = theano.function(
	inputs=[inputs, outputs],
	updates=[(thW1, update_W1), (thb1, update_b1), (thW2, update_W2), (thb2, update_b2)],
)

get_prediction = theano.function(
	inputs=[inputs, outputs],
	outputs=[cost, pred_index],
)

t0 = datetime.now()
thlosses = []
for epoch in range(20):
	tmpX, tmpY = shuffle(X_train, Y_train_onehot)
	for i in range(N // batch_size):
		X_batch = tmpX[i * batch_size: (i + 1) * batch_size, :]
		Y_batch = tmpY[i * batch_size: (i + 1) * batch_size, :]
		# 迭代一次，更新参数
		train(X_batch, Y_batch)
		loss, prediction_val = get_prediction(X_test, Y_test_onehot)
		thlosses.append(loss)
		error = np.mean(prediction_val != Y_test)
		print('epoch: %d, batch_num: %d, loss: %f' % (epoch, i, error))
		loss, prediction_val = get_prediction(X_test, Y_test_onehot)
		error = np.mean(prediction_val != Y_test)
		print("theano test error rate %f." % error)

loss, prediction_val = get_prediction(X_test, Y_test_onehot)
error = np.mean(prediction_val != Y_test)
print("theano test error rate %f." % error)
print("Elapsted time for tensorflow rmsprop: ", datetime.now() - t0)

import matplotlib.pyplot as plt
plt.plot(tflosses, label='tensorflow')
plt.plot(thlosses, label='theano')



	



