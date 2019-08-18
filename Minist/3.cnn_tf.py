import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

##### 数据读取 #####
train = pd.read_csv('Data/Minist/train.csv')
Y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)

# 标准化
X_train = X_train / 255.0
# 转到图像处理的四维
X_train = X_train.values.reshape(-1,28,28,1)
# 目标one-hot编码
Y_train = to_categorical(Y_train, num_classes = 10)
# 训练集中分训练与验证两部分
random_seed = 2
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
N = len(Y_train)


inputs = tf.placeholder("float32", shape=(None, 28, 28, 1))
outputs = tf.placeholder("float32", shape=(None, 10))

model = tf.keras.Sequential()
model.add(layers.Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(layers.Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation = "relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation = "softmax"))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
# sparse_categorical_crossentropy 对应 目标是个index就行，不用转为one-hot
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=10, batch_size=100, validation_data=[X_test, Y_test])


# import numpy as np
# from sklearn.utils import shuffle

# def weight_variable(shape):
# 	return tf.Variable(tf.random_normal(shape)) # , stddev=0.1
#
# def bias_variable(shape):
# 	return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32))
#
# def conv2d(x, W):
# 	# [filter_height, filter_width, in_channels, out_channels]
# 	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
# def max_pool_2x2(x):
# 	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#
# # 第一层卷积
# W_conv1 = weight_variable(shape=(5, 5, 1, 32))
# b_conv1 = bias_variable([32])
# h_conv1 = tf.nn.relu(conv2d(inputs, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# # 第二层卷积
# W_conv2 = weight_variable(shape=(5, 5, 32, 64))
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# # 全连接层
# W_fc1 = weight_variable(shape=(7 * 7 * 64, 1024))
# b_fc1 = bias_variable([1024])
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
#
# W_fc2 = weight_variable(shape=(1024, 10))
# b_fc2 = bias_variable([10])
# preds = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# pred_index = tf.argmax(preds, axis=1)
#
# # loss
# # loss = -tf.reduce_sum(outputs * tf.log(preds))
# entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputs, logits=preds)
# loss = tf.reduce_sum(entropy)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
# # correct_prediction = tf.equal(tf.argmax(preds,1), tf.argmax(outputs, axis=1))
# # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
#
# sess.run(tf.global_variables_initializer())
# batch_size = 100
# tflosses = []
# for epoch in range(2):
# 	tmpX, tmpY = shuffle(X_train, Y_train)
# 	for i in range(N // batch_size):
# 		X_batch = tmpX[i * batch_size: (i + 1) * batch_size, :]
# 		Y_batch = tmpY[i * batch_size: (i + 1) * batch_size, :]
#
# 		# 迭代一次，更新参数
# 		sess.run(train_step, feed_dict={inputs: X_batch, outputs: Y_batch, keep_prob: 1.0}) #
# 		# 损失
# 		tfloss = sess.run(loss, feed_dict={inputs: X_batch, outputs: Y_batch, keep_prob: 1.0})
# 		tflosses.append(tfloss)
# 		if i % 10 == 0:
# 			# 测试集错误率
# 			p_y_train_idx = sess.run(pred_index, feed_dict={inputs: X_test, keep_prob: 1.0})
# 			acc = np.mean(p_y_train_idx == Y_test)
# 			print('epoch: %d, batch_num: %d, loss: %f, acc: %f' % (epoch, i, tfloss, acc))
#
# # make prediction
# p_y_test_idx = sess.run(pred_index, feed_dict={inputs: X_test, keep_prob: 1.0})
# acc = np.mean(p_y_test_idx == Y_test)
# print("tensorflow test acc rate %f." % acc)
