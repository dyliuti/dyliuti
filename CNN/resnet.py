import numpy as np
import keras
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense

from CNN.resnet_convblock import ConvLayer, BatchNormLayer, ConvBlock
from CNN.resnet_identity_block import IdentityBlock
from CNN.resnet_first_layers import ReLULayer, MaxPoolLayer

class AvgPool:
	def __init__(self, ksize):
		self.ksize = ksize

	def forward(self, X):
		return tf.nn.avg_pool(X, ksize=[1, self.ksize, self.ksize, 1], strides=[1, 1, 1, 1], padding='VALID')

	def get_params(self):
		return []

class Flatten:
	def forward(self, X):
		return tf.contrib.layers.flatten(X)

	def get_params(self):
		return []

def custom_softmax(x):
	m = tf.reduce_max(x, 1)
	x = x - m
	e = tf.exp(x)
	return e / tf.reduce_sum(e, -1)

class DenseLayer:
	def __init__(self, mi, mo):
		self.W = tf.Variable((np.random.randn(mi, mo) * np.sqrt(2.0 / mi)).astype(np.float32))
		self.b = tf.Variable(np.zeros(mo, dtype=np.float32))

	def forward(self, X):
		# 下面这些都会导致结果出现细微的不同
		# return tf.nn.softmax(tf.matmul(X, self.W) + self.b)
		# return custom_softmax(tf.matmul(X, self.W) + self.b)
		# return keras.activations.softmax(tf.matmul(X, self.W) + self.b)
		return tf.matmul(X, self.W) + self.b

	def copyFromKerasLayers(self, layer):
		W, b = layer.get_weights()
		op1 = self.W.assign(W)
		op2 = self.b.assign(b)
		self.session.run((op1, op2))

	def get_params(self):
		return [self.W, self.b]

class TFResNet:
	def __init__(self):
		self.layers = [
			# before conv block
			ConvLayer(d=7, mi=3, mo=64, stride=2, padding='SAME'),
			BatchNormLayer(64),
			ReLULayer(),
			MaxPoolLayer(dim=3),
			# conv block
			ConvBlock(mi=64, fm_sizes=[64, 64, 256], stride=1),
			# identity block x 2
			IdentityBlock(mi=256, fm_sizes=[64, 64, 256]),
			IdentityBlock(mi=256, fm_sizes=[64, 64, 256]),
			# conv block
			ConvBlock(mi=256, fm_sizes=[128, 128, 512], stride=2),
			# identity block x 3
			IdentityBlock(mi=512, fm_sizes=[128, 128, 512]),
			IdentityBlock(mi=512, fm_sizes=[128, 128, 512]),
			IdentityBlock(mi=512, fm_sizes=[128, 128, 512]),
			# conv block
			ConvBlock(mi=512, fm_sizes=[256, 256, 1024], stride=2),
			# identity block x 5
			IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]),
			IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]),
			IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]),
			IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]),
			IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]),
			# conv block
			ConvBlock(mi=1024, fm_sizes=[512, 512, 2048], stride=2),
			# identity block x 2
			IdentityBlock(mi=2048, fm_sizes=[512, 512, 2048]),
			IdentityBlock(mi=2048, fm_sizes=[512, 512, 2048]),
			# pool / flatten / dense
			AvgPool(ksize=7),
			Flatten(),
			DenseLayer(mi=2048, mo=1000)
		]
		self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
		self.output = self.forward(self.input_)

	def copyFromKerasLayers(self, layers):
		# conv
		self.layers[0].copyFromKerasLayers(layers[2])
		# bn
		self.layers[1].copyFromKerasLayers(layers[3])
		# cb
		self.layers[4].copyFromKerasLayers(layers[7:17]) # size=12
		# ib x 2
		self.layers[5].copyFromKerasLayers(layers[19:27]) # size=10
		self.layers[6].copyFromKerasLayers(layers[29:37])
		# cb
		self.layers[7].copyFromKerasLayers(layers[39:49])
		# ib x 3
		self.layers[8].copyFromKerasLayers(layers[51:59])
		self.layers[9].copyFromKerasLayers(layers[61:69])
		self.layers[10].copyFromKerasLayers(layers[71:79])
		# cb
		self.layers[11].copyFromKerasLayers(layers[81:91])
		# ib x 5
		self.layers[12].copyFromKerasLayers(layers[93:101])
		self.layers[13].copyFromKerasLayers(layers[103:111])
		self.layers[14].copyFromKerasLayers(layers[113:121])
		self.layers[15].copyFromKerasLayers(layers[123:131])
		self.layers[16].copyFromKerasLayers(layers[133:141])
		# cb
		self.layers[17].copyFromKerasLayers(layers[143:153])
		# ib x 2
		self.layers[18].copyFromKerasLayers(layers[155:163])
		self.layers[19].copyFromKerasLayers(layers[165:173])
		# dense
		self.layers[22].copyFromKerasLayers(layers[176])


	def forward(self, X):
		for layer in self.layers:
			X = layer.forward(X)
		return X

	def predict(self, X):
		return self.session.run(self.output, feed_dict={self.input_: X})

	def set_session(self, session):
		self.session = session
		for layer in self.layers:
			if isinstance(layer, ConvBlock) or isinstance(layer, IdentityBlock):
				layer.set_session(session)
			else:
				layer.session = session

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()


if __name__ == '__main__':
	resnet_ = ResNet50(weights='imagenet')

	# 创建一个不包括 softmax 的 resnet
	x = resnet_.layers[-2].output
	W, b = resnet_.layers[-1].get_weights()
	y = Dense(1000)(x)
	resnet = Model(resnet_.input, y)
	resnet.layers[-1].set_weights([W, b])

	# 可以通过查看resnet来确定正确的层
	partial_model = Model(
		inputs=resnet.input,
		outputs=resnet.layers[176].output
	)

	print(partial_model.summary())

	my_partial_resnet = TFResNet()

	# 产生一张假图
	X = np.random.random((1, 224, 224, 3))

	# 得到keras的输出
	keras_output = partial_model.predict(X)

	### 获得模型输出 ###

	# 初始化变量
	init = tf.variables_initializer(my_partial_resnet.get_params())

	# 要注意：重新开启新的session，会打乱keras model
	session = keras.backend.get_session()
	my_partial_resnet.set_session(session)
	session.run(init)

	# 先确定下可以得到输出
	first_output = my_partial_resnet.predict(X)
	print("first_output.shape:", first_output.shape)

	# 从 Keras model 中拷贝参数
	my_partial_resnet.copyFromKerasLayers(resnet.layers)
	# my_partial_resnet.copyFromKerasLayers(partial_model.layers)

	# 比对2个 model
	output = my_partial_resnet.predict(X)
	print(first_output.sum())
	print(output.sum())
	print(keras_output.sum())
	diff = np.abs(output - keras_output).sum()
	if diff < 1e-10:
		print("OK的!")
	else:
		print("diff = %s" % diff)