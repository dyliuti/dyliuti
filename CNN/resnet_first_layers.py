import tensorflow as tf
import numpy as np
import keras

from keras.applications.resnet50 import ResNet50
from keras.models import Model

from CNN.resnet_convblock import ConvLayer, BatchNormLayer, ConvBlock

class ReLULayer:
	def forward(self, X):
		return tf.nn.relu(X)

	def get_params(self):
		return []

class MaxPoolLayer:
	def __init__(self, dim):
		self.dim = dim

	def forward(self, X):
		return tf.nn.max_pool(X, ksize=[1, self.dim, self.dim, 1], strides=[1, 2, 2, 1], padding='SAME')

	def get_params(self):
		return []

class PartialResNet:
	def __init__(self):
		self.layers = [
			# conv block 前的卷积层
			ConvLayer(d=7, mi=3, mo=64, stride=2, padding='SAME'),
			BatchNormLayer(64),
			ReLULayer(),
			MaxPoolLayer(dim=3),
			# conv block
			ConvBlock(mi=64, fm_sizes=[64, 64, 256], stride=1),
		]
		self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
		self.output = self.forward(self.input_)

	def copyFromKerasLayers(self, layers):
		self.layers[0].copyFromKerasLayers(layers[2])
		self.layers[1].copyFromKerasLayers(layers[3])
		self.layers[4].copyFromKerasLayers(layers[7:])

	def forward(self, X):
		for layer in self.layers:
			X = layer.forward(X)
		return X

	def predict(self, X):
		return self.session.run(self.output, feed_dict={self.input_: X})

	def set_session(self, session):
		self.session = session
		self.layers[0].session = session
		self.layers[1].session = session
		self.layers[4].set_session(session)

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()

# 对比第一个conv_block后的输出
if __name__ == '__main__':
	resnet = ResNet50(weights='imagenet')

	# 可以通过查看resnet来确定正确的层
	partial_model = Model(
		inputs=resnet.input,
		outputs=resnet.layers[16].output
	)
	print(partial_model.summary())
	# for layer in partial_model.layers:
	#   layer.trainable = False

	my_partial_resnet = PartialResNet()

	# 产生一张假图
	X = np.random.random((1, 224, 224, 3))

	# 得到keras的输出
	keras_output = partial_model.predict(X)

	# 得到自已创建的模型的输出
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