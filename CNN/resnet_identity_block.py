import tensorflow as tf
import numpy as np
from CNN.resnet_convblock import ConvLayer, BatchNormLayer

class IdentityBlock:
	def __init__(self, mi, fm_sizes, activation=tf.nn.relu):
		self.session = None
		self.f = activation

		# 初始化主道
		# Conv -> BN -> F() ---> Conv -> BN -> F() ---> Conv -> BN
		self.conv1 = ConvLayer(1, mi, fm_sizes[0], 1)
		self.bn1 = BatchNormLayer(fm_sizes[0])
		self.conv2 = ConvLayer(3, fm_sizes[0], fm_sizes[1], 1, 'SAME')
		self.bn2 = BatchNormLayer(fm_sizes[1])
		self.conv3 = ConvLayer(1, fm_sizes[1], fm_sizes[2], 1)
		self.bn3 = BatchNormLayer(fm_sizes[2])

		# 以备后用
		self.layers = [
			self.conv1, self.bn1,
			self.conv2, self.bn2,
			self.conv3, self.bn3,
		]

		# 当从上一层传入输入时，将不使用此方法
		self.input_ = tf.placeholder(tf.float32, shape=(1, 224, 224, mi))
		self.output = self.forward(self.input_)

	def forward(self, X):
		# 主道
		FX = self.conv1.forward(X)
		FX = self.bn1.forward(FX)
		FX = self.f(FX)
		FX = self.conv2.forward(FX)
		FX = self.bn2.forward(FX)
		FX = self.f(FX)
		FX = self.conv3.forward(FX)
		FX = self.bn3.forward(FX)

		# identity的精髓，一个卷积都没，直传X， Fx就类似于残差了
		return self.f(FX + X)

	def predict(self, X):
		return self.session.run(self.output, feed_dict={self.input_: X})

	def set_session(self, session):
		self.session = session
		self.conv1.session = session
		self.bn1.session = session
		self.conv2.session = session
		self.bn2.session = session
		self.conv3.session = session
		self.bn3.session = session

	def copyFromKerasLayers(self, layers):
		# Conv2D -> BatchNormalization -> Activation
		# Conv2D -> BatchNormalization -> Activation
		# Conv2D -> BatchNormalization
		# Add  -> Activation
		self.conv1.copyFromKerasLayers(layers[0])
		self.bn1.copyFromKerasLayers(layers[1])
		self.conv2.copyFromKerasLayers(layers[3])
		self.bn2.copyFromKerasLayers(layers[4])
		self.conv3.copyFromKerasLayers(layers[6])
		self.bn3.copyFromKerasLayers(layers[7])

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params

if __name__ == '__main__':
	identity_block = IdentityBlock(mi=256, fm_sizes=[64, 64, 256])

	# 产生一张假图
	X = np.random.random((1, 224, 224, 256))

	init = tf.global_variables_initializer()
	with tf.Session() as session:
		identity_block.set_session(session)
		session.run(init)

		output = identity_block.predict(X)
		print("output.shape:", output.shape)

