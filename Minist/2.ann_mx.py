from Data import DataExtract, DataTransform
import mxnet as mx


X_train, X_test, Y_train, Y_test =  DataExtract.load_minist_csv(pca=False)
class_num = 10
Y_train_onehot = DataTransform.y2one_hot(Y_train, class_num=class_num)
Y_test_onehot  = DataTransform.y2one_hot(Y_test, class_num=class_num)
X_train = mx.io.array(X_train)
X_test = mx.io.array(X_test)
Y_train = mx.io.array(Y_train)
Y_test = mx.io.array(Y_test)
Y_train_onehot = mx.io.array(Y_train_onehot)
Y_test_onehot = mx.io.array(Y_test_onehot)

N, D = X_train.shape
M = 512
batch_size = 300
epochs = 50

##### 建立模型 #####
data = mx.sym.Variable(name='data')
ful1 = mx.sym.FullyConnected(data=data, num_hidden=M, name='ful1')
act1 = mx.sym.Activation(data=ful1, act_type='relu', name='act1')
ful2 = mx.sym.FullyConnected(data=act1, num_hidden=M, name='ful2')
act2 = mx.sym.Activation(data=ful2, act_type='relu', name='act2')
ful3 = mx.sym.FullyConnected(data=act2, num_hidden=class_num, name='ful3')
sym = mx.sym.SoftmaxOutput(data=ful3, name='softmax')
print(sym.list_arguments())

train_iter = mx.io.NDArrayIter(data={'data': X_train},
							   label={'softmax_label': Y_train},
							   batch_size=batch_size,
							   shuffle=True)
val_iter = mx.io.NDArrayIter(data={'data': X_test},
							 label={'softmax_label': Y_test},
							 batch_size=batch_size)

model = mx.mod.Module(symbol=sym, context=mx.gpu())
model.fit(train_data=train_iter,
		  eval_data=val_iter,
		  eval_metric=['acc', 'loss'],
		  optimizer='rmsprop',
		  num_epoch=epochs)

# loss = mx.metric.Loss()
# acc = mx.metric.Accuracy()

