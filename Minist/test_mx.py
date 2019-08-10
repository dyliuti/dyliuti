import mxnet as mx

A = mx.sym.Variable('A')
B = mx.sym.Variable('B')
C = A * B
D = mx.sym.Variable('D')
E = C + D
a = mx.nd.empty(1)
b = mx.nd.ones(1)
d = mx.nd.ones(1)

##### 测试 单输出 与多输出 #####
# executor = E.bind(ctx=mx.cpu(), args={'A':a, 'B':b, 'D':d})
G = mx.sym.Group([E, C])
executor = G.bind(ctx=mx.cpu(), args={'A':a, 'B':b, 'D':d})
a[:] = 10

executor.forward()
e_out = executor.outputs[0]
c_out = executor.outputs[1]

##### 测试 #####
import mxnet as mx

A = mx.sym.Variable('A')
B = mx.sym.Variable('B')
C = A * B
D = mx.sym.Variable('D')
E = C + D
a = mx.nd.empty(1)
b = mx.nd.ones(1)
d = mx.nd.ones(1)
grad_a = mx.nd.empty(4)
executor = E.bind(ctx=mx.cpu(), args={'A':a, 'B':b, 'D':d}, args_grad={'A': grad_a})
executor.forward()

# grad_e: backward的输入节点
grad_e = mx.nd.ones(1) * 2
executor.backward(out_grads=grad_e) #


##### simple bind #####
import mxnet as mx
in_ = mx.sym.Variable('data')
fc1 = mx.sym.FullyConnected(data=in_, num_hidden=128, name='fc1')
act1 = mx.sym.Activation(fc1, act_type='relu')

data_shape = (100, 768)
executor = act1.simple_bind(ctx=mx.cpu(), data=data_shape, grad_req='write')
arg_arrays = executor.arg_arrays
grad_arrays = executor.grad_arrays