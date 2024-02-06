import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None
        ## ADDED
        self._saved_tensor_2 = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

    ## ADDED
    def _saved_for_backward_2(self, tensor):
        self._saved_tensor_2 = tensor
    
class Relu(Layer):
	def __init__(self, name):
		super(Relu, self).__init__(name)

	def forward(self, input):
		self._saved_for_backward(input)
		return np.maximum(0, input)

	def backward(self, grad_output):
		input = self._saved_tensor
		return grad_output * (input > 0)

class Sigmoid(Layer):
	def __init__(self, name):
		super(Sigmoid, self).__init__(name)

	def forward(self, input):
		output = 1 / (1 + np.exp(-input))
		self._saved_for_backward(output)
		return output

	def backward(self, grad_output):
		output = self._saved_tensor
		return grad_output * output * (1 - output)

class Selu(Layer):
    def __init__(self, name):
        super(Selu, self).__init__(name)

    def forward(self, input):
        # TODO START
        l = 1.0507
        a = 1.67326
        self._saved_for_backward(input)
        output = l * input * (input>0) + l*a*(np.exp(input) - 1) * (input<=0)
        self._saved_for_backward_2(output)
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        l = 1.0507
        a = 1.67326
        input = self._saved_tensor
        output = self._saved_tensor_2
        return grad_output * (l * (input>0) + (output + l*a) * (input<=0))
        # TODO END

class Swish(Layer):
    def __init__(self, name):
        super(Swish, self).__init__(name)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        middle = 1 + np.exp(-input)
        self._saved_for_backward_2(middle)
        output = input / middle
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        input = self._saved_tensor
        middle = self._saved_tensor_2
        return grad_output * ((middle + input * (middle - 1))/(middle ** 2))
        # TODO END

class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)

    def forward(self, input):
        # TODO START
        c = 0.044715
        self._saved_for_backward(input)
        middle = np.sqrt(2/np.pi) * (input + c * (input ** 3))
        self._saved_for_backward_2(middle)
        output = 0.5 * input * (1 + np.tanh(middle))
        return output
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        input = self._saved_tensor
        middle = self._saved_tensor_2
        return grad_output * (0.5 * (1 + np.tanh(middle)) + (0.0535161 * (input**3) + 0.398942 * input) * (1/np.cosh(middle))**2)
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        output = input @ self.W + self.b
        self._saved_for_backward(input)
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        grad_input = grad_output @ self.W.T
        self.grad_W = self._saved_tensor.T @ grad_output
        self.grad_b = np.sum(grad_output)
        return grad_input
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
