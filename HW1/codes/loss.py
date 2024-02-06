from __future__ import division
import numpy as np


class MSELoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        batch_size = input.shape[0]
        output = np.sum((input - target)**2) / batch_size
        return output
        # TODO END

    def backward(self, input, target):
		# TODO START
        gradient = 2 * (input - target)
        return gradient
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        batch_size = input.shape[0]
        max = np.max(input, axis=1, keepdims=True)
        p = np.exp(input-max) / np.sum(np.exp(input-max), axis=1, keepdims=True)
        output = -np.sum(target * np.log(p)) / batch_size
        return output
        # TODO END

    def backward(self, input, target):
        # TODO START
        max = np.max(input, axis=1, keepdims=True)
        return np.exp(input-max) / np.sum(np.exp(input-max), axis=1, keepdims=True) - target
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        d = 5
        batch_size = input.shape[0]
        xtn = np.sum(input * target, axis=1, keepdims=True)
        h = np.maximum(0, d - xtn + input)*(1 - target)
        return np.sum(h) / batch_size
        # TODO END

    def backward(self, input, target):
        # TODO START
        d = 5
        xtn = np.sum(input * target, axis=1, keepdims=True)
        grad_not_t = ((d - xtn + input) > 0) * (1 - target)
        grad_t = -np.sum(grad_not_t, axis=1, keepdims=True) * target
        return grad_t + grad_not_t
        # TODO END


# Bonus
class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=2.0):
        self.name = name
        if alpha is None:
            self.alpha = [0.1 for _ in range(10)]
        self.gamma = gamma

    def forward(self, input, target):
        # TODO START
        a = np.array(self.alpha)
        g = self.gamma
        batch_size = input.shape[0]
        max = np.max(input, axis=1, keepdims=True)
        p = np.exp(input-max) / np.sum(np.exp(input-max), axis=1, keepdims=True)
        e = (a * target + (1 - a) * (1 - target)) * (1 - p)**g * target * np.log(p)
        return -np.sum(e) / batch_size
        # TODO END

    def backward(self, input, target):
        # TODO START
        a = np.array(self.alpha)
        g = self.gamma
        max = np.max(input, axis=1, keepdims=True)
        p = np.exp(input-max) / np.sum(np.exp(input-max), axis=1, keepdims=True)
        deleh = (a * target + (1 - a) * (1 - target)) * (g * (1 - p)**(g - 1) * target * np.log(p) - (1 - p)**g * target / p)
        N = p.shape[0]
        K = p.shape[1]
        p_colvec = p.reshape(N, 1, K)
        p_rowvec = p.reshape(N, K, 1)
        I = np.eye(K)
        delhx = - p_rowvec @ p_colvec + I * p_colvec
        return np.sum(delhx * deleh[:,np.newaxis,:],axis=2)
        # TODO END
    