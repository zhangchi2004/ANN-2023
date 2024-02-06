# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = Parameter(torch.ones(num_features))
		self.bias = Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			mean = torch.mean(input, dim=0)
			var = torch.var(input, dim=0)
			self.running_mean = 0.9 * self.running_mean + 0.1 * mean
			self.running_var = 0.9 * self.running_var + 0.1 * var
			input = (input - mean) / torch.sqrt(var + 1e-5)
		else:
			input = (input - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
   
		input = input * self.weight + self.bias
		return input
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			mask = (torch.rand(input.shape) > self.p).float().to(device)
			input = input * mask / (1 - self.p)
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.lin1 = nn.Linear(3 * 32 * 32, 128)
		self.bn1 = BatchNorm1d(128)
		self.relu1 = nn.ReLU()
		self.dropout1 = Dropout(drop_rate)
		self.lin2 = nn.Linear(128, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		x = self.lin1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.dropout1(x)
		logits = self.lin2(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		#ADDED
		y = y.long()
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
