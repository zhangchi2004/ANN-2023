# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
#ADDED
import torch.cuda
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = Parameter(torch.ones(num_features))
		self.bias = Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			mean = torch.mean(input, dim=[0, 2, 3])
			var = torch.var(input, dim=[0, 2, 3])
			self.running_mean = 0.9 * self.running_mean + 0.1 * mean
			self.running_var = 0.9 * self.running_var + 0.1 * var
			input = (input - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + 1e-5)
		else:
			input = (input - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + 1e-5)
		return input
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			mask = torch.cuda.FloatTensor(*(input.shape[0],input.shape[1])).uniform_().view(input.shape[0],-1,1,1) > self.p
			input = input * mask / (1 - self.p)
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
		self.bn1 = BatchNorm2d(32)
		self.relu1 = nn.ReLU()
		self.dropout1 = Dropout(drop_rate)
		self.maxpool1 = nn.MaxPool2d(2, stride=2)
		self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
		self.bn2 = BatchNorm2d(64)
		self.relu2 = nn.ReLU()
		self.dropout2 = Dropout(drop_rate)
		self.maxpool2 = nn.MaxPool2d(2, stride=2)
		self.lin1 = nn.Linear(64 * 8 * 8, 10)
		
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.dropout1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu2(x)
		x = self.dropout2(x)
		x = self.maxpool2(x)
		x = x.view(x.shape[0], -1)
		logits = self.lin1(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		# ADDED
		y = y.long()
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
