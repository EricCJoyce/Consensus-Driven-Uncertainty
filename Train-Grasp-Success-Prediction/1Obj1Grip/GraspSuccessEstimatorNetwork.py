import torch.nn as nn
import torch.nn.functional as F

class GraspSuccessEstimatorNetwork(nn.Module):
	def __init__(self, input_shape):
		super(GraspSuccessEstimatorNetwork, self).__init__()
		self.input_shape = input_shape								#  Save reference.

		self.linear_0 = nn.Linear(self.input_shape, 16)				#  input_shape inputs, 16 units.
		self.linear_1 = nn.Linear(16, 8)							#  16 inputs,          8 units.
		self.linear_2 = nn.Linear(8, 4)								#  8 inputs,           4 units.
		self.linear_3 = nn.Linear(4, 4)								#  4 inputs,           4 units.
		self.linear_4 = nn.Linear(4, 1)								#  4 inputs,           1 unit.

	#  Input shape is (batch-size, self.input_shape)
	def forward(self, x):
		y = F.relu(self.linear_0(x))
		y = F.relu(self.linear_1(y))
		y = F.relu(self.linear_2(y))
		y = F.relu(self.linear_3(y))
		y = F.sigmoid(self.linear_4(y))

		return y