import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable


class BlackBox(nn.Module):
		def __init__(self, inp_shape):
		    
			super(BlackBox, self).__init__()
			self.inp_shape = inp_shape
			self.hidden_dim_1 = 10
			self.hidden_dim_2 = 128
			self.hidden_dim_3 = 64
			self.predict_net= nn.Sequential(
		                 nn.Linear(self.inp_shape, self.hidden_dim_1),
						 # nn.Dropout(p=0.5),
						 # nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
				         # nn.Dropout(p=0.5),
						 # nn.Linear(self.hidden_dim_2, self.hidden_dim_3),
				         # nn.Dropout(p=0.5),
						 nn.Linear(self.hidden_dim_1, 1),
				         # nn.Dropout(p=0.5),
						 nn.Sigmoid()
		                )
		def forward(self, x):
			return self.predict_net(x)