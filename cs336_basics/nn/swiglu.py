import torch
from einops import rearrange, einsum

class SwiGLUNetwork(torch.nn.Module):
	
	def __init__(self, W1, W2, W3):
		super().__init__()
		self.W1 = torch.nn.parameter.Parameter(data=W1)
		self.W2 = torch.nn.parameter.Parameter(data=W2)
		self.W3 = torch.nn.parameter.Parameter(data=W3)

	def forward(self, x):
		w1x = x @ rearrange(self.W1, "a b -> b a")
		silu = w1x*torch.sigmoid(w1x)
		w3x = x @ rearrange(self.W3, "a b -> b a")
		y = silu*w3x
		return y @ rearrange(self.W2, "a b -> b a")