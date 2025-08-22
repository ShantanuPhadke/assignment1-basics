import torch

from cs336_basics.nn.rms_norm import RMSProp
from cs336_basics.nn.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.nn.swiglu import SwiGLUNetwork

class TransformerBlock(torch.nn.Module):
	def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: int=10000):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_ff = d_ff
		self.max_seq_len = max_seq_len
		self.theta = theta
		self.rms_norm_layer1 = RMSProp(self.d_model)
		self.rms_norm_layer2 = RMSProp(self.d_model)
		self.mha_rope_layer = MultiHeadSelfAttention(self.d_model, self.num_heads, with_rope=True, theta=self.theta, max_seq_len=max_seq_len, token_positions=[i for i in range(max_seq_len)])
		W1 = torch.rand(self.d_ff, self.d_model)
		W2 = torch.rand(self.d_model, self.d_ff)
		W3 = torch.rand(self.d_ff, self.d_model)
		self.swiglu_network = SwiGLUNetwork(W1, W2, W3)

	def forward(self, x):
		self.mha_rope_layer.token_positions = [i for i in range(x.shape[1])]
		y = x + self.mha_rope_layer(self.rms_norm_layer1(x))
		return y + self.swiglu_network(self.rms_norm_layer2(y))