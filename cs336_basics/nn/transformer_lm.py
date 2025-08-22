import torch

from cs336_basics.nn.embedding import Embedding
from cs336_basics.nn.transformer_block import TransformerBlock
from cs336_basics.nn.rms_norm import RMSProp
from cs336_basics.nn.linear import Linear
from cs336_basics.nn.softmax import softmax

class TransformerLM(torch.nn.Module):
	def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, vocab_size: int, num_layers: int, theta: int=10000):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_ff = d_ff
		self.max_seq_len = max_seq_len
		self.vocab_size = vocab_size
		self.num_layers = num_layers
		self.theta = theta

		self.token_embedding_layer = Embedding(self.vocab_size, self.d_model)
		self.transformer_blocks = []
		for i in range(self.num_layers):
			exec('self.transformer_block_' + str(i) + '=TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.max_seq_len, theta=self.theta)')
			exec('self.transformer_blocks.append(self.transformer_block_' + str(i) + ')')
		self.rms_norm_layer = RMSProp(self.d_model)
		self.linear_layer = Linear(self.d_model, self.vocab_size)

	def forward(self, x):
		y = self.token_embedding_layer(x)
		for i in range(len(self.transformer_blocks)):
			y = self.transformer_blocks[i](y)
		y = self.rms_norm_layer(y)
		y = self.linear_layer(y)
		return y