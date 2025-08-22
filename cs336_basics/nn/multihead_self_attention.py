import torch
from .scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.nn.rotary_positional_embedding import RotaryPositionalEmbedding

class MultiHeadSelfAttention(torch.nn.Module):
	def __init__(self, d_model, num_heads, with_rope=False, theta=None, max_seq_len=None, token_positions=None):
		super().__init__()
		self.d_k = d_model // num_heads
		self.d_v = d_model // num_heads
		self.d_model = d_model
		self.num_heads = num_heads

		self.W_Q = torch.nn.parameter.Parameter(data=torch.empty(num_heads*self.d_k, d_model))
		self.W_K = torch.nn.parameter.Parameter(data=torch.empty(num_heads*self.d_k, d_model))
		self.W_V = torch.nn.parameter.Parameter(data=torch.empty(num_heads*self.d_v, d_model))
		self.W_O = torch.nn.parameter.Parameter(data=torch.empty(d_model, num_heads*self.d_v))

		self.with_rope = with_rope
		if self.with_rope:
			self.theta = theta
			self.max_seq_len = max_seq_len
			self.token_positions = token_positions
			self.rope_layer = RotaryPositionalEmbedding(self.theta, self.d_model//self.num_heads, self.max_seq_len)

	def forward(self, x):
		b, msl, _ = x.shape
		Q = x @ torch.transpose(self.W_Q, -1, -2)
		K = x @ torch.transpose(self.W_K, -1, -2)
		V = x @ torch.transpose(self.W_V, -1, -2)

		if self.with_rope:
			Q_modified = torch.stack(torch.split(Q, self.d_model//self.num_heads, dim=-1), dim=1)
			for i in range(Q_modified.shape[0]):
				Q_modified[i] = self.rope_layer(Q_modified[i,...], self.token_positions)
			Q = Q_modified.transpose(1, 2).flatten(-2, -1)
			K_modified = torch.stack(torch.split(K, self.d_model//self.num_heads, dim=-1), dim=1)
			for i in range(K_modified.shape[0]):
				K_modified[i] = self.rope_layer(K_modified[i,...], self.token_positions)
			K = K_modified.transpose(1,2).flatten(-2, -1)


		attentions = []
		attention_mask = torch.triu(torch.ones(b, msl, msl), diagonal=1) == 0
		for i in range(self.num_heads):
			Q_i = Q[:,:,i*self.d_k:(i+1)*self.d_k]
			K_i = K[:,:,i*self.d_k:(i+1)*self.d_k]
			V_i = V[:,:,i*self.d_v:(i+1)*self.d_v]
			attentions.append(scaled_dot_product_attention(Q_i, K_i, V_i, mask=attention_mask))
		multi_head_attention = torch.concat(attentions, dim=-1)
		return multi_head_attention @ torch.transpose(self.W_O, -1, -2)

