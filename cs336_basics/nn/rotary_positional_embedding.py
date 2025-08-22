import torch
import math
import numpy as np
from einops import rearrange

class RotaryPositionalEmbedding(torch.nn.Module):
	def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
		super(RotaryPositionalEmbedding, self).__init__()
		print('d_k = ' + str(d_k) + ', max_seq_len = ' + str(max_seq_len)) 
		self.d_k = d_k
		self.max_seq_len = max_seq_len

		half_dim = d_k // 2
		inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
		# positions: [max_seq_len]
		positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
		# outer product
		freqs = torch.einsum("i,j->ij", positions, inv_freq)

		# Precompute cos/sin [max_seq_len, half_dim]
		cos = freqs.cos()
		sin = freqs.sin()

		# Register as buffers
		self.register_buffer("cosine_values", cos, persistent=False)
		self.register_buffer("sine_values", sin, persistent=False)


	def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
		x1 = x[...,::2]
		x2 = x[...,1::2]
		print('x1.shape = ' + str(x1.shape))
		print('x2.shape = ' + str(x2.shape))
		print('self.cosine_values.shape = ' + str(self.cosine_values[token_positions].shape))
		print('self.sine_values.shape = ' + str(self.sine_values[token_positions].shape))
		x_rotated = torch.stack([x1 * self.cosine_values[token_positions] - x2 * self.sine_values[token_positions],
								 x1 * self.sine_values[token_positions] + x2 * self.cosine_values[token_positions]], dim=3)
		return x_rotated.flatten(2,3)
		