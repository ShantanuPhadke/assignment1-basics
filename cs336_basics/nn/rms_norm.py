import torch

class RMSProp(torch.nn.Module):
	def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
		super().__init__()
		self.d_model = d_model
		self.eps = eps
		self.device = device
		self.dtype = dtype
		self.W = torch.nn.parameter.Parameter(data=torch.randn(d_model))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		in_dtype = x.dtype
		x = x.to(torch.float32)
		rms = torch.sqrt((1.0/self.d_model)*torch.sum(torch.square(x), 2) + self.eps)
		result = torch.Tensor(torch.mul(torch.div(x, rms.unsqueeze(2)), self.W))
		return result.to(in_dtype)