import torch, math

class Linear(torch.nn.Module):
	def __init__(self, in_features, out_features, device=None, dtype=None):
		super().__init__()
		self.W = torch.nn.parameter.Parameter(data=torch.empty(out_features, in_features))
		torch.nn.init.trunc_normal_(self.W, mean=0.0, std=math.sqrt(1.0/(out_features + in_features)), a=-3.0, b=3.0)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x @ self.W.T
