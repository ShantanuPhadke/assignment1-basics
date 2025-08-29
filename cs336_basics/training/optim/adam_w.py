import torch
from typing import Optional
from collections.abc import Callable, Iterable

class AdamW(torch.optim.Optimizer):
	def __init__(self, params, lr, betas, eps, weight_decay):
		if lr < 0:
			raise ValueError(f"Invalid learning rate: {lr}")
		defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
		super().__init__(params, defaults)

	def step(self, closure: Optional[Callable] = None):
		loss = None if closure is None else closure()
		#print(self.param_groups)
		for group in self.param_groups:
			lr = group["lr"]
			betas = group["betas"]
			eps = group["eps"]
			weight_decay = group["weight_decay"]

		for p in group["params"]:
			if p.grad is None:
				continue

		state = self.state[p] # Get state associated with p.
		m = state.get("m", torch.zeros(p.data.shape))
		v = state.get("v", torch.zeros(p.data.shape))
		t = state.get("t", 1) # Get iteration number from the state, or initial value.
		grad = p.grad.data # Get the gradient of loss with respect to p.
		state["m"] = betas[0]*m + (1-betas[0])*grad
		state["v"] = betas[1]*v + (1-betas[1])*grad**2
		lr_t = lr*((1-betas[1]**t)**0.5/(1-betas[0]**t))
		p.data -= lr_t * (state["m"]/(state["v"]**0.5 + eps))
		p.data -= lr*weight_decay*p.data
		state["t"] = t + 1
		return loss