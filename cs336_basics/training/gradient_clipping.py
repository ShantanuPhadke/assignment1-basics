import torch

def clip_gradient(params, max_value, eps=1e-6):
	print('len(params) = ' + str(len(params)) + ', max_value = ' + str(max_value))
	grad_norms = []
	for p in params:
		l2_norm = 0
		if p.grad is not None:
			l2_norm = torch.norm(p.grad, p=2)
			grad_norms.append(l2_norm)
	
	grad_norms = torch.Tensor(grad_norms)

	l2_grad_norms_norm = torch.norm(grad_norms, p=2)
	if l2_grad_norms_norm >= max_value:
		for p in params:
			if p.grad is not None:
				p.grad*=((max_value)/(l2_grad_norms_norm + eps))
	