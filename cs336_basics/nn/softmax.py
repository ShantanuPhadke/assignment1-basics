import torch

def softmax(tensor: torch.Tensor, dimension: int):
	# dimension = 0 -> alll rows should be probability distrubtions, dimension = 1 -> all columns, etc...
	values_max_subtracted = tensor - torch.unsqueeze(torch.max(tensor, dim=dimension).values, dimension)
	softmax_dim_sums = torch.sum(torch.exp(values_max_subtracted), dim=dimension)
	dim_softmaxes = torch.exp(values_max_subtracted)/torch.unsqueeze(softmax_dim_sums, dimension)
	return dim_softmaxes

