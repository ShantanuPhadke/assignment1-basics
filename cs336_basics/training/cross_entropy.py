import torch

from cs336_basics.nn.softmax import softmax

def cross_entropy(logits, targets):
	logits_softmax = softmax(logits, dimension=0)
	logits_maxes = torch.max(logits, dim=1)
	logits = logits - logits_maxes.values.unsqueeze(1)
	return torch.mean(torch.log(torch.sum(torch.exp(logits), dim=-1)) - torch.gather(logits, -1, targets.unsqueeze(-1)).view(-1))