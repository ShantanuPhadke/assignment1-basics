import torch

def load_data(x, batch_size, context_length, device='mps'):
	randx = torch.randint(x.size - context_length, (batch_size,))
	batchx = torch.stack([torch.Tensor(x[i:i + context_length]) for i in randx])
	batchy = torch.stack([torch.Tensor(x[i+1:i + context_length+1]) for i in randx])
	return batchx, batchy