import torch

def save_checkpoint(model, optimizer, iteration, out):
	# Save model weights, optimizer information and iteration information
	checkpoint_obj = {
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'iteration': iteration
	}
	torch.save(checkpoint_obj, out)

def load_checkpoint(src, model, optimizer):
	checkpoint_obj = torch.load(src)
	model.load_state_dict(checkpoint_obj['model'])
	optimizer.load_state_dict(checkpoint_obj['optimizer'])
	return checkpoint_obj['iteration']