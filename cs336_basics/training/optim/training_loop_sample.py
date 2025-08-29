import torch

from cs336_basics.training.sgd_sample import SGD

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
learning_rates = [1e1, 1e2, 1e3]
num_iterations = 10

for learning_rate in learning_rates:
	opt = SGD([weights], lr=learning_rate)
	print(f'Training Loss over time for Learning Rate = {learning_rate}')
	print('-'*100)
	for t in range(num_iterations):
		opt.zero_grad() # Reset the gradients for all learnable parameters.
		loss = (weights**2).mean() # Compute a scalar loss value.
		print(f'Step {t}: ' + str(loss.cpu().item()))
		loss.backward() # Run backward pass, which computes gradients.
		opt.step() # Run optimizer step.
	print('-'*100)
	print()