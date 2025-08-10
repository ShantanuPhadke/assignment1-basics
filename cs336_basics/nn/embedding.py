import torch

class Embedding(torch.nn.Module):
	def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
		super().__init__()
		self.embedding_matrix = torch.nn.parameter.Parameter(data=torch.empty(num_embeddings, embedding_dim))
		torch.nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=1.0, a=-3.0, b=3.0)

	def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
		embeddings = []
		for batch_id in range(token_ids.size()[0]):
			current_batch_embeddings = []
			for sequence_id in range(token_ids.size()[1]):
				current_batch_embeddings.append(self.embedding_matrix[token_ids[batch_id][sequence_id].item()])
			embeddings.append(torch.stack(current_batch_embeddings))
		return torch.stack(embeddings)