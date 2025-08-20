import torch
from einops import einsum
from .softmax import softmax

def scaled_dot_product_attention(queries, keys, values, mask=None):
	pre_softmax_scores = (queries @ torch.transpose(keys, -2, -1))/torch.Tensor([queries.shape[-1]**0.5])
	if mask != None:
		pre_softmax_scores = torch.where(mask == False, pre_softmax_scores+float('-inf'), pre_softmax_scores)
	softmax_scores = softmax(pre_softmax_scores, -1)
	final_attention_outputs = torch.nan_to_num(softmax_scores) @ values
	return final_attention_outputs