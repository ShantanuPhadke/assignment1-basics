from collections.abc import Iterator, Iterable

class BPEIterator(Iterator):
	def __init__(self, my_tokenizer, my_iterable: Iterable[str]):
		self.my_tokenizer = my_tokenizer
		self.my_iterable = my_iterable
		self.my_str_iterator = iter(self.my_iterable)
		self.current_encoding = None
		self.current_encoding_index = 0

	def __iter__(self):
		return self

	def __next__(self):
		if not self.current_encoding or self.current_encoding_index == len(self.current_encoding)-1:
			token = next(self.my_str_iterator)
			if not token:
				raise StopIteration("End of iteration reached")
			self.current_encoding = self.my_tokenizer.encode(token)
			self.current_encoding_index = 0
		else:
			self.current_encoding_index+=1
		return self.current_encoding[self.current_encoding_index]


