from collections.abc import Iterator, Iterable
import time

class BPEIterator(Iterator):
	def __init__(self, my_tokenizer, my_iterable: Iterable[str], debug=False):
		self.my_tokenizer = my_tokenizer
		self.my_iterable = my_iterable
		self.my_str_iterator = iter(self.my_iterable)
		self.current_encoding = None
		self.current_encoding_index = 0
		self.tokens_processed = -1
		self.current_start_time = time.time()
		self.current_time_taken = 0
		self.debug = debug

	def __iter__(self):
		return self

	def __next__(self):
		if not self.current_encoding or self.current_encoding_index == len(self.current_encoding)-1:
			self.tokens_processed+=1
			if self.debug and self.tokens_processed % 10000 == 0 and self.tokens_processed > 0:
				self.current_time_taken = time.time() - self.current_start_time
				self.current_start_time = time.time()
				print('Processed ' + str(self.tokens_processed) + ' tokens! Last 10000 tokens processed in ' + str(self.current_time_taken) + ' seconds!')
			token = next(self.my_str_iterator)
			if not token:
				raise StopIteration("End of iteration reached")
			self.current_encoding = self.my_tokenizer.encode(token)
			self.current_encoding_index = 0
		else:
			self.current_encoding_index+=1
		return self.current_encoding[self.current_encoding_index]


