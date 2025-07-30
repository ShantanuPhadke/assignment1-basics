import time

class CodeProfiler:
	def __init__(self):
		self.history = [] # History of the profiling for this code
		self.current_start_time = None
		self.current_end_time = None

	def start_new_profiler(self, name: str):
		if not self.current_start_time:
			self.current_start_time = time.time()
		self.history.append({'step_name': name})

	def log_profiler(self):
		if not self.current_end_time and self.current_start_time:
			self.current_end_time = time.time()
		self.history[len(self.history)-1]['running_time'] = self.current_end_time - self.current_start_time
		self.current_start_time = None
		self.current_end_time = None

	def __str__(self):
		return '\n'.join(map(lambda entry: "Amount of time taken for step " + entry['step_name'] + ": " + str(entry['running_time']) + " seconds", self.history))
