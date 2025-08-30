import math

def get_cosine_annealing_learning_rate_schedule(iteration, max_learning_rate, min_final_learning_rate, num_warm_up_iterations, num_cosine_annealing_iterations):
	if iteration < num_warm_up_iterations:
		return (iteration/num_warm_up_iterations) * max_learning_rate
	elif num_warm_up_iterations <= iteration <= num_cosine_annealing_iterations:
		cosine_numerator = (iteration-num_warm_up_iterations)*math.pi
		consine_denominator = num_cosine_annealing_iterations-num_warm_up_iterations
		return min_final_learning_rate + 0.5*(1 + math.cos(cosine_numerator/consine_denominator))*(max_learning_rate - min_final_learning_rate)
	else:
		return min_final_learning_rate
