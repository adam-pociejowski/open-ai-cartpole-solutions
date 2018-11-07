import gym
import numpy as np
import random

env = gym.make('CartPole-v1')
render = False
number_of_observations = env.observation_space.shape[0]
number_of_actions = env.action_space.n
num_of_generations = 20
episode_max_length = 1000

# Genetic algorithm parameters
mutation_rate = 0.01
max_mutation_value_change = 0.1
number_of_chromosomes = 50
number_of_elite_chromosomes = 6

# Neural network parameters
input_layer_nodes = number_of_observations + 1
hidden_layer_nodes = 4
output_layer_nodes = 2


def relu(x):
	return np.maximum(0,x)


def predict_using_neural_network(observation, chromosome):
	input_values = observation / max(np.max(np.linalg.norm(observation)), 1)
	input_values = np.insert(1.0, 1, input_values)
	hidden_layer_values = relu(np.dot(input_values, chromosome[0]))
	output_layer_values = relu(np.dot(hidden_layer_values, chromosome[1]))
	result = np.argmax(output_layer_values)
	return result


def prepare_random_population(number_of_chromosomes):
	population = []
	for i in range(number_of_chromosomes):
		hidden_layer_weights = np.random.rand(input_layer_nodes, hidden_layer_nodes)
		output_layer_weights = np.random.rand(hidden_layer_nodes, output_layer_nodes)
		population.append([hidden_layer_weights, output_layer_weights])
	return population


def run_episode_for_chromosome(chromosome):
	observation = env.reset()
	total_reward = 0
	for t in range(episode_max_length):
		if render:
			env.render()
		action = predict_using_neural_network(observation, chromosome)
		observation, reward, done, info = env.step(action)
		total_reward += reward
		if done:
			break
	return total_reward


def run_episode_for_population(population):
	rewards = []
	for index in range(len(population)):
		reward = run_episode_for_chromosome(population[index])
		rewards.append(reward)
	return rewards


def change_to_flatten_list(chromosome):
	input_layer = chromosome[0]
	input_layer = input_layer.reshape(input_layer.shape[1], -1)
	hidden_layer = chromosome[1]
	return np.append(input_layer, hidden_layer.reshape(hidden_layer.shape[1], -1))


def mutation(chromosome):
	random_value = np.random.randint(0, len(chromosome))
	if random_value < mutation_rate:
		n = np.random.randint(0, len(chromosome))
		chromosome[n] += np.random.rand()*max_mutation_value_change
	return chromosome


def crossover(best_chromosomes):
	new_population = best_chromosomes
	for index in range(number_of_chromosomes - number_of_elite_chromosomes):
		parents = random.sample(range(number_of_elite_chromosomes), 2)
		cut_point = random.randint(0, len(best_chromosomes[0]))
		new_chromosome = np.append(best_chromosomes[parents[0]][:cut_point], best_chromosomes[parents[1]][cut_point:])
		new_chromosome = mutation(new_chromosome)
		new_population.append(new_chromosome)
	return new_population


def generate_next_population(population, rewards):
	best_chromosomes_indexes = np.asarray(rewards).argsort()[-number_of_elite_chromosomes:][::-1]
	best_chromosomes_list = []
	for index in best_chromosomes_indexes:
		chromosome_flatten = change_to_flatten_list(population[index])
		best_chromosomes_list.append(chromosome_flatten)

	new_population_flatten = crossover(best_chromosomes_list)
	new_population = []
	for chromosome_flatten in new_population_flatten:
		input_layer_flatten = np.array(chromosome_flatten[:hidden_layer_nodes * input_layer_nodes])
		input_layer_reshaped = np.reshape(input_layer_flatten, (-1, population[0][0].shape[1]))
		hidden_layer_flatten = np.array(chromosome_flatten[hidden_layer_nodes * input_layer_nodes:])
		hidden_layer_reshaped = np.reshape(hidden_layer_flatten, (-1, population[0][1].shape[1]))
		new_population.append([input_layer_reshaped, hidden_layer_reshaped])

	return new_population


if __name__ == '__main__':
	population = prepare_random_population(number_of_chromosomes=number_of_chromosomes)
	rewards = run_episode_for_population(population)
	for generation in range(num_of_generations):
		population = generate_next_population(population, rewards)
		rewards = run_episode_for_population(population)
		best = np.amax(rewards)
		avg = np.average(rewards)
		print("Generation: {}, Avg: {}, Best: {}".format(generation + 1, avg, best))

	env.close()
