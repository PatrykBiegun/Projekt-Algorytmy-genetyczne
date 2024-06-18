import numpy as np
import matplotlib.pyplot as plt

# Definicja funkcji Himmelblaua w 10 wymiarach
def Himmelblau(v):
    return sum((v[2*i]**2 + v[2*i+1] - 11)**2 + (v[2*i] + v[2*i+1]**2 - 7)**2 for i in range(len(v)//2))

# Inicjalizacja populacji
def initialize_population(num_individuals, bounds, dimensions):
    return np.array([np.random.uniform(bounds[0], bounds[1], dimensions) for _ in range(num_individuals)])

# Obliczanie fitness z karą za znalezione minima
def calculate_fitness_with_penalty(population, sigma_share, found_minima):
    fitness_scores = np.array([Himmelblau(ind) for ind in population])
    adjusted_fitness = np.zeros_like(fitness_scores)
    for i, ind_i in enumerate(population):
        penalty = sum(np.exp(-np.linalg.norm(ind_i - minimum[0]) / sigma_share) for minimum in found_minima)
        adjusted_fitness[i] = fitness_scores[i] + penalty
    return adjusted_fitness

# Selektor turniejowy z niszowaniem
def tournament_selection(population, fitness, tournament_size):
    selected = []
    for _ in range(len(population)):
        participants = np.random.choice(range(len(population)), tournament_size, replace=False)
        winner = participants[np.argmin(fitness[participants])]
        selected.append(population[winner])
    return np.array(selected)

# Krzyżowanie i mutacja
def crossover_and_mutate(selected, bounds, mutation_rate=0.1, mutation_strength=0.05):
    children = []
    for i in range(0, len(selected), 2):
        if i + 1 < len(selected):
            cross_point = np.random.rand()
            child1 = np.clip(selected[i] * cross_point + selected[i + 1] * (1 - cross_point), bounds[0], bounds[1])
            child2 = np.clip(selected[i + 1] * cross_point + selected[i] * (1 - cross_point), bounds[0], bounds[1])
            children.extend([child1, child2])
    mutated_children = [child + np.random.normal(0, mutation_strength, len(selected[0])) if np.random.rand() < mutation_rate else child for child in children]
    return np.array(mutated_children)

# Identyfikacja unikalnych minimów lokalnych
def identify_unique_minima(population, distance_threshold):
    unique_minima = []
    for ind in population:
        if all(np.linalg.norm(ind - unique_ind[0]) >= distance_threshold for unique_ind in unique_minima):
            unique_minima.append((ind, Himmelblau(ind)))
    unique_minima.sort(key=lambda x: x[1])
    return unique_minima

num_individuals = 1000
bounds = (-5, 5)
dimensions = 10  # Zwiększenie do 10 wymiarów
sigma_share = 1.5
generations = 1000
num_minima = 4
tournament_size = 4

# Inicjalizacja populacji
population = initialize_population(num_individuals, bounds, dimensions)
found_minima = []
fixed_minima = []

generation = 0
while len(found_minima) < num_minima and generation < generations:
    fitness_scores = calculate_fitness_with_penalty(population, sigma_share, [fm[0] for fm in found_minima])
    selected = tournament_selection(population, fitness_scores, tournament_size)
    population = crossover_and_mutate(selected, bounds)
    new_minima = identify_unique_minima(population, sigma_share * 2)
    for individual, fitness in new_minima:
        if len(found_minima) < num_minima and all(np.linalg.norm(individual - fm[0]) >= sigma_share for fm in found_minima):
            found_minima.append((individual, fitness))
            fixed_minima.append(individual)
            print(f'Found and fixed local minimum: {individual}, Fitness: {fitness}')
    generation += 1

for idx, (individual, fitness) in enumerate(found_minima):
    print(f'Final local minimum {idx + 1}: {individual}, Fitness: {fitness}')
