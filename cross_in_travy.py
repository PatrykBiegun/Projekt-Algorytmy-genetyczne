import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definicja funkcji Cross-in-Tray
def Cross_in_Travy(v):
    x, y = v
    return -0.0001 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - (np.sqrt(x**2 + y**2) / np.pi)))) + 1)**0.1

# Inicjalizacja populacji
def initialize_population(num_individuals, bounds):
    return np.array([np.random.uniform(bounds[0], bounds[1], 2) for _ in range(num_individuals)])

# Obliczanie fitness z karą za znalezione minima
def calculate_fitness_with_penalty(population, sigma_share, found_minima):
    fitness_scores = np.array([Cross_in_Travy(ind) for ind in population])
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
    mutated_children = [child + np.random.normal(0, mutation_strength, 2) if np.random.rand() < mutation_rate else child for child in children]
    return np.array(mutated_children)

# Identyfikacja unikalnych minimów lokalnych
def identify_unique_minima(population, distance_threshold):
    unique_minima = []
    for ind in population:
        if all(np.linalg.norm(ind - unique_ind[0]) >= distance_threshold for unique_ind in unique_minima):
            unique_minima.append((ind, Cross_in_Travy(ind)))
    unique_minima.sort(key=lambda x: x[1])
    return unique_minima

def plot_current_state(ax, population, found_minima, fixed_minima):
    ax.clear()
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    if fixed_minima:
        fixed_population = np.array(fixed_minima)
        ax.scatter(fixed_population[:, 0], fixed_population[:, 1], Cross_in_Travy(fixed_population.T), color='yellow', s=200, edgecolor='black', linewidth=2, label='Found Minima')
    ax.scatter(population[:, 0], population[:, 1], Cross_in_Travy(population.T), color='r')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Function value')
    ax.set_title('3D Visualization of Cross-in-Tray Function Evolution')
    ax.legend()

num_individuals = 50
bounds = (-10, 10)
sigma_share = 1.5
generations = 1000
num_minima = 4
tournament_size = 4

x = np.linspace(bounds[0], bounds[1], 100)
y = np.linspace(bounds[0], bounds[1], 100)
X, Y = np.meshgrid(x, y)
Z = np.array([[Cross_in_Travy([xi, yi]) for xi in x] for yi in y])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
population = initialize_population(num_individuals, bounds)
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
    plot_current_state(ax, population, found_minima, fixed_minima)
    plt.pause(0.01)
    generation += 1

plt.show()

for idx, (individual, fitness) in enumerate(found_minima):
    print(f'Final local minimum {idx + 1}: {individual}, Fitness: {fitness}')
