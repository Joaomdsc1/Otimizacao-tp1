import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
import copy
from math import sqrt
from statistics import mean, stdev

class TSPGeneticAlgorithm:
    def __init__(self, ncities: int, population_size: int = 100, max_generations: int = 500, 
                 crossover_rate: float = 0.8, mutation_rate: float = 0.2, elitism_rate: float = 0.1):

        self.ncities = ncities
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        
        # Gerar cidades aleatórias com coordenadas entre 0 e 100
        self.cities = 100 * np.random.rand(ncities, 2)
        
        # Calcular matriz de distâncias
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.ncities, self.ncities))
        for i in range(self.ncities):
            for j in range(i+1, self.ncities):
                dist = sqrt((self.cities[i,0] - self.cities[j,0])**2 + 
                            (self.cities[i,1] - self.cities[j,1])**2)
                matrix[i,j] = dist
                matrix[j,i] = dist
        return matrix
    
    def _initialize_population(self) -> List[List[int]]:
        population = []
        for _ in range(self.population_size):
            individual = list(range(self.ncities))
            np.random.shuffle(individual)
            population.append(individual)
        return population
    
    def _calculate_fitness(self, individual: List[int]) -> float:
        total_distance = 0
        for i in range(self.ncities - 1):
            total_distance += self.distance_matrix[individual[i], individual[i+1]]
        # Adiciona a distância de volta à cidade inicial
        total_distance += self.distance_matrix[individual[-1], individual[0]]
        return 1 / total_distance  # Queremos maximizar o fitness (inverso da distância)
    
    def _select_parents(self, population: List[List[int]], fitness: List[float]) -> List[List[int]]:
        parents = []
        for _ in range(2):  # Seleciona 2 pais
            tournament = random.sample(list(zip(population, fitness)), k=min(5, self.population_size))
            tournament.sort(key=lambda x: x[1], reverse=True)
            parents.append(tournament[0][0])
        return parents
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        size = len(parent1)
        child1, child2 = [-1]*size, [-1]*size
        
        # Escolhe dois pontos de corte
        start, end = sorted(random.sample(range(size), 2))
        
        # Copia o segmento entre os pontos de corte
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        
        # Preenche os genes restantes
        self._fill_child(child1, parent2, end, start)
        self._fill_child(child2, parent1, end, start)
        return child1, child2
    
    def _fill_child(self, child: List[int], parent: List[int], end: int, start: int):
        size = len(child)
        current_pos = (end + 1) % size
        parent_pos = (end + 1) % size
        while -1 in child:
            if parent[parent_pos] not in child:
                child[current_pos] = parent[parent_pos]
                current_pos = (current_pos + 1) % size
            parent_pos = (parent_pos + 1) % size
    
    def _mutate(self, individual: List[int]) -> List[int]:
        if random.random() > self.mutation_rate:
            return individual.copy()
            
        start, end = sorted(random.sample(range(len(individual)), 2))
        mutated = individual.copy()
        mutated[start:end+1] = reversed(mutated[start:end+1])
        return mutated
    
    def _apply_elitism(self, population: List[List[int]], fitness: List[float], 
                      new_population: List[List[int]]) -> List[List[int]]:
        elite_size = int(self.elitism_rate * self.population_size)
        if elite_size == 0:
            return new_population
            
        combined = list(zip(population, fitness))
        combined.sort(key=lambda x: x[1], reverse=True)
        elite = [x[0] for x in combined[:elite_size]]
        
        # Substitui os piores da nova população pelos elite
        new_population.sort(key=lambda x: self._calculate_fitness(x))
        new_population[-elite_size:] = elite
        
        return new_population
    
    def run(self) -> Tuple[List[int], float, dict]:
        population = self._initialize_population()
        best_individual = None
        best_fitness = -float('inf')
        
        # Para plotar estatísticas
        stats = {
            'best': [],
            'avg': [],
            'worst': []
        }
        
        for generation in range(self.max_generations):
            # Calcula fitness para toda a população
            fitness = [self._calculate_fitness(ind) for ind in population]
            
            # Atualiza estatísticas
            current_best = max(fitness)
            current_avg = mean(fitness)
            current_worst = min(fitness)
            
            stats['best'].append(current_best)
            stats['avg'].append(current_avg)
            stats['worst'].append(current_worst)
            
            # Atualiza o melhor indivíduo global
            if current_best > best_fitness:
                best_fitness = current_best
                best_individual = population[fitness.index(current_best)]
            
            # Cria nova população
            new_population = []
            
            # Aplica elitismo
            elite_size = int(self.elitism_rate * self.population_size)
            elite = []
            if elite_size > 0:
                elite_fitness = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)[:elite_size]
                elite = [x[0] for x in elite_fitness]
            
            while len(new_population) < self.population_size - elite_size:
                # Seleciona pais
                parents = self._select_parents(population, fitness)
                
                # Aplica cruzamento
                child1, child2 = self._crossover(parents[0], parents[1])
                
                # Aplica mutação
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Adiciona elite à nova população
            new_population.extend(elite)
            new_population = new_population[:self.population_size]
            
            population = new_population
            
            # Printa progresso a cada 50 gerações
            if generation % 50 == 0:
                print(f"Geração {generation}: Melhor fitness = {current_best:.6f}, Média = {current_avg:.6f}")
        
        return best_individual, 1/best_fitness, stats
    
    def plot_stats(self, stats: dict):
        plt.figure(figsize=(10, 6))
        plt.plot(stats['best'], label='Melhor fitness')
        plt.plot(stats['avg'], label='Fitness médio')
        plt.plot(stats['worst'], label='Pior fitness')
        plt.xlabel('Geração')
        plt.ylabel('Fitness (1/distância)')
        plt.title('Evolução do Fitness ao Longo das Gerações')
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_solution(self, solution: List[int]):
        plt.figure(figsize=(8, 8))
        # Plota as cidades
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=50)
        
        # Plota o caminho
        tour = self.cities[solution + [solution[0]]]
        plt.plot(tour[:, 0], tour[:, 1], 'b-')
        
        # Adiciona números às cidades
        for i, (x, y) in enumerate(self.cities):
            plt.text(x, y, str(i), fontsize=12)
        
        plt.title(f'Melhor Rota Encontrada - Distância: {1/self._calculate_fitness(solution):.2f}')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.grid()
        plt.show()

# Parâmetros do algoritmo
ncities = 50
population_size = 100
max_generations = 500
crossover_rate = 0.8
mutation_rate = 0.2
elitism_rate = 0.1

# Item a) Implementação do algoritmo genético
print("Implementação do Algoritmo Genético para o Problema do Caixeiro Viajante")
print(f"Número de cidades: {ncities}")
print(f"Tamanho da população: {population_size}")
print(f"Número máximo de gerações: {max_generations}")
print(f"Taxa de cruzamento: {crossover_rate}")
print(f"Taxa de mutação: {mutation_rate}")
print(f"Taxa de elitismo: {elitism_rate}")

# Item b) Executar 10 vezes e calcular estatísticas
print("\nExecutando 10 vezes o algoritmo...")
results = []
best_solutions = []

for run in range(10):
    print(f"\nExecução {run+1}:")
    ga = TSPGeneticAlgorithm(ncities, population_size, max_generations, 
                            crossover_rate, mutation_rate, elitism_rate)
    best_ind, best_dist, stats = ga.run()
    best_solutions.append((best_ind, best_dist))
    results.append(best_dist)
    print(f"Melhor distância encontrada: {best_dist:.2f}")
    
    # Na primeira execução, plotar as estatísticas e a solução
    if run == 0:
        ga.plot_stats(stats)
        ga.plot_solution(best_ind)

# Calcula estatísticas
mean_dist = mean(results)
std_dist = stdev(results) if len(results) > 1 else 0

print("\nResultados das 10 execuções:")
for i, (ind, dist) in enumerate(best_solutions):
    print(f"Execução {i+1}: Distância = {dist:.2f}")

print(f"\nMédia das distâncias: {mean_dist:.2f}")
print(f"Desvio padrão das distâncias: {std_dist:.2f}")
