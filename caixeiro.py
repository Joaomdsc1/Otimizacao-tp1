import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt
from collections import defaultdict

class TSPSolver:
    def __init__(self, ncities=20, pop_size=100, generations=200, 
                 crossover_rate=0.8, mutation_rate=0.02, elitism_rate=0.1):
        self.ncities = ncities
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        
        # Gerar cidades aleatórias
        self.cities = np.random.rand(ncities, 2) * 100
        self.distance_matrix = self._create_distance_matrix()
        
        # Histórico para análise
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []
    
    def _create_distance_matrix(self):
        """Cria matriz de distâncias entre todas as cidades"""
        matrix = np.zeros((self.ncities, self.ncities))
        for i in range(self.ncities):
            for j in range(i+1, self.ncities):
                dx = self.cities[i,0] - self.cities[j,0]
                dy = self.cities[i,1] - self.cities[j,1]
                matrix[i,j] = matrix[j,i] = sqrt(dx*dx + dy*dy)
        return matrix
    
    def _calculate_distance(self, route):
        """Calcula a distância total de uma rota"""
        total = 0
        for i in range(len(route)-1):
            total += self.distance_matrix[route[i], route[i+1]]
        # Retornar ao ponto inicial
        total += self.distance_matrix[route[-1], route[0]]
        return total
    
    def initialize_population(self):
        """Inicializa população com rotas aleatórias"""
        return [random.sample(range(self.ncities), self.ncities) 
                for _ in range(self.pop_size)]
    
    def evaluate_fitness(self, population):
        """Avalia fitness de toda a população"""
        distances = [self._calculate_distance(ind) for ind in population]
        # Queremos minimizar a distância, então fitness = 1/distância
        return [1/d for d in distances]
    
    def selection(self, population, fitness):
        """Seleção por torneio binário"""
        selected = []
        for _ in range(self.pop_size):
            # Escolhe 2 indivíduos aleatoriamente
            contestants = random.sample(range(self.pop_size), 2)
            # Seleciona o de melhor fitness
            winner = max(contestants, key=lambda x: fitness[x])
            selected.append(population[winner])
        return selected
    
    def crossover(self, parent1, parent2):
        """Crossover OX (Order Crossover)"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        size = len(parent1)
        # Escolhe dois pontos de corte
        a, b = sorted(random.sample(range(size), 2))
        
        # Cria filhos
        child1 = [-1]*size
        child2 = [-1]*size
        
        # Copia o segmento entre a e b
        child1[a:b] = parent1[a:b]
        child2[a:b] = parent2[a:b]
        
        # Preenche os restantes com a ordem do outro pai
        current_pos1 = current_pos2 = b % size
        for i in range(size):
            pos = (b + i) % size
            if parent2[pos] not in child1:
                child1[current_pos1] = parent2[pos]
                current_pos1 = (current_pos1 + 1) % size
            if parent1[pos] not in child2:
                child2[current_pos2] = parent1[pos]
                current_pos2 = (current_pos2 + 1) % size
                
        return child1, child2
    
    def mutation(self, individual):
        """Mutação por inversão de sub-rota"""
        if random.random() < self.mutation_rate:
            a, b = sorted(random.sample(range(len(individual)), 2))
            individual[a:b] = individual[a:b][::-1]
        return individual
    
    def run(self):
        """Executa o algoritmo genético"""
        population = self.initialize_population()
        
        for gen in range(self.generations):
            fitness = self.evaluate_fitness(population)
            
            # Armazena histórico
            self.best_fitness_history.append(max(fitness))
            self.avg_fitness_history.append(np.mean(fitness))
            self.worst_fitness_history.append(min(fitness))
            
            # Seleção
            selected = self.selection(population, fitness)
            
            # Crossover
            new_population = []
            for i in range(0, self.pop_size, 2):
                if i+1 < self.pop_size:
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])
            
            # Mutação
            for i in range(len(new_population)):
                new_population[i] = self.mutation(new_population[i])
            
            # Elitismo
            elite_size = int(self.pop_size * self.elitism_rate)
            elite_indices = np.argsort(fitness)[-elite_size:]
            elite = [population[i] for i in elite_indices]
            
            # Nova população combina elite e novos indivíduos
            population = elite + new_population[:self.pop_size-elite_size]
            
            # Critério de parada opcional
            if gen > 50 and np.std(self.best_fitness_history[-10:]) < 1e-6:
                break
        
        # Retorna o melhor indivíduo
        fitness = self.evaluate_fitness(population)
        best_idx = np.argmax(fitness)
        best_route = population[best_idx]
        best_distance = 1/fitness[best_idx]
        
        return best_route, best_distance
    
    def plot_results(self):
        """Plota os resultados da otimização"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.best_fitness_history, label='Melhor fitness')
        plt.plot(self.avg_fitness_history, label='Fitness médio')
        plt.plot(self.worst_fitness_history, label='Pior fitness')
        plt.xlabel('Geração')
        plt.ylabel('Fitness (1/distância)')
        plt.title('Convergência do Algoritmo Genético')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        best_route, _ = self.run()
        # Fechar o ciclo retornando à primeira cidade
        best_route = best_route + [best_route[0]]
        plt.plot(self.cities[best_route, 0], self.cities[best_route, 1], 'o-')
        plt.scatter(self.cities[:,0], self.cities[:,1], c='red')
        plt.title('Melhor Rota Encontrada')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Execução do desafio
def run_tsp_challenge():
    print("=== Desafio do Caixeiro Viajante ===")
    ncities = int(input("Número de cidades (padrão 20): ") or 20)
    
    # Parâmetros baseados no estudo com Rastrigin
    params = {
        'pop_size': 100,
        'generations': 200,
        'crossover_rate': 0.75,
        'mutation_rate': 0.02,
        'elitism_rate': 0.1
    }
    
    print("\nExecutando 10 simulações com os mesmos parâmetros...")
    results = []
    best_routes = []
    
    for i in range(10):
        solver = TSPSolver(ncities=ncities, **params)
        best_route, best_distance = solver.run()
        results.append(best_distance)
        best_routes.append(best_route)
        print(f"Execução {i+1}: Melhor distância = {best_distance:.2f}")
    
    # Estatísticas
    mean_dist = np.mean(results)
    std_dist = np.std(results)
    
    print("\n=== Resultados Finais ===")
    print(f"Melhor distância encontrada: {min(results):.2f}")
    print(f"Pior distância encontrada: {max(results):.2f}")
    print(f"Média das distâncias: {mean_dist:.2f}")
    print(f"Desvio padrão: {std_dist:.2f}")
    
    # Plot da melhor solução
    best_idx = np.argmin(results)
    solver = TSPSolver(ncities=ncities, **params)
    solver.best_fitness_history = []  # Reset para plotar apenas a última execução
    solver.run()  # Para preencher os históricos
    solver.plot_results()

if __name__ == "__main__":
    run_tsp_challenge()