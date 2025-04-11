import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class GeneticAlgorithm:
    def __init__(self, objective_func, bounds, population_size=50, chromosome_length=20, 
                 max_generations=100, crossover_rate=0.7, mutation_rate=0.02, 
                 elitism_rate=0.6, maximize=False, verbose=True):
        """
        Inicializa o Algoritmo Genético com os parâmetros especificados.
        
        Args:
            objective_func (function): Função objetivo a ser otimizada
            bounds (list): Lista de tuplas com os limites inferior e superior para cada variável
            population_size (int): Tamanho da população
            chromosome_length (int): Comprimento do cromossomo em bits para cada variável
            max_generations (int): Número máximo de gerações
            crossover_rate (float): Taxa de crossover (0-1)
            mutation_rate (float): Taxa de mutação (0-1)
            elitism_rate (float): Taxa de elitismo (0-1)
            maximize (bool): True para maximização, False para minimização
            verbose (bool): Se True, imprime informações durante a execução
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.maximize = maximize
        self.verbose = verbose
        self.dimensions = len(bounds)
        
        # Histórico para análise
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []
        self.execution_time = 0
        
    def initialize_population(self):
        """Inicializa a população com cromossomos aleatórios."""
        return np.random.rand(self.population_size, self.dimensions, self.chromosome_length)
    
    def decode_chromosome(self, chromosome):
        """Converte o cromossomo binário para valores reais dentro dos limites especificados."""
        real_values = []
        for i in range(self.dimensions):
            # Converte para string binária
            binary_str = ''.join(['1' if bit > 0.5 else '0' for bit in chromosome[i]])
            # Converte para inteiro
            integer = int(binary_str, 2)
            # Escala para o intervalo [Xmin, Xmax]
            scaled_value = self.bounds[i][0] + (integer / (2**self.chromosome_length - 1)) * (self.bounds[i][1] - self.bounds[i][0])
            real_values.append(scaled_value)
        return real_values
    
    def evaluate_fitness(self, population):
        """Avalia a fitness da população inteira."""
        fitness = []
        for individual in population:
            decoded = self.decode_chromosome(individual)
            try:
                fitness_value = self.objective_func(decoded)
                # Inverte o sinal se for minimização
                fitness.append(fitness_value if self.maximize else -fitness_value)
            except:
                # Trata valores inválidos com fitness muito ruim
                fitness.append(-np.inf if self.maximize else np.inf)
        return np.array(fitness)
    
    def selection(self, population, fitness):
        """Seleção por torneio binário."""
        selected_indices = []
        for _ in range(int(self.population_size * (1 - self.elitism_rate))):
            # Escolhe dois indivíduos aleatoriamente
            contestants = np.random.choice(len(population), 2, replace=False)
            # Seleciona o de melhor fitness
            winner = contestants[np.argmax(fitness[contestants])]
            selected_indices.append(winner)
        return population[selected_indices]
    
    def crossover(self, parent1, parent2):
        """Realiza crossover de um ponto entre dois pais."""
        if np.random.rand() < self.crossover_rate:
            # Seleciona um ponto de corte aleatório
            crossover_point = np.random.randint(1, self.chromosome_length)
            # Cria os filhos combinando os pais
            child1 = np.concatenate((parent1[:, :crossover_point], parent2[:, crossover_point:]), axis=1)
            child2 = np.concatenate((parent2[:, :crossover_point], parent1[:, crossover_point:]), axis=1)
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutation(self, chromosome):
        """Aplica mutação bit a bit."""
        for i in range(self.dimensions):
            for j in range(self.chromosome_length):
                if np.random.rand() < self.mutation_rate:
                    chromosome[i,j] = 1 - chromosome[i,j]
        return chromosome
    
    def run(self):
        """Executa o algoritmo genético."""
        start_time = time.time()
        population = self.initialize_population()
        
        for generation in range(self.max_generations):
            # Avaliação da fitness
            fitness = self.evaluate_fitness(population)
            best_fitness = np.max(fitness)
            avg_fitness = np.mean(fitness)
            worst_fitness = np.min(fitness)
            
            # Armazena histórico
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.worst_fitness_history.append(worst_fitness)
            
            if self.verbose and generation % 10 == 0:
                print(f"Geração {generation}: Melhor = {best_fitness:.4f}, Médio = {avg_fitness:.4f}, Pior = {worst_fitness:.4f}")
            
            # Critérios de parada
            if generation > 10:
                # 1. Melhoria mínima nas últimas 5 gerações
                if np.std(self.best_fitness_history[-5:]) < 1e-4:
                    if self.verbose:
                        print(f"Parada por convergência (pouca melhoria) na geração {generation}")
                    break
                
                # 2. Fitness atingiu valor alvo (se conhecido)
                if abs(best_fitness) < 1e-3:  # Para problemas de minimização com ótimo em zero
                    if self.verbose:
                        print(f"Parada por atingir valor alvo na geração {generation}")
                    break
            
            # Seleção
            selected = self.selection(population, fitness)
            
            # Crossover
            new_population = []
            for i in range(0, len(selected), 2):
                if i+1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])
            
            # Mutação
            for i in range(len(new_population)):
                new_population[i] = self.mutation(new_population[i])
            
            # Elitismo (mantém os melhores indivíduos)
            elite_size = int(self.population_size * self.elitism_rate)
            elite_indices = np.argsort(fitness)[-elite_size:]
            elite = population[elite_indices]
            
            # Nova população combina elite e novos indivíduos
            population = np.concatenate((elite, np.array(new_population)[:self.population_size-elite_size]))
        
        # Retorna o melhor indivíduo encontrado
        fitness = self.evaluate_fitness(population)
        best_idx = np.argmax(fitness)
        best_individual = population[best_idx]
        best_solution = self.decode_chromosome(best_individual)
        best_fitness = self.objective_func(best_solution)
        
        self.execution_time = time.time() - start_time
        
        if self.verbose:
            print("\n--- Resultados Finais ---")
            print(f"Melhor solução encontrada: {best_solution}")
            print(f"Valor da função objetivo: {best_fitness}")
            print(f"Tempo de execução: {self.execution_time:.2f} segundos")
            print(f"Total de gerações: {len(self.best_fitness_history)}")
        
        return best_solution, best_fitness

# Funções objetivo pré-definidas
def peaks_function(x):
    """Função Peaks para maximização. Domínio recomendado: x,y ∈ [-3, 3]"""
    x, y = x[0], x[1]
    return 3*(1-x)**2 * np.exp(-x**2 - (y+1)**2) - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) - 1/3 * np.exp(-(x+1)**2 - y**2)

def ackley_function(x):
    """Função Ackley para minimização. Domínio recomendado: x,y ∈ [-35, 35]"""
    x, y = x[0], x[1]
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20

def rastrigin_function(x):
    """Função Rastrigin para minimização. Domínio recomendado: x,y ∈ [-5.12, 5.12]"""
    x, y = x[0], x[1]
    A = 10
    return A*2 + (x**2 - A*np.cos(2*np.pi*x)) + (y**2 - A*np.cos(2*np.pi*y))

def sphere_function(x):
    """Função Sphere (esfera) para minimização. Domínio recomendado: x,y ∈ [-5, 5]"""
    return sum(xi**2 for xi in x)

# Interface do usuário
def run_optimization():
    """Função principal que gerencia a interação com o usuário."""
    print("=== Ferramenta de Otimização por Algoritmo Genético ===")
    print("\nSelecione a função objetivo:")
    print("1 - Função Peaks (maximização)")
    print("2 - Função Ackley (minimização)")
    print("3 - Função Rastrigin (minimização)")
    print("4 - Função Sphere (minimização)")
    print("5 - Definir função personalizada")
    
    try:
        choice = int(input("Opção (1-5): "))
    except:
        print("Opção inválida. Usando Função Rastrigin como padrão.")
        choice = 3
    
    if choice == 1:
        func = peaks_function
        bounds = [(-3, 3), (-3, 3)]
        maximize = True
        print("\nFunção Peaks selecionada (maximização no intervalo [-3, 3])")
    elif choice == 2:
        func = ackley_function
        bounds = [(-35, 35), (-35, 35)]
        maximize = False
        print("\nFunção Ackley selecionada (minimização no intervalo [-35, 35])")
    elif choice == 3:
        func = rastrigin_function
        bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        maximize = False
        print("\nFunção Rastrigin selecionada (minimização no intervalo [-5.12, 5.12])")
    elif choice == 4:
        func = sphere_function
        bounds = [(-5, 5), (-5, 5)]
        maximize = False
        print("\nFunção Sphere selecionada (minimização no intervalo [-5, 5])")
    elif choice == 5:
        print("\nDefina sua função personalizada.")
        print("Exemplo: Para f(x,y) = x^2 + y^2, digite: x[0]**2 + x[1]**2")
        func_str = input("Digite a expressão da função (use x[0], x[1], etc. para as variáveis): ")
        
        try:
            # Cria uma função lambda a partir da string fornecida
            func = lambda x: eval(func_str)
            # Testa a função
            test_input = [0]*2
            func(test_input)
            
            print("\nDefina os limites para cada variável:")
            bounds = []
            for i in range(2):  # Assumindo 2 dimensões por padrão
                xmin = float(input(f"Limite inferior para x[{i}]: "))
                xmax = float(input(f"Limite superior para x[{i}]: "))
                bounds.append((xmin, xmax))
            
            maximize_input = input("Maximizar a função? (s/n): ").lower()
            maximize = maximize_input == 's'
            
            print("\nFunção personalizada configurada com sucesso!")
        except:
            print("\nErro na definição da função. Usando Função Rastrigin como padrão.")
            func = rastrigin_function
            bounds = [(-5.12, 5.12), (-5.12, 5.12)]
            maximize = False
    else:
        print("Opção inválida. Usando Função Rastrigin como padrão.")
        func = rastrigin_function
        bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        maximize = False
    
    print("\nConfiguração dos parâmetros do Algoritmo Genético:")
    print("(Pressione Enter para usar os valores padrão)")
    
    try:
        pop_size = int(input(f"Tamanho da população (10-100) [50]: ") or 50)
        chrom_len = int(input(f"Tamanho do cromossomo (10-35) [20]: ") or 20)
        max_gen = int(input(f"Número máximo de gerações (10-50) [30]: ") or 30)
        crossover = float(input(f"Taxa de cruzamento (0.6-0.8) [0.7]: ") or 0.7)
        mutation = float(input(f"Taxa de mutação (0.01-0.05) [0.02]: ") or 0.02)
        elitism = float(input(f"Taxa de elitismo (0.55-0.75) [0.6]: ") or 0.6)
    except:
        print("Valores inválidos. Usando configurações padrão.")
        pop_size = 50
        chrom_len = 20
        max_gen = 30
        crossover = 0.7
        mutation = 0.02
        elitism = 0.6
    
    # Cria e executa o algoritmo genético
    ga = GeneticAlgorithm(func, bounds, pop_size, chrom_len, max_gen, 
                         crossover, mutation, elitism, maximize)
    
    solution, fitness = ga.run()
    
    # Plotagem dos resultados
    plt.figure(figsize=(15, 5))
    
    # Gráfico de convergência
    plt.subplot(1, 2, 1)
    plt.plot(ga.best_fitness_history, 'b-', label='Melhor fitness')
    plt.plot(ga.avg_fitness_history, 'g-', label='Fitness médio')
    plt.plot(ga.worst_fitness_history, 'r-', label='Pior fitness')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Convergência do Algoritmo Genético')
    plt.legend()
    plt.grid(True)
    
    # Plotagem da função objetivo (para 2D)
    if len(bounds) == 2:
        plt.subplot(1, 2, 2)
        x = np.linspace(bounds[0][0], bounds[0][1], 100)
        y = np.linspace(bounds[1][0], bounds[1][1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = func([X[i,j], Y[i,j]])
        
        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar()
        plt.scatter(solution[0], solution[1], c='red', marker='x', s=100, linewidths=2, label='Solução')
        plt.title('Função Objetivo e Solução Encontrada')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Função para estudo paramétrico
def parameter_study():
    """Realiza um estudo sistemático dos parâmetros do AG."""
    print("\n=== Estudo Paramétrico do Algoritmo Genético ===")
    print("Este estudo usará a função Rastrigin como caso de teste.")
    print("Cada parâmetro será variado enquanto os outros permanecem fixos.\n")
    
    # Configuração base
    base_params = {
        'population_size': 50,
        'chromosome_length': 20,
        'max_generations': 30,
        'crossover_rate': 0.7,
        'mutation_rate': 0.02,
        'elitism_rate': 0.6
    }
    
    # Variação de cada parâmetro
    param_ranges = {
        'population_size': range(10, 101, 10),
        'chromosome_length': range(10, 36, 5),
        'max_generations': range(10, 51, 10),
        'crossover_rate': np.linspace(0.6, 0.8, 5),
        'mutation_rate': np.linspace(0.01, 0.05, 5),
        'elitism_rate': np.linspace(0.55, 0.75, 5)
    }
    
    results = {}
    
    for param_name, param_range in param_ranges.items():
        print(f"\nEstudando variação do parâmetro: {param_name}")
        param_results = []
        
        for value in param_range:
            # Cria cópia dos parâmetros base e atualiza o parâmetro atual
            params = base_params.copy()
            params[param_name] = value
            
            # Executa o GA com configuração atual
            ga = GeneticAlgorithm(rastrigin_function, [(-5.12, 5.12), (-5.12, 5.12)], 
                                 **params, maximize=False, verbose=False)
            solution, fitness = ga.run()
            
            # Armazena o melhor fitness encontrado e o tempo de execução
            best_fitness = min(ga.best_fitness_history)
            param_results.append({
                'value': value,
                'best_fitness': best_fitness,
                'execution_time': ga.execution_time,
                'generations': len(ga.best_fitness_history)
            })
            
            print(f"{param_name}={value:.3f} -> Fitness: {best_fitness:.4f}, Tempo: {ga.execution_time:.2f}s, Gerações: {len(ga.best_fitness_history)}")
        
        results[param_name] = param_results
    
    # Plotagem dos resultados
    plt.figure(figsize=(15, 10))
    for i, (param_name, param_results) in enumerate(results.items()):
        plt.subplot(2, 3, i+1)
        x = [r['value'] for r in param_results]
        y = [r['best_fitness'] for r in param_results]
        plt.plot(x, y, 'o-')
        plt.xlabel(param_name)
        plt.ylabel('Melhor Fitness')
        plt.title(f"Variação de {param_name}")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Exibe tabela resumo
    print("\n=== Resumo do Estudo Paramétrico ===")
    print("Parâmetro\tMelhor Valor\tMelhor Fitness")
    for param_name, param_results in results.items():
        best_result = min(param_results, key=lambda x: x['best_fitness'])
        print(f"{param_name}\t{best_result['value']:.3f}\t\t{best_result['best_fitness']:.4f}")

# Menu principal
def main():
    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("1 - Executar otimização")
        print("2 - Realizar estudo paramétrico")
        print("3 - Sair")
        
        try:
            option = int(input("Opção: "))
        except:
            option = 0
        
        if option == 1:
            run_optimization()
        elif option == 2:
            parameter_study()
        elif option == 3:
            print("Encerrando o programa...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
