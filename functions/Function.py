from math import floor
from algorithms.DE import DE
from algorithms.GA import GA
import numpy as np

class Function:
    def __init__(self, dimension, bounds):
        print('Optimization Function : ', end='')

        self.dimension = dimension
        self.bounds = bounds

        # For Binary Encoded GA
        self.var_len = 30
        self.soln_len = self.dimension * self.var_len



    # Get Fitness
    def getFitness(self, X):
        f = self.eval(X)
        res = 99999 if f == 0 else 1/f
        return res
    
    # Optimize Using Genetic Algorithms
    def optimizeUsingGA(self, population_size=100, iterations=1000):
        print('Running genetic algorithm...')
        sample_size = population_size
        n_generations = iterations

        # Probabilities
        prob_crossover=0.5
        prob_mutation=0.05

        # Genetic Algorithm Object
        ga = GA(self, self.dimension, sample_size, self.bounds, soln_len=self.soln_len, var_len=self.var_len, prob_crossover=prob_crossover, prob_mutation=prob_mutation)

        # Initialize Population
        ga.initializePopulation()
        # print(ga.population)

        # Generations
        for _ in range(n_generations):
            # Calcualte fitness
            ga.calculateFitnesses()
            # Calculate probability of selection
            ga.calculateProbabilitiesRoulette()
            # Generate mating pool
            mating_pool = ga.rouletteWheelSelection()
            # Apply Crossover
            children = ga.crossover(mating_pool)
            # Mutate the children
            mutated_children = ga.mutate(children)
            # Select survivors
            new_population = ga.selectEliteSurvivor(mutated_children)
            # Generation Update
            ga.updateGeneration(new_population)

        X_res = ga.getResult()
        opt_value = self.eval(X_res)
        return X_res, opt_value

    # Optimize Using Differential Evolution
    def optimizeUsingDE(self, population_size=100, iterations=1000):
        print('Running differential evolution...')

        # Parameters
        prob_recombination = 0.7
        beta = 0.5
        n_diff_vectors = 1
        population_size = max(population_size, 2 * n_diff_vectors + 1)
        sample_size = population_size
        n_generations = iterations
        gamma = 1
        # gamma_step = 1 / n_generations

        # DE Object
        de = DE(self, self.dimension, sample_size, self.bounds, beta, prob_recombination)
        de.initializePopulation()

        for _ in range(n_generations):
            new_population = []
            de.calculateFitnesses()
            for i in range(sample_size):
                # Get parent
                parent = de.population[i]
                # Evaluate fitness of the parent
                fitness_parent = de.fitnesses[i]
                # Generate trial vector by mutating the parent
                trial_vector = de.mutate(i, n_diff_vectors, gamma)
                # Generate a child by crossover
                child = de.binomialCrossover(parent, trial_vector)
                # Evaluate the fitness of the child
                fitness_child = self.getFitness(child)
                # Select the individual with better fitness for next generation
                if(fitness_child > fitness_parent):
                    new_population.append(child)
                else:
                    new_population.append(parent)

            de.updatePopulation(new_population)
            # gamma = gamma + gamma_step
        
        X_res = de.getResult()
        opt_value = self.eval(X_res)
        return X_res, opt_value
