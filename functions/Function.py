from math import floor
from algorithms.ABC import ABC
from algorithms.DE import DE
from algorithms.GA import GA
import numpy as np

from algorithms.PSO import PSO


class Function:
    def __init__(self, dimension, bounds):
        print("Optimization Function : ", end="")

        self.dimension = dimension
        self.bounds = bounds

        # For Binary Encoded GA
        self.var_len = 30
        self.soln_len = self.dimension * self.var_len

    # Get Fitness
    def getFitness(self, X):
        f = self.eval(X)
        return 1 / (1 + f) if f >= 0 else 1 + abs(f)

    # Optimize Using Genetic Algorithms
    def optimizeUsingGA(self, population_size=100, iterations=1000):
        print("Running genetic algorithm...")
        sample_size = population_size
        n_generations = iterations

        # Probabilities
        prob_crossover = 0.5
        prob_mutation = 0.05

        # Genetic Algorithm Object
        ga = GA(
            self,
            self.dimension,
            sample_size,
            self.bounds,
            soln_len=self.soln_len,
            var_len=self.var_len,
            prob_crossover=prob_crossover,
            prob_mutation=prob_mutation,
        )

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
        print("Running differential evolution...")

        # Parameters
        prob_recombination = 0.7
        beta = 0.5
        n_diff_vectors = 1
        population_size = max(population_size, 2 * n_diff_vectors + 1)
        sample_size = population_size
        n_generations = iterations
        gamma = 1
        gamma_step = 1 / n_generations

        # DE Object
        de = DE(
            self, self.dimension, sample_size, self.bounds, beta, prob_recombination
        )
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
                if fitness_child > fitness_parent:
                    new_population.append(child)
                else:
                    new_population.append(parent)

            de.updatePopulation(new_population)
            # gamma = gamma + gamma_step

        X_res = de.getResult()
        opt_value = self.eval(X_res)
        return X_res, opt_value

    # Optimize using Artificial Bee Colony
    def optimizeUsingABC(self, swarm_size=100, iterations=1000):
        print("Running artificial bee colony...")

        # Parameters
        colony_size = swarm_size
        n_iterations = iterations

        # Initialize the ABC Object
        abc = ABC(self, self.dimension, colony_size, self.bounds)

        # Initialize (Move the scouts)
        abc.initializeFoodSources()
        abc.calculateFitnesses()
        abc.resetTrials()

        # Memorize the best solution
        best_fitness, best_soln = abc.getCurrentBest()

        for _ in range(n_iterations):
            # Employeed Bee Phase
            abc.performEmployedBeePhase()
            # Generate prob of selection of each solution before onlooker bee phase
            prob_selection = abc.generateProbabilities()
            # Onlooker Bee Phase
            abc.performOnlookerBeePhase(prob_selection)
            # Memorize the best soln
            temp_fitness, temp_soln = abc.getCurrentBest()
            if temp_fitness > best_fitness:
                best_fitness = temp_fitness
                best_soln = temp_soln
            # Scout Bee Phase
            abc.performScoutBeePhase()

        opt_val = self.eval(best_soln)
        return best_soln, opt_val

    # Optimize using Particle Swarm Optimization
    def optimizeUsingPSO(self, population_size=100, iterations=1000):
        print("Running particle swarm optimization...")

        # Parameters
        sample_size = population_size
        n_iterations = iterations

        # Initialize the ABC Object
        pso = PSO(self, self.dimension, sample_size, self.bounds)

        # Initialize Base
        pso.initializeAll()

        # Generations
        for _ in range(n_iterations):
            for i in range(sample_size):
                x_new = pso.generateNewSolution(i)
                pso.updateSolution(x_new, i)

        # Solution
        return pso.getSolution()

