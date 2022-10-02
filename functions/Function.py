from algorithms.GA import GA


class Function:
    def __init__(self, dimension, bounds, prob_crossover=0.5, prob_mutation=0.05):
        print('Optimization Function : ', end='')

        # Make the soln_len divisible by dimension if not already

        self.var_len = 32
        self.dimension = dimension
        self.bounds = bounds
        self.soln_len = self.dimension * self.var_len
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation

    # Get Fitness
    def getFitness(self, X):
        f = self.eval(X)
        return 1e9 if f == 0 else 1/f
    
    # Optimize Using GA
    def optimizeUsingGA(self):
        print('Running genetic algorithm...')
        sample_size = self.dimension * 250
        n_generations = self.dimension * 1000

        # Genetic Algorithm Object
        ga = GA(self, self.dimension, sample_size, self.bounds, soln_len=self.soln_len, var_len=self.var_len, prob_crossover=self.prob_crossover, prob_mutation=self.prob_mutation)

        # Initialize Population
        ga.initializePopulation()

        # Generations
        for _ in range(n_generations):
            # calcualte fitness
            ga.calculateFitnesses()
            # calculate probability
            ga.calculateProbabilitiesRoulette()
            # Select parents and crossover
            children = ga.select()
            # Mutate the children
            mutated_children = ga.mutate(children)
            # Generation Update
            ga.updateGeneration(mutated_children)

        X_res = ga.getResult()
        opt_value = self.eval(X_res)
        return X_res, opt_value