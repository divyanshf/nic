# Binary Encoded GA
# Single Point Crossover
# Bit Flip Mutation
# Roulette Wheel Selection
import numpy as np

class GA:
    def __init__(self, objective_function, dimension, sample_size, bounds):
        self.objective_function = objective_function
        self.sample_size = sample_size
        self.dimension = dimension
        self.population = []
        self.bounds = bounds
        self.soln_len = 12
        self.fitnesses = [0 for _ in range(sample_size)]
        self.prob_selection = []
        self.prob_crossover = 0.5
        self.prob_mutation = 0.05

    # Initialize population randomly
    def initializePopulation(self):
        self.population = []
        for i in range(self.sample_size):
            x = np.random.random_integers(0, 2 ** self.soln_len)
            self.population.append(self.encode(x))
        return self.population
    
    # Calculate Fitness for all population
    def calculateFitnesses(self):
        self.fitnesses = []
        for soln in self.population:
            X = self.getVariables(soln) #   x1, x2, ... xn
            fit = self.objective_function.getFitness(X)
            self.fitnesses.append(fit);
        return self.fitnesses
    
    # Calcualte Probabilities for each soln depending on fitness
    def calculateProbabilities(self):
        self.prob_selection = []
        total_fitness = np.sum(self.fitnesses)
        for i in range(self.sample_size):
            self.prob_selection.append(self.fitnesses[i] / total_fitness)

    # Select from population
    def select(self):
        children = []
        for _ in range(int(self.sample_size / 2)):
            rnd = np.random.random_integers(0,1)
            pind1 = np.random.choice(self.sample_size, p=self.prob_selection)
            pind2 = np.random.choice(self.sample_size, p=self.prob_selection)
            if rnd > self.prob_crossover:
                children.append(self.population[pind1])
                children.append(self.population[pind2])
            else:
                p1, p2 = self.crossover(self.population[pind1], self.population[pind2])
                children.append(p1)
                children.append(p2)
        return children

    # Crossover
    def crossover(self, p1, p2):
        crossover_site = np.random.random_integers(1, self.soln_len - 1)
        temp1 = p1[crossover_site:]
        temp2 = p2[crossover_site:]
        p1 = p1[:crossover_site] + temp2
        p2 = p2[:crossover_site] + temp1
        return p1, p2

    # Mutation
    def mutate(self, children):
        total_bits = self.soln_len * self.sample_size
        n_mutations = int(self.prob_mutation * total_bits)
        for _ in range(n_mutations):
            ind = np.random.random_integers(0, total_bits-1)
            soln_idx = int(ind / self.soln_len)
            bit_idx = int(ind % self.soln_len)
            temp = children[soln_idx]
            child = temp[:bit_idx] + ('0' if temp[bit_idx] == '1' else '1') + temp[bit_idx+1:]
            children[soln_idx] = child
        return children

    # Get all dimensional variables from the solution string
    def getVariables(self, soln):
        X = []
        var_len = int(self.soln_len / self.dimension)
        for i in range(0, self.soln_len, var_len):
            X.append(self.decodeString(soln[i:i+var_len]))
        return X
    
    # Update generation
    def updateGeneration(self, children):
        self.population = children

    # Get the individual with best fitness
    def getResult(self):
        self.calculateFitnesses()
        best_idx = np.argmax(self.fitnesses)
        res = self.getVariables(self.population[best_idx])
        return res

    # Encode value to binary string
    def encode(self, x):
        s = bin(x).replace("0b", "")
        remain = self.soln_len - len(s)
        for _ in range(remain):
            s = '0' + s;
        return s

    # Decode binary string to value in bounds
    def decodeString(self, var_string):
        mask = 1
        temp = 0
        for ch in reversed(var_string):
            temp += mask * (ch == '1')
            mask <<= 1
        return self.bounds[0] + ((self.bounds[1] - self.bounds[0]) / (2**(self.soln_len / self.dimension) - 1)) * temp
    
