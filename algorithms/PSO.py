import numpy as np


class PSO:
    def __init__(self, objective_function, dimension, sample_size, bounds):
        self.objective_function = objective_function
        self.dimension = dimension
        self.bounds = bounds
        self.sample_size = sample_size
        self.population = []
        self.velocities = []
        self.fitnesses = []
        self.pBest = []
        self.pFitnesses = np.Inf
        self.gBest = []
        self.gFitness = np.Inf
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5

    # Initialize Population Randomly
    def initializeAll(self):
        self.population = [
            np.random.uniform(self.bounds[0], self.bounds[1], self.dimension)
            for _ in range(self.sample_size)
        ]
        self.fitnesses = [
            self.objective_function.getFitness(p) for p in self.population
        ]

        self.velocities = [
            np.random.uniform(self.bounds[0], self.bounds[1], self.dimension)
            for _ in range(self.sample_size)
        ]

        self.pBest = self.population
        self.pFitnesses = self.fitnesses

        self.gBest = self.population[np.argmax(self.fitnesses)]
        self.gFitness = self.objective_function.getFitness(self.gBest)

    # Generate a new solution
    def generateNewSolution(self, index):
        r1 = np.random.uniform(0, 1, self.dimension)
        r2 = np.random.uniform(0, 1, self.dimension)
        current = np.copy(self.population[index])
        self.velocities[index] = (
            self.w * self.velocities[index]
            + (self.c1 * r1 * (self.pBest[index] - current))
            + (self.c2 * r2 * (self.gBest - current))
        )
        x_new = current + self.velocities[index]

        # Bound the solution
        for j in range(self.dimension):
            if x_new[j] < self.bounds[0]:
                x_new[j] = self.bounds[0]
            if x_new[j] > self.bounds[1]:
                x_new[j] = self.bounds[1]

        return x_new

    # Update a solution with new solution
    def updateSolution(self, x_new, index):
        self.population[index] = x_new
        self.fitnesses[index] = self.objective_function.getFitness(x_new)

        # Update personal best at index if needed
        if self.pFitnesses[index] < self.fitnesses[index]:
            self.pFitnesses[index] = self.fitnesses[index]
            self.pBest[index] = x_new

        # Update global best if needed
        if self.gFitness < self.fitnesses[index]:
            self.gFitness = self.fitnesses[index]
            self.gBest = x_new

    # Get solution
    def getSolution(self):
        return self.gBest, self.objective_function.eval(self.gBest)
