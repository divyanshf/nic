from functions.Rastrigin import Rastrigin
from algorithms.GA import GA

dimension = 2
population_size = 100
bounds = [-5.12, 5.12]

def main():
    fx = Rastrigin(dimension)
    ga = GA(fx, dimension, population_size, bounds)

    # Initialize Population
    ga.initializePopulation()

    iterations = 1000

    # Generations
    for _ in range(iterations):
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
    opt_value = fx.eval(X_res)
    print(X_res, '\n')
    print(opt_value)
    

if __name__ == '__main__':
    main()