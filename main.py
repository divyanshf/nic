from functions.Ackley import Ackley
from functions.Rastrigin import Rastrigin
from algorithms.GA import GA
from functions.Rosenbrock import Rosenbrock
from functions.Sphere import Sphere
import pandas as pd


# Main
def main():

    rows = ['Rastrigin', 'Ackley', 'Sphere', 'Rosenbrock']
    cols = ['Solution', 'Optimal Value']
    all_solns = []

    # Rastrigin
    rastrigin = Rastrigin(2, [-5.12, 5.12])
    x, o = rastrigin.optimizeUsingGA()
    all_solns.append([x, o])

    # # # Ackley
    ackley = Ackley(2, [-5, 5])
    x, o = ackley.optimizeUsingGA()
    all_solns.append([x, o])


    # # Sphere
    sphere = Sphere(2, [-10000, 10000])
    x, o = sphere.optimizeUsingGA()
    all_solns.append([x, o])

    # # Rosenbrock
    rosen = Rosenbrock(2, [-10000, 10000])
    x, o = rosen.optimizeUsingGA()
    all_solns.append([x, o])

    # Display
    df = pd.DataFrame(all_solns, index=rows, columns=cols)
    print(df)


if __name__ == '__main__':
    main()