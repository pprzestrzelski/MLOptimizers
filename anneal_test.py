#!/usr/bin/env python

from anneal import SimulatedAnnealing
from utils import let_me_choose_the_route

if __name__ == '__main__':
    # coords = [[round(random.uniform(-1000,1000),4),round(random.uniform(-1000,1000),4)] for i in range(100)]
    target, coords = let_me_choose_the_route()

    # == Optimal values for att48
    # stopping_iter=500000
    # temp=900
    # alpha=0.9985
    # initial_solution="random"
    # shuffle_algorithm="reverse"
    sa = SimulatedAnnealing(coords, stopping_iter=500000, temp=900, alpha=0.9985, stopping_temp=0.00001,
                            target=target, initial_solution="random", shuffle_algorithm="reverse",
                            debug=True)
    sa.visualize_initial_solution()
    sa.exec()
    sa.print_summary()
    sa.visualize_route()
    sa.plot_fitness()
    sa.plot_temperature()
