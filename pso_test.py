#!/usr/bin/env python

from pso import PSO
from utils import let_me_choose_the_route


if __name__ == '__main__':
    # target, coords = read_data_for_tsp('circle8.route')
    target, coords = let_me_choose_the_route()
    pso = PSO(coords, v_max=50, particle_count=50, max_epochs=5000, target_value=target, debug=False)
    pso.exec()
    pso.print_summary()
    pso.visualize_route()
