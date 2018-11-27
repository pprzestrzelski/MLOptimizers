# Inspired by: many thanks to the author of https://github.com/chncyhn/simulated-annealing-tsp

import math
import random
import utils
import matplotlib.pyplot as plt

# TODO: https://github.com/ildoonet/simulated-annealing-for-tsp based on the following article
# TODO: read https://www.sciencedirect.com/science/article/pii/S1568494611000573


class SimulatedAnnealing(object):
    """
    Simulated annealing algorithm
    coords - coordinates of cities as [[x1, y1], [x2, y2] ... [xN, yN]]
    temp - temperature used in Boltzmann factor (stopping_temp < temp < ~1000); never drops below 0.0
    alpha - cooling fraction (~= 0.8 <= alpha <= 0.99)
    initial_solution - initial solution of TSP problem: greedy or random
    shuffle_algorithm - algorithm mixing cities within solution: reverse or two_point_swap
    !STOP CONDITIONS:
    stopping_temp - temperature below which algorithm stops (min has to be more than 0.0)
    stopping_iter - max number of iterations
    """
    def __init__(self, coords, temp=-1, alpha=-1, stopping_temp=-1, stopping_iter=-1, target=0.0,
                 initial_solution="random", shuffle_algorithm="reverse", debug=False):
        self.coords = coords
        self.N = len(coords)
        self.T = self.N * 100 if temp == -1 else temp
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 0.000001 if stopping_temp == -1 else stopping_temp
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.target_value = target
        self.shuffle_algorithm = shuffle_algorithm
        self.debug = debug

        self.dist_matrix = self.create_distance_matrix(coords)
        self.nodes = [i for i in range(self.N)]

        self.init_sol = self.calculate_initial_solution(initial_solution)
        self.curr_sol = self.init_sol
        self.best_sol = list(self.curr_sol)

        self.curr_fit = self.fitness(self.curr_sol)
        self.initial_fitness = self.curr_fit
        self.best_fit = self.curr_fit

        self.fitness_list = [self.curr_fit]
        self.temperature_list = [self.T]

        random.seed()

    def calculate_initial_solution(self, name):
        if name == "random":
            solution = self.random_solution()
        elif name == "greedy":
            solution = self.greedy_solution()
        else:
            solution = self.random_solution()

        return solution

    # Greedy algorithm to get an initial solution (closest-neighbour)
    def greedy_solution(self):
        cur_node = random.choice(self.nodes)
        solution = [cur_node]
        free_list = list(self.nodes)
        free_list.remove(cur_node)
        epsilon = 0.0001

        while free_list:
            closest_dist = min([self.dist_matrix[cur_node][j] for j in free_list])
            for candidate in range(self.N):
                if abs(self.dist_matrix[cur_node][candidate] - closest_dist) < epsilon and candidate in free_list:
                    cur_node = candidate
            free_list.remove(cur_node)
            solution.append(cur_node)
        print('Greedy solution travel length: ', self.fitness(solution))

        return solution

    def random_solution(self):
        solution = []
        free_list = list(self.nodes)
        while free_list:
            cur_node = random.choice(free_list)
            solution.append(cur_node)
            free_list.remove(cur_node)
        print('Random solution travel length: ', self.fitness(solution))

        return solution

    @staticmethod
    def distance(coord1, coord2):
        return round(math.sqrt(math.pow(coord1[0] - coord2[0], 2) + math.pow(coord1[1] - coord2[1], 2)), 4)

    def create_distance_matrix(self, coords):
        """
        Creates matrix of distances between cities indexed from 0 to n
        :param coords: an array of [[x1,y1],...[xn,yn]]
        :return: nxn nested list from a list of length n
        """

        n = len(coords)
        mat = [[self.distance(coords[i], coords[j]) for i in range(n)] for j in range(n)]
        return mat

    def exec(self):
        print("Target travel length: {}, search space: {:.2E}"
              .format(self.target_value, self.calculate_search_space_size()))

        iteration = 1
        while self.T >= self.stopping_temperature and iteration < self.stopping_iter:
            candidate = list(self.curr_sol)
            self.shuffle_cities(candidate)

            if self.accept(candidate):
                self.T *= self.alpha
                # Reverse operation
                # self.T /= self.alpha

            self.fitness_list.append(self.curr_fit)
            self.temperature_list.append(self.T)

            progress = iteration / self.stopping_iter * 100.0
            if self.debug and progress % 100:
                utils.print_progress_bar(progress, self.best_fit, self.curr_fit, self.target_value)

            if self.debug and self.T < self.stopping_temperature:
                print("\nIt is so cold... annealing temperature ({}) dropped below the stop temperature ({})"
                      .format(round(self.T, 2), round(self.stopping_temperature), 6), end="")

            iteration += 1

        if self.debug:
            print()

    def shuffle_cities(self, cities):
        # Reverse cities order between i and k [i : i+k]
        if self.shuffle_algorithm == "reverse":
            self.reverse_cities_within_range(cities)
        # Choose randomly two points and swap them -- seems to be less efficient
        elif self.shuffle_algorithm == "two_point_swap":
            self.swap_two_random_cities(cities)
        else:
            self.swap_two_random_cities(cities)

    def reverse_cities_within_range(self, cities):
        k = random.randint(2, self.N-1)
        i = random.randint(0, self.N-k)
        cities[i:(i+k)] = reversed(cities[i:(i+k)])

    def swap_two_random_cities(self, cities):
        # TODO: test => k, i = random.sample(range(self.N), k=2)
        k = random.randrange(0, self.N - 1)
        i = random.randrange(0, self.N - 1)
        while i == k:
            i = random.randrange(0, self.N - 1)
        cities[i], cities[k] = cities[k], cities[i]

    def accept(self, candidate_solution):
        cand_fit = self.fitness(candidate_solution)
        if cand_fit < self.curr_fit:                        # Accept-win
            self.curr_fit = cand_fit
            self.curr_sol = candidate_solution
            if self.best_fit > cand_fit:
                self.best_fit = cand_fit
                self.best_sol = candidate_solution
        elif self.p_accept(cand_fit) > random.random():     # Accept-loss
            self.curr_fit = cand_fit
            self.curr_sol = candidate_solution
        else:                                               # Reject solution
            return False
        return True

    # Objective value of a solution
    def fitness(self, sol):
        return round(sum([self.dist_matrix[sol[i-1]][sol[i]] for i in range(1, self.N)])
                     + self.dist_matrix[sol[0]][sol[self.N-1]], 4)

    def p_accept(self, candidate_fitness):
        """
        Boltzmann distribution
        Acceptance probability - probability of accepting if the candidate is worse than current
        Depends on the current temperature and difference between candidate and current
        Two variants of pseudo-Boltzmann factor:
            1. exp(-abs(candidate_fitness - current_fitness) / temperature)
            2. a) normalized_delta = (candidate_fitness - current_fitness) / current_fitness
               b) exp(-1 * delta / temperature)
        Option with normalized delta of fitness shows better performance
        """

        # normalized delta
        delta = (candidate_fitness - self.curr_fit) / self.curr_fit
        return math.exp(-1 * delta / self.T)

    def visualize_initial_solution(self):
        utils.plot_tsp([self.init_sol], self.coords)

    def visualize_route(self):
        utils.plot_tsp([self.best_sol], self.coords)

    def plot_fitness(self):
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel('Fitness')
        plt.xlabel('Iteration')
        plt.show()

    def plot_temperature(self):
        plt.plot([i for i in range(len(self.temperature_list))], self.temperature_list)
        plt.ylabel('Temperature')
        plt.xlabel('Iteration')
        plt.show()

    def print_summary(self):
        print('Best fitness obtained: ', self.best_fit)
        print('Shortest route: ', self.best_sol)
        improvement = (self.initial_fitness - self.best_fit) / self.initial_fitness * 100
        print('Improvement over initial solution: ', round(improvement, 2), "%")

    def get_best_solution(self):
        return self.best_fit, self.best_sol

    def calculate_search_space_size(self):
        return int(math.factorial(self.N - 1) / 2)
