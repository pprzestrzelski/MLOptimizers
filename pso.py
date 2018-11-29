# Many thanks to http://mnemstudio.org/particle-swarm-tsp-example-1.htm

import random
import math
import utils
from city import City

import crossover


class Particle:
    def __init__(self, city_count):
        self.best_sol = [0] * city_count
        self.pbest = 0
        self.curr_sol = [0] * city_count
        self.curr_p = 0.0
        self.velocity = 0.0

    def get_curr_sol(self):
        return self.curr_sol

    def get_data_from_curr_sol(self, index):
        return self.curr_sol[index]

    def set_data_in_curr_sol(self, index, value):
        self.curr_sol[index] = value

    def set_curr_sol(self, solution):
        self.curr_sol = list(solution)

    def get_best_sol(self):
        return self.best_sol

    def set_best_sol(self, solution):
        self.best_sol = list(solution)

    def get_pbest(self):
        return round(self.pbest, 4)

    def set_pbest(self, v):
        self.pbest = v

    def get_p(self):
        return self.curr_p

    def set_p(self, p):
        self.curr_p = p

    def get_v(self):
        return self.velocity

    def set_v(self, v):
        self.velocity = v


class PSO(object):
    def __init__(self, coords, particle_count=10, v_max=4, max_epochs=1000, target_value=0, debug=False):
        self.particles = []
        self.cities = []
        self.vec_of_coordinates = coords    # [[x1, y1], [x2, y2]...[xN, yN]]
        self.city_count = len(coords)
        self.particle_count = particle_count
        self.v_max = self.set_v_max(v_max)
        self.max_epochs = max_epochs
        self.target_value = target_value
        self.debug = debug
        self.initialize_data(coords)

    def set_v_max(self, v_max):
        if v_max >= self.city_count or v_max < 0:
            value = self.city_count
        else:
            value = v_max

        return value

    def initialize_data(self, coords):
        self.initialize_cities(coords)
        self.initialize_particles()

    def initialize_cities(self, coords):
        for i in range(self.city_count):
            self.cities.append(City(coords[i][0], coords[i][1]))

    def initialize_particles(self):
        for i in range(self.particle_count):
            new_particle = Particle(self.city_count)

            for j in range(self.city_count):
                new_particle.set_data_in_curr_sol(j, j)

            new_particle.set_v(random.randrange(0, self.v_max))

            random.shuffle(new_particle.get_curr_sol())
            new_particle.set_best_sol(new_particle.get_curr_sol())

            fitness = self.calculate_fitness(new_particle.get_curr_sol())
            new_particle.set_p(fitness)
            new_particle.set_pbest(fitness)

            self.particles.append(new_particle)

    def calculate_fitness(self, path):
        """
        Euclidean distance
        :param path: vector of arranged cities indexes
        :return: euclidean distance between all cities from an array
        """
        fitness = 0.0
        for i in range(self.city_count - 1):
            city_1 = self.cities[path[i]]
            city_2 = self.cities[path[i + 1]]
            fitness += City.calculate_distance(city_1, city_2)

        last_city = path[-1]
        first_city = path[0]
        fitness += City.calculate_distance(self.cities[last_city], self.cities[first_city])

        return fitness

    def exec(self):
        print("Target travel length: {}, search space: {:.2E}"
              .format(self.target_value, self.calculate_search_space_size()))

        target_reached = False
        epoch = 1
        while not target_reached and epoch <= self.max_epochs:
            # # PSO algorithm:
            # 0. Initialize particles
            # 1. Evaluate solutions in particles
            target_reached = self.evaluate_fitness()

            # => auxiliary operation <=
            self.quicksort(self.particles, 0, self.particle_count - 1)

            # 2. Update velocities of particles
            self.update_velocities()

            # 3. Update positions of particles and repeat steps 1-3 until conditions are not fulfilled
            self.update_positions()

            progress = epoch / self.max_epochs * 100.0
            if not self.debug and progress % 100:
                utils.print_progress_bar(progress, self.particles[0].get_pbest(),
                                         self.particles[-1].get_pbest(), self.target_value)
            epoch += 1

        if self.debug:
            print("Algorithm finished...\n")
        else:
            print()

    # Returns True if target value has been reached!
    def evaluate_fitness(self):
        if self.debug:
            print("=========== Fitness evaluation ===========")

        for i in range(self.particle_count):
            if self.debug:
                print("Particle {} with route: ".format(i), end="")
                for j in range(self.city_count):
                    print(str(self.particles[i].get_best_sol()) + ", ", end="")

            p = self.particles[i].get_p()
            pbest = self.particles[i].get_pbest()
            if p < pbest:
                self.particles[i].set_pbest(p)
                self.particles[i].set_best_sol(self.particles[i].get_curr_sol())
            else:
                self.particles[i].set_p(pbest)
                self.particles[i].set_curr_sol(self.particles[i].get_best_sol())

            if self.debug:
                print("Distance: " + str(self.particles[i].get_pbest()))

            if self.particles[i].get_pbest() <= self.target_value:
                if self.debug:
                    print("==> TARGET VALUE REACHED for particle #{}".format(i))
                return True

        return False

    def update_velocities(self):
        # Goldbarg E., Goldbarg M. and de Souza G. (2008) "Particle Swarm Optimization Algorithm for
        # the Traveling Salesman Problem", Chapter in: "Traveling Salesman Problem"
        w = 1.2     # inertia - usually between 0.8 and 1.2
        gworst = self.particles[-1].get_pbest()
        # gbest = self.particles[0].get_pbest()
        # c1 = 2.0
        # c2 = 2.0

        for i in range(self.particle_count):
            pbest = self.particles[i].get_pbest()
            # v = self.particles[i].get_v()
            # p = self.particles[i].get_p()
            # r1 = random.uniform(0.0, 1.0)
            # r2 = random.uniform(0.0, 1.0)

            # Original equation on velocity estimation:
            # v_value = w * v + c1 * r1 * (pbest - p) + c2 * r2 * (gbest - p)
            #   where: w * v - inertial component
            #          c1 * r1 * (pbest - p) - social component
            #          c2 * r2 * (gbest - p) - cognitive component
            # more to read: www.cs.armstrong.edu/saad/csci8100/pso_slides.pdf
            v_value = math.floor(w * ((self.v_max * pbest) / gworst))

            if v_value > self.v_max:
                self.particles[i].set_v(self.v_max)
            else:
                self.particles[i].set_v(v_value)

    def update_positions(self):
        for i in range(self.particle_count):
            changes = self.particles[i].get_v()
            rate = float(changes) / float(self.v_max)
            rate = (1 - rate)

            if self.debug:
                print("Changes for particle with distance {}: {}".format(
                    self.particles[i].get_pbest(),
                    str(changes)))

            curr = self.particles[i].get_curr_sol()
            if i > 0:
                global_best = self.particles[0].get_best_sol()
                nearest_neighbour_best = self.particles[i-1].get_best_sol()

                for c in range(changes):
                    if random.uniform(0.0, 1.0) > 0.50:         # 50/50 roulette
                        self.swap_cities(curr)

                    if random.uniform(0.0, 1.0) > rate:
                        if random.uniform(0.0, 1.0) > 0.95:     # 5% probability to crossover with the global best
                            crossover.cx_ordered(list(global_best), curr)
                        else:                                   # 95% probability to crossover with a nearest particle
                            crossover.cx_ordered(list(nearest_neighbour_best), curr)
            else:
                if random.uniform(0.0, 1.0) > 0.5:
                    self.mutate_reverse(curr)
                else:
                    self.swap_cities(curr)

            curr_fit = self.calculate_fitness(curr)
            self.particles[i].set_p(curr_fit)

    def swap_cities(self, solution):
        idx_a, idx_b = random.sample(range(self.city_count), k=2)
        solution[idx_a], solution[idx_b] = solution[idx_b], solution[idx_a]

    def mutate_reverse(self, src):
        a, b = random.sample(range(self.city_count), k=2)
        if a > b:
            a, b = b, a
        src[a:(a + b)] = reversed(src[a:(a + b)])

    def simple_crossover(self, src, dest):
        # push destination's data points closer to source's data points.
        city_a = random.randrange(0, self.city_count)  # source's city to target.
        city_b = 0
        index_a_in_dest = 0
        index_b_in_dest = 0

        # city_b will be source's neighbour immediately succeeding city_a (circular).
        for i in range(self.city_count):
            if self.particles[src].get_data_from_curr_sol(i) == city_a:
                if i == self.city_count - 1:
                    city_b = self.particles[src].get_data_from_curr_sol(0)  # if end of array, take from beginning.
                else:
                    city_b = self.particles[src].get_data_from_curr_sol(i + 1)
                break

        # Move city_b next to city_a by switching values.
        for j in range(self.city_count):
            if self.particles[dest].get_data_from_curr_sol(j) == city_a:
                index_a_in_dest = j

            if self.particles[dest].get_data_from_curr_sol(j) == city_b:
                index_b_in_dest = j

        # get temp index succeeding index_a_in_dest.
        if index_a_in_dest == self.city_count - 1:
            tmp_index = 0
        else:
            tmp_index = index_a_in_dest + 1

        # Switch index_b_in_dest value with tmp_index value.
        temp = self.particles[dest].get_data_from_curr_sol(tmp_index)
        self.particles[dest].set_data_in_curr_sol(
            tmp_index,
            self.particles[dest].get_data_from_curr_sol(index_b_in_dest))
        self.particles[dest].set_data_in_curr_sol(index_b_in_dest, temp)

    def quicksort(self, array, left, right):
        pivot = self.quicksort_partition(array, left, right)

        if left < pivot:
            self.quicksort(array, left, pivot - 1)

        if right > pivot:
            self.quicksort(array, pivot + 1, right)

        return array

    def quicksort_partition(self, numbers, left, right):
        pivot = numbers[left]

        while left < right:
            while (numbers[right].get_pbest() >= pivot.get_pbest()) and (left < right):
                right -= 1

            if left != right:
                numbers[left] = numbers[right]
                left += 1

            while (numbers[left].get_pbest() <= pivot.get_pbest()) and (left < right):
                left += 1

            if left != right:
                numbers[right] = numbers[left]
                right -= 1

        numbers[left] = pivot
        pivot = left

        return pivot

    def print_summary(self):
        if self.particles[0].get_pbest() <= self.target_value:
            print("Target reached.")
        else:
            print("Target not reached.")
        print("Shortest Route:", self.particles[0].get_best_sol())
        print("Distance:" + str(self.particles[0].get_pbest()))

    def visualize_route(self):
        utils.plot_tsp([self.particles[0].get_best_sol()], self.vec_of_coordinates)

    def calculate_search_space_size(self):
        return int(math.factorial(self.city_count - 1) / 2)
