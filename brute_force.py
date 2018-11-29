import math
from itertools import permutations
from city import City
from utils import let_me_choose_the_route


# TODO: optimize brute force solutions - do not use itertools.permutations
def brute_force(cities):
    paths = []
    lengths = []
    city_count = len(cities)
    search_space = int(math.factorial(city_count))  # True search space in TSP is (n - 1)! / 2
    p = 1
    for path in permutations(range(len(cities))):
        paths.append(path)
        lengths.append(calculate_route_distance(cities, path))
        progress = p / search_space * 100.0
        if progress % 100:
            print("\rWork in progress: {}%".format(round(progress, 0)), end='')
        p += 1

    print()
    return lengths, paths


def calculate_route_distance(cities, path):
    dist = 0.0
    city_count = len(cities)
    for i in range(city_count - 1):
        city_1 = cities[path[i]]
        city_2 = cities[path[i + 1]]
        dist += City.calculate_distance(city_1, city_2)

    last_city = path[-1]
    first_city = path[0]
    dist += City.calculate_distance(cities[last_city], cities[first_city])

    return dist


if __name__ == '__main__':
    target, coords = let_me_choose_the_route()
    cities = []
    for i in range(len(coords)):
        cities.append(City(coords[i][0], coords[i][1]))
    lengths, paths = brute_force(cities)
    print("Min:", round(min(lengths), 4))
