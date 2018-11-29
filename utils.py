import matplotlib.pyplot as plt
import os


def let_me_choose_the_route():
    routes, route_paths = find_route_files("./routes/")
    n = len(routes)
    if n == 0:
        print("No routes to read...")
        return
    else:
        default = 0
        print("List of available routes (#1 is default):")
        i = 0
        for route in routes:
            print("  {} -> {}".format(i+1, route))
            i += 1

        r_str = input("Choose your route: ")
        try:
            r = int(r_str)
        except ValueError:
            print("Non int value passed, default route will be returned")
            return read_data_for_tsp(route_paths[default])
        if r <= 0 or r > n:
            print("Out of range value has been chosen. Default route will be returned.")
            return read_data_for_tsp(route_paths[default])

        return read_data_for_tsp(route_paths[r-1])


def find_route_files(dir_name):
    routes = []
    route_paths = []
    for file in os.listdir(dir_name):
        if file.endswith(".route"):
            routes.append(file)
            route_paths.append(os.path.abspath(dir_name + file))
    return routes, route_paths


def read_data_for_tsp(file):
    coords = []
    target_length = 0.0
    with open(file, 'r') as f:
        i = 0
        for line in f.readlines():
            if i == 0:
                spliced = line.split(' ')
                target_length = float(spliced[1])
            else:
                spliced = line.split(' ')
                coords.append([float(spliced[1]), float(spliced[2])])
            i += 1
    return target_length, coords


def print_progress_bar(progress, curr_best, curr_worst, target=0.0):
    best_worst_range = curr_worst - curr_best
    if target < 0.000001:
        print("\rWork in progress: {}% (Current best {}, bwr {})".format(
            round(float(progress), 0),
            round(float(curr_best), 4),
            round(float(best_worst_range), 4)),
            end="")
    else:
        err = (curr_best - target) / target * 100.0
        print("\rWork in progress: {}% (Current best {}, error {}%, bwr {})".format(
            round(float(progress), 0),
            round(float(curr_best), 4),
            round(float(err), 2),
            round(float(best_worst_range), 4)),
            end="")


def plot_tsp(paths, points, num_iters=1):
    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list

    """

    # Unpack the primary TSP path and transform it into a list of ordered
    # coordinates

    x = []
    y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])

    plt.plot(x, y, 'co')

    # Set a scale for the arrow heads (there should be a reasonable default for this, WTF?)
    a_scale = float(max(x))/float(100)

    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = []
            yi = []
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                      head_width=a_scale, color='r',
                      length_includes_head=True, ls='dashed',
                      width=0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                          head_width=a_scale, color='r', length_includes_head=True,
                          ls='dashed', width=0.001/float(num_iters))

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width=a_scale,
              color='g', length_includes_head=True)
    for i in range(0, len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width=a_scale,
                  color='g', length_includes_head=True)

    # Set axis too slightly larger than the set of x and y
    dx = (max(x) - min(x)) * 0.05
    dy = (max(y) - min(y)) * 0.05

    plt.xlim(min(x)-dx, max(x)+dx)
    plt.ylim(min(y)-dy, max(y)+dy)
    plt.show()
