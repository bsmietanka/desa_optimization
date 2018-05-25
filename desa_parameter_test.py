from desa_algorithm import desa_solver, sphere, ackeley, rosenbrock, himmelblau
import argparse
from matplotlib import pyplot as plt

functions = { 'sphere', 'ackeley',
              'rosenbrock', 'himmelblau'}

kwargs = {"budget" : 1e3, "mutation" : 0.7, "crosspoint" : 0.7,
          "start_temp" : 25000, "alpha" : 0.8, "pop_size" : 15}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test parameter influence on performance of desa solver')
    parser.add_argument('function', type=str, choices=functions,
                        help='name of function to optimize')
    parser.add_argument('parameter', nargs='?', choices=kwargs.keys(),
                        help='parameter name')
    parser.add_argument('value', nargs='?', type=float,
                        help='parameter value')
    args = parser.parse_args()

    if args.function == 'sphere':
        title = 'Sphere'
        f = sphere()
    elif args.function == 'ackeley':
        title = 'Ackeley'
        f = ackeley()
    elif args.function == 'rosenbrock':
        title = 'Rosenbrock'
        f = rosenbrock()
    elif args.function == 'himmelblau':
        title = 'Himmelblau'
        f = himmelblau()
    else:
        print("Specify one of the following function: {}".format(functions))
        exit()
    kwargs[args.parameter] = args.value
    best, best_val, history = desa_solver(f, f.lbounds, f.ubounds,
                    budget=int(kwargs['budget']), pop_size=int(kwargs['pop_size']),
                    mutation=kwargs['mutation'], crosspoint=kwargs['crosspoint'],
                    start_temp=kwargs['start_temp'], alpha=kwargs['alpha'],
                    log=True, seed=True)

    plt.plot(history[:,0], history[:,1], 'ro', markersize=1)
    plt.title("{}, {} = {},\nbest: {} = {}".format(title,
                args.parameter, args.value, best, best_val))
    plt.savefig("results/{}_{}_{}.png".format(args.function, args.parameter, args.value))
