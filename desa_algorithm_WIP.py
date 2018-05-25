import numpy as np
import sys
from matplotlib import pyplot as plt
import logging

logger = logging.getLogger()
handler = logging.FileHandler('debug.log')
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# TODO: check common ranges for input parameter
# TODO: replace prints with logger
def desa_solver(func,
                lbounds,
                ubounds,
                pop_size=15,
                maxiter=1000,
                mutation=0.7, # usually in [0, 2]
                crosspoint=0.7,
                start_temp=25000.0,
                alpha=0.1,
                end_temp=1e-9,
                # check convergence for early stopping the algorithm
                budget=1000):

    logger.debug("budget  :  {}".format(budget))
    # TODO? check if all input parameters are what we expect
    dimensions = len(lbounds)
    # maybe population_size should be independent from dimesionsx
    population_size = pop_size * dimensions
    temp = start_temp

    # initialize population
    population = np.random.uniform(lbounds, ubounds, (population_size, dimensions))
    # population = lbounds + (ubounds - lbounds) * np.random.rand(population_size, dimensions)

    #serves as a mask for selecting 3 random elements from population except current element
    last = population_size - 1
    idxs = np.arange(0,last)
    idxs[0] = last

    # initial population evaluation
    values = np.apply_along_axis(func, 1, population)
    budget -= population_size
    # find initial candidate for best
    best_idx = np.argmin(values)
    best = population[best_idx]
    best_val = values[best_idx]
    logger.debug("best value  :  {}".format(best_val))

    history = population
    iter = 0
    while budget > 0 and iter < maxiter:
        iter += 1

        # end condition
        if func.final_target_hit:
            logger.debug('target hit in iteration {}  :  {}'.format(iter - 1, str(func)[:17]))
            logger.debug("evaluations left  :  {}".format(budget))
            return history, best, best_val

        # strategy rand/1/bin
        next_generation = np.zeros(population.shape)
        new_values = np.zeros(values.shape)
        for i, agent in enumerate(population):
            if i > 0:
                idxs[i - 1] = i - 1
            if i < last:
                idxs[i] = last
            a, b, c = select_three(idxs)

            mutant = population[a] + mutation * (population[b] + population[c])
            mutant = np.clip(mutant, lbounds, ubounds)
            candidate = crossover(mutant, agent, crosspoint)

            candidate_value = func(candidate)
            budget -= 1
            if candidate_value <= values[i]:
                new_values[i] = candidate_value
                next_generation[i] = candidate
            else:
                new_values[i] = values[i]
                next_generation[i] = population[i]

        population = next_generation
        values = new_values

        # search for global best
        current_best = np.argmin(values)
        # print "Current best : {}".format(values[current_best])
        if best_val > values[current_best]:
            logger.debug("best value  :  {}".format(best_val))
            best_val = values[current_best]
            best = population[current_best]

        history = np.append(history, population, axis=0)

    # return best coordinates and value\
    logger.debug("target not hit (iterations {}) :  {}".format(iter, str(func)[:17]))
    return history, best, best_val

##################################################
######           helper functions           ######
##################################################

def crossover(mutant, ancestor, crosspoint):
    crossover_mask = np.random.random_sample(mutant.shape) < crosspoint
    # print(crossover_mask)
    return np.where(crossover_mask, mutant, ancestor)

def select_three(array):
    return np.random.choice(array, 3, replace=False)
    # return np.array([1,2,3])

class my_func:

    def __init__(self):
        self.final_target_hit = False

    def __call__(self, x):
        # print(x)
        output = np.square(x)
        output = np.sum(output, 0)
        # smallest = output[np.argmin(output)]
        if output <= 100 * sys.float_info.epsilon:
            self.final_target_hit = True

        # print(output)
        return output


if __name__ == "__main__":
    f = my_func()
    history, _, _ = desa_solver(f, [4,4], [5,5],
        pop_size=int(sys.argv[1] if len(sys.argv) > 1 else 5),
        budget=500000,
        maxiter=500,
        mutation=1,
        crosspoint=0.7,
        start_temp=10000)

    plt.plot(history[:,0], history[:,1], 'ro', markersize=1)
    plt.show()