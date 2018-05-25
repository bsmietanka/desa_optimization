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
                crosspoint=0.8, # (0,1]
                start_temp=25000.0, # 0 to disable simulated annealing
                alpha=0.9,
                end_temp=1e-9,
                budget=1000,
                log=False):

    if log:
        logger.debug("budget  :  {}".format(budget))
    # TODO?: check if all input parameters are what we expect
    dimensions = len(lbounds)
    # TODO?: maybe population_size should be independent from dimesions
    population_size = pop_size * dimensions
    temp = start_temp

    # initialize population
    population = np.random.uniform(lbounds, ubounds, (population_size, dimensions))
    # population = lbounds + (ubounds - lbounds) * np.random.rand(population_size, dimensions)

    # serves as a mask for selecting 3 random elements from population except current element
    last = population_size - 1
    idxs = np.tile(np.arange(0,last), (population_size, 1))
    for i in range (0, last):
        idxs[i][i] = last

    # initial population evaluation
    values = np.apply_along_axis(func, 1, population)
    budget -= population_size
    # find initial candidate for best
    best_idx = np.argmin(values)
    current_best = best_idx
    best = population[best_idx]
    best_val = values[best_idx]

    if log:
        logger.debug("best value  :  {}".format(best_val))
        history = population
    iter = 0
    while budget > 0 and iter < maxiter:
        iter += 1
        # end condition
        if func.final_target_hit:
            if log:
                logger.debug('target hit in iteration {}  :  {}'.format(iter - 1, str(func)[:17]))
                logger.debug("evaluations left  :  {}".format(budget))
            else:
                history = []
            return history, best, best_val

        # TODO: strategy selection
        # selection and mutation
        # strategy rand/1/bin
        # selected = np.apply_along_axis(select_three, 1, idxs)
        # mutants = population[selected[:,2]] + mutation * (population[selected[:,0]] - population[selected[:,1]])
        # strategy best/1/bin
        selected = np.apply_along_axis(select_two, 1, idxs)
        mutants = population[current_best] + mutation * (population[selected[:,0]] - population[selected[:,1]])

        # clip values to fit into lower and upper bounds
        # think of better way of keeping in bounds, e.g. reflection
        mutants = np.clip(mutants, lbounds, ubounds)

        # crossover and candidate evaluation
        candidates = crossover(mutants, population, crosspoint, dimensions)
        candidate_values = np.apply_along_axis(func, 1, candidates)
        budget -= population_size

        # save better agents and their respective values
        succession_mask = np.less_equal(candidate_values, values)
        values = np.where(succession_mask, candidate_values, values)
        # needs reshape to save all coordinates of agent
        succession_mask = np.tile(succession_mask.reshape(population_size, 1), dimensions)
        population = np.where(succession_mask, candidates, population)

        # simulated annealing - create mask to accept 
        # worse results based on current temperature
        if temp > 0:
            annealing_mask = simulated_annealing(values, candidate_values, temp)
            values = np.where(annealing_mask, candidate_values, values)
            # reshape mask
            annealing_mask = np.tile(annealing_mask.reshape(population_size, 1), dimensions)
            population = np.where(annealing_mask, candidates, population)
            # update current temperature
            temp = cooling(temp, alpha, end_temp)

        # search for global best
        current_best = np.argmin(values)
        if best_val > values[current_best]:
            if log:
                logger.debug("best value  :  {}".format(best_val))
            best_val = values[current_best]
            best = population[current_best]

        if log:
            history = np.append(history, population, axis=0)

    if log:
        logger.debug("target not hit (iterations {}) :  {}".format(iter, str(func)[:17]))
    else:
        history = []
    return best, best_val, history

##################################################
######           helper functions           ######
##################################################

def crossover(mutants, ancestors, crosspoint, dimensions):
    crossover_mask = np.random.random_sample(mutants.shape) < crosspoint
    # print(crossover_mask)
    return np.where(crossover_mask, mutants, ancestors)

def select_three(array):
    return np.random.choice(array, 3, replace=False)
    # return np.array([1,2,3])

def select_two(array):
    return np.random.choice(array, 2, replace=False)

def simulated_annealing(prev_score, next_score, temperature):
    if temperature > 0:
        rejecting_prob = np.exp( -np.absolute(np.subtract(next_score, prev_score, dtype=np.float64))/temperature )
        return np.less(np.random.random_sample(rejecting_prob.shape), rejecting_prob)
    else:
        return np.full(next_score.shape, False)

def cooling(temp, alpha, end_temp):
    if temp > end_temp:
        return alpha * temp
    else:
        return 0

class my_func:

    def __init__(self):
        self.final_target_hit = False

    def __call__(self, x):
        # print(x)
        output = np.square(x)
        output = np.sum(output, 0)
        # smallest = output[np.argmin(output)]
        if output <= 4 * sys.float_info.epsilon:
            self.final_target_hit = True

        # print(output)
        return output

def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between
    `lbounds` and `ubounds`
    """
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), None, None
    max_chunk_size = 1 + 4e4 / dim
    while budget > 0:
        chunk = int(max([1, min([budget, max_chunk_size])]))
        # about five times faster than "for k in range(budget):..."
        X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
        F = [fun(x) for x in X]
        budget -= chunk
        if fun.final_target_hit:
            break
    return x_min, budget


if __name__ == "__main__":
    f = my_func()
    history, _, _ = desa_solver(f, [-5,-5], [5,5],
        pop_size=int(sys.argv[1] if len(sys.argv) > 1 else 20),
        budget=100000,
        maxiter = 1000,
        mutation=0.7,
        crosspoint=0.8,
        start_temp=10000,
        log=True)
    _, budget1 = random_search(f, [-5,-5], [5,5], 100000)
    print(budget1)
    print(history.shape)
    plt.plot(history[:,0], history[:,1], 'ro', markersize=1)
    plt.show()
