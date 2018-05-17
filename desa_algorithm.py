import numpy as np
import sys
from matplotlib import pyplot as plt

# TODO: check common ranges for input parameter
# TODO: replace prints with logger
def desa_solver(func,
                lbounds,
                ubounds,
                pop_size=15,
                maxiter=-1,
                mutation=0.7, # usually in [0, 2]
                crosspoint=0.7,
                start_temp=25000.0,
                alpha=0.1,
                end_temp=1e-9,
                # check convergence for early stopping the algorithm
                budget=1000):

    # TODO? check if all input parameters are what we expect
    dimensions = len(lbounds)
    population_size = pop_size * dimensions
    temp = start_temp

    if budget > 0:
        evaluations_left = budget
    else:
        evaluations_left = 5000000

    # initialize population
    population = np.random.uniform(lbounds, ubounds, (population_size, dimensions))
    # population = lbounds + (ubounds - lbounds) * np.random.rand(population_size, dimensions)

    #serves as a mask for selecting 3 random elements from population except current element
    last = population_size - 1
    idxs = np.tile(np.arange(0,last), (population_size, 1))
    for i in range (0, last):
        idxs[i][i] = last

    # TODO: check if bbob func object support evaluation from matrix
    # initial population evaluation
    # values = np.array([func(x) for x in population])
    values = np.apply_along_axis(func, 1, population)
    budget -= population_size
    # find initial candidate for best
    best_idx = np.argmin(values)
    best = population[best_idx]
    best_val = values[best_idx]

    history = population

    if maxiter <= 0:
        maxiter = evaluations_left / population_size
    iter = 0
    while evaluations_left >= population_size or iter < maxiter:
        iter += 1

        # end condition
        if func.final_target_hit:
            # print("Final target reached in iteration: {}", iter)
            break
        
        if evaluations_left < population_size:
            # print("Evaluation budget used")
            break

        # order of operation in genetic algorithm:
        # 1. mutation
        # 2. crossover
        # 3. selection

        # for each agent select 3 other agents
        selected = np.apply_along_axis(select_three, 1, idxs)
        # differential mutation
        mutants = population[selected[:,0]] + mutation * (population[selected[:,1]] - population[selected[:,2]])
        # clip values to fit into lower and upper bounds
        # mutants = np.clip(mutants, lbounds, ubounds)
        # crossover
        candidates = crossover(mutants, population, crosspoint, dimensions)
        # candidates = mutants
        # evaluate candidates
        candidate_values = np.apply_along_axis(func, 1, candidates)
        # candidate_values = func(candidates)
        # update budget
        budget -= population_size
        # save better agents and their respective values
        succession_mask = np.less_equal(candidate_values, values)
        values = np.where(succession_mask, candidate_values, values)
        # needs a little reshape to save all coordinates of agent
        succession_mask = np.tile(succession_mask.reshape(population_size, 1), dimensions)
        population = np.where(succession_mask, candidates, population)

        # simulated annealing - create mask to accept 
        # worse results based on current temperature
        # annealing_mask = simulated_annealing(values, candidate_values, temp)
        # values = np.where(annealing_mask, candidate_values, values)
        # # reshape mask
        # annealing_mask = np.tile(annealing_mask.reshape(population_size, 1), dimensions)
        # population = np.where(annealing_mask, candidates, population)
        # # update current temperature
        # temp = cooling_schedule(temp, alpha)
        # search for global best
        current_best = np.argmin(values)
        # print "Current best : {}".format(values[current_best])
        if best_val > values[current_best]:
            best_val = values[current_best]
            best = population[current_best]

        # history = np.append(history, population, axis=0)
        # plt.plot(population[:,0], population[:,1], 'ro')
        # plt.show()
    # return best coordinates and value

    # print(history.shape)
    # plt.plot(history[:,0], history[:,1], 'ro', markersize=1)
    # plt.show()

    return history, best, best_val

##################################################
######           helper functions           ######
##################################################

def crossover(mutants, ancestors, crosspoint, dimensions):
    crossover_mask = np.random.random_sample(mutants.shape) < crosspoint
    # print(crossover_mask)
    return np.where(crossover_mask, mutants, ancestors)

def select_three(array):
    return np.random.choice(array, 3, replace=False)

def simulated_annealing(prev_score, next_score, temperature):
    if temperature > 0:
        rejecting_prob = np.exp( -np.absolute(np.subtract(next_score, prev_score, dtype=np.float64))/temperature )
        return np.less(np.random.random_sample(rejecting_prob.shape), rejecting_prob)
    else:
        return np.full(next_score.shape, False)

def cooling_schedule(temp, alpha):
    return alpha * temp

class my_func:

    def __init__(self):
        self.final_target_hit = False

    def __call__(self, x):
        # print(x)
        output = np.square(x)
        output = np.sum(output, 1)
        smallest = output[np.argmin(output)]
        if smallest <= 4 * sys.float_info.epsilon:
            self.final_target_hit = True

        # print(output)
        return output


if __name__ == "__main__":
    f = my_func()
    desa_solver(f, [4,4], [5,5],
        pop_size=int(sys.argv[1] if len(sys.argv) > 1 else 60),
        budget=-1,
        mutation=1,
        crosspoint=0.7,
        start_temp=10000)
