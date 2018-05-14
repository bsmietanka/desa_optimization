import numpy as np

# TODO: check common ranges for input parameter
def desa_solver(func,
                lbounds,
                ubounds,
                iterations=100,
                population_size=20,
                mutation=0.8, # usually in [0.5, 2]
                crosspoint=0.7,
                start_temp=100,
                alpha=0.01,
                budget=1000):

    # TODO? check if all input parameters are what we expect
    if population_size < 4:
        raise AttributeError("Population size cannot be less than 4")
    
    dimensions = len(lbounds)
    temp = start_temp

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
    values = np.array([func(x) for x in population])
    budget -= population_size
    # find initial candidate for best
    best_idx = np.argmin(values)
    best = population[best_idx]
    best_val = values[best_idx]
    
    for _ in range(0, iterations):
        # end condition
        if func.final_target_hit or budget < population_size:
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
        mutants = np.clip(mutants, lbounds, ubounds)
        # crossover
        candidates = crossover(mutants, population, crosspoint, dimensions)
        # evaluate candidates
        candidate_values = np.apply_along_axis(func, 1, candidates)
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
        annealing_mask = simulated_annealing(values, candidate_values, temp)
        values = np.where(annealing_mask, candidate_values, values)
        # reshape mask
        annealing_mask = np.tile(annealing_mask.reshape(population_size, 1), dimensions)
        population = np.where(annealing_mask, candidates, population)
        # update current temperature
        temp = cooling_schedule(temp, alpha)
        # search for global best
        current_best = np.argmin(values)
        if best_val > values[current_best]:
            best_val = values[current_best]
            best = population[current_best]
    # return best coordinates and value
    return best, best_val

##################################################
######           helper functions           ######
##################################################

def crossover(mutants, ancestors, crosspoint, dimensions):
    crossover_mask = np.random.random_sample(mutants.shape) < crosspoint
    return np.where(crossover_mask, mutants, ancestors)

def select_three(array):
    return np.random.choice(array, 3, replace=False)

def simulated_annealing(prev_score, next_score, temperature):
    rejecting_prob = np.exp( -np.absolute(next_score-prev_score)/temperature )
    return np.less(np.random.random_sample(rejecting_prob.shape), rejecting_prob)

def cooling_schedule(temp, alpha):
    return alpha * temp

def my_func(x):
    return 1

if __name__ == "__main__":
    desa_solver(my_func, [0,0], [10, 10])
