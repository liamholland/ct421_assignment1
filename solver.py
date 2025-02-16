import tsplib95 as tsp
import random as rand
import matplotlib.pyplot as plt
import time
import csv

problem = tsp.load('tsp_files\\berlin52.tsp')

SOLUTION_POOL_SIZE = 1000
NUM_NODES = len(list(problem.get_nodes()))
MAX_ITERATIONS = 1500

# initialise the population
def initialise_population():
    # random initialisation
    return [rand.sample(list(problem.get_nodes()), NUM_NODES) for _ in range(SOLUTION_POOL_SIZE)]

# elitism selection
def elitism(solutions, num_to_select):
    # get the top n of the solutions
    solutions_sorted = sorted(solutions, key=fitness)
    return solutions_sorted[:num_to_select]  # first n of the list (shortest n paths)

# tournament selection
def tournament(solutions, num_to_select, size=3):
    # get the min value of the tournament, since fitness is the path length
    take_sample = lambda: min(rand.sample(solutions, k=size), key=fitness)
    return [take_sample() for _ in range(num_to_select)]

def roulette_wheel(solutions, num_to_select):
    total_fitness = sum([fitness(sol) for sol in solutions])    # find the total fitness of the solution pool 
    weights = [1 - (fitness(sol)/total_fitness) for sol in solutions]   # create a corresponding weight against the total weight for each solution
    return rand.choices(solutions, weights, k=num_to_select)    # return a selected pool based on the solution weights

# swap two items in the path
def swap_mutation(solutions, rate):

    # select a random set of samples based on the mutation rate
    num_selections = int(len(solutions) * rate)
    rand.shuffle(solutions)
    samples = solutions[:num_selections]

    for i in range(num_selections):

        mutated_solution = samples[i].copy()

        # pick two indeces to swap 
        first_index, second_index = rand.sample(range(0, NUM_NODES - 1), k=2)

        # swap the positions
        mutated_solution[first_index], mutated_solution[second_index] = (
            mutated_solution[second_index],
            mutated_solution[first_index],
        )

        samples[i] = mutated_solution

    # combine the mutated solutions back into the rest of the population
    return samples + solutions[num_selections:]

# reverse a subsection of the solution
def rsm_mutation(solutions, rate):

    # create a sample set based on the rate
    num_selections = int(len(solutions) * rate)
    rand.shuffle(solutions)
    samples = solutions[:num_selections]

    for i in range(num_selections):

        mutated_solution = samples[i].copy()
        
        # i < j
        upper_bound = rand.randint(1, NUM_NODES - 1)
        lower_bound = rand.randint(0, upper_bound - 1)

        # reverse the section of the path
        mutated_solution[lower_bound:upper_bound] = mutated_solution[lower_bound:upper_bound][::-1]

        samples[i] = mutated_solution

    # combine the mutated samples back into the population
    return samples + solutions[num_selections:]

# reverse a subsection of the solution
def psm_mutation(solutions, rate):

    # create a sample set based on the rate
    num_selections = int(len(solutions) * rate)
    rand.shuffle(solutions)
    samples = solutions[:num_selections]

    for i in range(num_selections):

        mutated_solution = samples[i].copy()
        
        # i < j
        upper_bound = rand.randint(1, NUM_NODES - 1)
        lower_bound = rand.randint(0, upper_bound - 1)

        # reverse the section of the path
        rand.shuffle(mutated_solution[lower_bound:upper_bound])

        samples[i] = mutated_solution

    # combine the mutated samples back into the population
    return samples + solutions[num_selections:]

# combines halves of solutions
def one_point_crossover(solutions, rate):

    solutions_sorted = sorted(solutions, key=fitness)
    crossed_solutions = solutions_sorted

    # crossover the best pairs of solutions
    for i in range(0, int(SOLUTION_POOL_SIZE / 2), 2):
        # enforce crossover rate
        if rand.random() >= rate:
            continue
        
        parents = [solutions_sorted[i], solutions_sorted[i + 1]]

        # pick a random point
        crossover_point = rand.randint(0, NUM_NODES)
        temp = parents[0]
        parents[0] = parents[0][:crossover_point] + parents[1][crossover_point:]
        parents[1] = temp[:crossover_point] + parents[1][crossover_point:]

        crossed_solutions[i] = parents[0]
        crossed_solutions[i + 1] = parents[1]

    return crossed_solutions

def pmx_crossover(parents, num_offspring, rate):
    
    samples = parents.copy()

    offspring = []

    # create num_offspring offspring
    while len(offspring) < num_offspring:
        p1, p2 = rand.sample(samples, k=2)  # get two random nodes

        # enforce crossover rate
        if rand.random() >= rate:
            continue

        # pick a cut in each half
        first_cut, second_cut = sorted([rand.randint(0, NUM_NODES), rand.randint(0, NUM_NODES)])

        # create offspring
        # two parents can produce two offspring
        o1 = [-1 for _ in range(len(p1))]
        o2 = [-1 for _ in range(len(p2))]

        # map middles
        o1[first_cut:second_cut] = p2[first_cut:second_cut]
        o2[first_cut:second_cut] = p1[first_cut:second_cut]

        unmapped_idexes = list(range(first_cut)) + list(range(second_cut, len(p1)))

        # find and fix conflicts
        for j in unmapped_idexes:
            if p1[j] not in o1:
                o1[j] = p1[j]
            else:
                # find a value in o2 we have not replaced
                counter = 0
                while o2[counter] in o1:
                    counter += 1
                o1[j] = o2[counter]

        offspring.append(o1)

        for j in unmapped_idexes:
            if p2[j] not in o2:
                o2[j] = p2[j]
            else:
                # find a value in o1 we have not replaced
                counter = 0
                while o1[counter] in o2:
                    counter += 1
                o2[j] = o1[counter]

        offspring.append(o2)

    # account for creating an extra offspring and return
    return offspring[:num_offspring]

def cycle_crossover(parents, num_offsrping, rate):

    samples = parents.copy()

    offspring = []

    while len(offspring) < num_offsrping:
        p1, p2 = rand.sample(samples, k=2)  # get two random nodes

        # enforce crossover rate
        if rand.random() >= rate:
            continue

        # create offspring
        # two parents can produce two offspring
        o1 = [-1 for _ in range(len(p1))]
        o2 = [-1 for _ in range(len(p2))]

        first = p1[0]
        o1[0] = first
        next_value = p2[0]
        next_index = p1.index(next_value)

        # run cycle
        while next_value != first:
            o1[next_index] = next_value
            next_value = p2[next_index]
            next_index = p1.index(next_value)

        # fill in remaining values
        for i in range(len(o1)):
            if o1[i] < 0:
                o1[i] = p1[i]

        offspring.append(o1)

        first = p2[0]
        o2[0] = first
        next_value = p1[0]
        next_index = p2.index(next_value)

        # run cycle
        while next_value != first:
            o2[next_index] = next_value
            next_value = p1[next_index]
            next_index = p2.index(next_value)

        # fill in remaining values
        for i in range(len(o2)):
            if o2[i] < 0:
                o2[i] = p2[i]

        offspring.append(o2)


    return offspring[:num_offsrping]

def order_crossover(parents, num_offspring, rate):

    samples = parents.copy()

    offspring = []

    # create num_offspring offspring
    while len(offspring) < num_offspring:
        p1, p2 = rand.sample(samples, k=2)  # select a random pair

        # enforce crossover rate
        # 1- crossover_rate percentage of randomly selected pairs are ignored
        if rand.random() >= rate:
            continue

        # pick two cuts
        first_cut, second_cut = sorted([rand.randint(0, NUM_NODES), rand.randint(0, NUM_NODES)])

        # create offspring
        # two parents can produce two offspring
        o1 = [-1 for _ in range(len(p1))]
        o2 = [-1 for _ in range(len(p2))]

        # map middles
        o1[first_cut:second_cut] = p1[first_cut:second_cut]
        o2[first_cut:second_cut] = p2[first_cut:second_cut]

        unfilled_indexes = list(range(first_cut)) + list(range(second_cut, len(o1)))

        # use a two-counter approach
        # keeping track of two separate indexes
        pCounter = 0
        oCounter = 0

        while oCounter < len(unfilled_indexes):
            while p2[pCounter] in o1:
                pCounter += 1
            o1[unfilled_indexes[oCounter]] = p2[pCounter]
            oCounter += 1

        offspring.append(o1)

        pCounter = 0
        oCounter = 0

        while oCounter < len(unfilled_indexes):
            while p1[pCounter] in o2:
                pCounter += 1
            o2[unfilled_indexes[oCounter]] = p1[pCounter]
            oCounter += 1

        offspring.append(o2)
    
    # account for creating an extra offspring and return
    return offspring[:num_offspring]


# fitness function
def fitness(solution):
    total = 0
    
    # add the distance between each node from left to right in the path
    for i in range(len(solution) - 1):
        total += problem.get_weight(solution[i], solution[i + 1])

    # go back to the start
    total += problem.get_weight(solution[len(solution) - 1], solution[0])

    return total


# global variables for convergence check
current_best = -1
runs_since_update = 0

def check_for_convergence(fitnesses, n):
    global current_best
    global runs_since_update
    
    # we want to perform at least n generations
    # to give the program a chance to converge
    if(len(fitnesses) < n):
        return False

    last_n_runs = fitnesses[-n:]    # get the last n runs in a list
    min_last_n_runs = min(last_n_runs)  # get the minimum (best) fitness in that period

    runs_since_update += 1

    if current_best < 0:
        # current_best not set; set it
        current_best = min_last_n_runs
    elif min_last_n_runs < current_best:
        # new best; update it and reset the counter
        current_best = min_last_n_runs
        runs_since_update = 0
    elif runs_since_update > n:
        # stopping condition; the GA has stagnated
        current_best = -1
        runs_since_update = 0
        return True

    return False

# main program loop
def run_ga(mutation_rate, crossover_rate, tournament_size, num_parents, convergence_check_length, safe=False):

    # initialise solutions
    solution_pool = initialise_population()

    average_fitness_at_i = []

    # start timer
    start_time = time.time()

    # limit to MAX_ITERATIONS generations
    for i in range(MAX_ITERATIONS):

        # select parents

        parents = tournament(solution_pool, num_parents, tournament_size)

        if safe:
            assert len(parents) == num_parents

            for sol in parents:
                assert sorted(sol) == list(problem.get_nodes()), f"failed after selection"

        # crossover to produce offspring

        offspring = order_crossover(parents, SOLUTION_POOL_SIZE - num_parents, crossover_rate)

        if safe:
            assert len(offspring) == SOLUTION_POOL_SIZE - num_parents

            for sol in offspring:
                assert sorted(sol) == list(problem.get_nodes()), f"failed after crossover"

        # combine parents and offspring back into the solution pool
        solution_pool = parents + offspring

        # mutation

        solution_pool = rsm_mutation(solution_pool, mutation_rate)
        
        if safe:
            assert len(solution_pool) == SOLUTION_POOL_SIZE

            for sol in solution_pool:
                assert sorted(sol) == list(problem.get_nodes()), f"failed after mutation" 

        # calculate fitness

        average_fitness = 0
        for s in solution_pool:
            average_fitness += fitness(s)

        average_fitness /= SOLUTION_POOL_SIZE

        print(i, average_fitness, end='\r')

        average_fitness_at_i.append(average_fitness)    # record the average fitness at this iteration

        # check for convergence
        if check_for_convergence(average_fitness_at_i, convergence_check_length):
            break

    return solution_pool, average_fitness_at_i, (time.time() - start_time)


# function to iterate over many different combinations of parameters
def run_grid_search():
    global SOLUTION_POOL_SIZE

    population_sizes = [100, 250, 500, 1000]

    # initialise data sets

    data_for_p = [[] for _ in range(len(population_sizes))]

    for p in range(len(population_sizes)):              # population size
        SOLUTION_POOL_SIZE = population_sizes[p]

        for m in [0.1, 0.3, 0.5]:           # mutation rate
            for c in [0.5, 0.75, 1.0]:                   # crossover rate
                # for s in [0.2, 0.3, 0.4, 0.5]:
                for s in [0.2]:
                    print(f"population: {population_sizes[p]}; mutation: {m}; crossover: {c}; parent percentage: {s}")

                    # run the program with the parameters
                    final_solution_pool, final_fitnesses, time = run_ga(m, c, 3, int(SOLUTION_POOL_SIZE * s), 50)

                    # print out statistics
                    best = min(final_solution_pool, key=fitness)
                    best_fitness = fitness(best)
                    num_iterations = len(final_fitnesses)

                    print(f"best path: {best}")
                    print(f"fitness: {best_fitness}")
                    print(f"number of iterations: {num_iterations}")
                    print(f"time: {time}")

                    # save results
                    data_for_p[p].append((population_sizes[p], m, c, s, best_fitness, time, num_iterations))

    # write out data
    with open("results.csv", "x", newline="") as file:
        writer = csv.writer(file)

        for p in range(len(population_sizes)):
            writer.writerows(data_for_p[p])

    # print out best param combinations
    for p in range(len(population_sizes)):
        # extract fitness values
        fitness_for_p_at_combination = [f for _, _, _, _, f, _, _ in data_for_p[p]]

        # extract parameter combinations
        parameters_for_p_at_combinatin = [(m, c, s) for _, m, c, s, _, _, _ in data_for_p[p]]

        min_path_length = min(fitness_for_p_at_combination)
        best_param_combinations = []

        # print the parameter combination if it produced the best fitness
        for i in range(len(fitness_for_p_at_combination)):
            if fitness_for_p_at_combination[i] == min_path_length:
                best_param_combinations.append(parameters_for_p_at_combinatin[i])

        print(f"best parameters for population: {population_sizes[p]}: {best_param_combinations}")

# function to run a single combination of paramters
def run_single():
    final_solution_pool, final_fitnesses, time = run_ga(0.1, 1.0, 3, int(SOLUTION_POOL_SIZE * 0.2), 50, True)

    best = min(final_solution_pool, key=fitness)
    print(f"best path: {best}")
    print(f"fitness: {fitness(best)}")
    print(f"number of iterations: {len(final_fitnesses)}")
    print(f"time: {time} seconds")

    plt.plot(range(len(final_fitnesses)), final_fitnesses)
    plt.show()


## RUN THE PROGRAM ##

# run_grid_search()
run_single()
