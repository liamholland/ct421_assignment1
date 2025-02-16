# CT421

### Assignment 1

This project was all about using genetic algorithms to find optimal solutions for the travelling salesman problem.

The genetic algorithm is written as a Python script.

### Usage

1. Set the problem at line 7
```Python
problem = tsp.load('tsp_files\\berlin52.tsp')
```

2. Set the maximum number of generations at line 11
```Python
MAX_ITERATIONS = 1500
```

##### Single Run

3. Set the population at line 9
```Python
SOLUTION_POOL_SIZE = 1000
```

4. Set parameters at line 479
```
final_solution_pool, final_fitnesses, time = run_ga(0.1, 1.0, 3, int(SOLUTION_POOL_SIZE * 0.2), 50, True)
```
The order of parameters is mutation rate, crossover rate, tournament size, number of parents brought to next generation, number of generations in convergence check and debugging on flag

5. Uncomment the `run_single()` function at line 494
```Python
# run_grid_search()
run_single()
```

##### Run Grid Search

3. Set the population sizes at line 421
```Python
population_sizes = [100, 250, 500, 1000]
```

4. Set the mutation and crossover rates and others at line 430/431/etc.
```Python
for m in [0.1, 0.3, 0.5]:
    for c in [0.5, 0.75, 1.0]:
```
Other parameters can be added in other loops, such as parent set size `s`

5. Uncomment the `run_grid_search()` function at line 494
```Python
run_grid_search()
# run_single()
```

##### Both

7. Set the operators to use
- Selection is at line 367
    - Options are `tournament`, `roulette_wheel` and `elitism`
- Crossover is at line 377
    - Options are `one_point_crossover`, `pmx_crossover`, `cycle_crossover` and `order_crossover`
- Mutation is at line 390
    - Options are `swap_mutation`, `rsm_mutation` and `psm_mutation`


8. Run the program
```sh
python3 solver.py
```

### Output

The program outputs the current generation and average fitness while it is running. When complete, it prints the statistics to the screen. If running grid search, all of this data will be saved to a file called `results.csv`, as well as the best parameters being printed to the screen. Running a single run of parameters will also show a graph plotting fitness over generations on completion 