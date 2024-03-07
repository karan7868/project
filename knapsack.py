
import mlrose_hiive as ml
import random
import datetime
import timeit
import numpy as np
from utils import Utils
from mlrose_hiive.generators import KnapsackGenerator

# Create the Knapsack Generator
OUTPUT_DIRECTORY = './Folder/Knapsack'
item_types = 39
max_items = 15
iterations_range = [2, 4, 8, 16, 32, 64, 128, 256] #512, 1024, 2048]
restart_list=[0, 5, 25, 75]
population_sizes=[200, 300, 400, 500],
mutation_rates=[0.2, 0.4, 0.6, 0.8, 1]
temperature_list=[1, 10, 50, 100, 250, 500, 1000,2500, 5000, 10000]
keep_percent_list=[0.1, 0.25, 0.5, 0.75]
print(f'Knapsack Value Optimization for {item_types} item_types and {max_items} max_items')
# Generate the fitness problem and the optimization function
problem = KnapsackGenerator.generate(
                                    seed=random.seed(77), 
                                     number_of_items_types=item_types,
                                     max_item_count=max_items,
                                     max_weight_per_item=4, 
                                     max_value_per_item=4,
                                     max_weight_pct=0.5
                                     )


# Run the models and plot them
rhc_stats, rhc_curves, rhc_ideal = Utils.rhs_optimization_params(problem, iterations_range, restart_list)
ga_stats, ga_curves, popsize_sa_ideal, mutation_rate_ideal = Utils.opt_sa_params(problem, iterations_range, temperature_list)
sa_stats, sa_curves, initial_temp_ideal = Utils.opt_ga_params(problem, population_sizes,mutation_rates )
mmc_stats, mmc_curves, prcnt_ideal, popsize_mimic_ideal = Utils.opt_mimic_params(problem, iterations_range,population_sizes,keep_percent_list)


rhc_state = []
rhc_fitness = []
rhc_time = []
for iter in iterations_range:
    start_time = timeit.default_timer()
    best_state, best_fitness, curve = ml.random_hill_climb(problem=problem, 
                                                           max_iters=iter, 
                                                           max_attempts=1000, 
                                                           restarts=rhc_ideal, 
                                                           curve=True
                                                           )
    end_time = timeit.default_timer()
    convergence_time = (end_time - start_time) # seconds
    rhc_state.append(best_state)
    rhc_fitness.append(best_fitness)
    rhc_time.append(convergence_time)
    print('The fitness at the best state found using Random Hill Climbing is: ', max(rhc_fitness))
    
ideal_pop_size = 500 
ideal_mutation_rate = 0.2

ga_state = []
ga_fitness = []
ga_time = []
for iter in iterations_range:
    print(f"Genetic Algorithm with iteration range = {iter}")
    start_time = timeit.default_timer()
    best_state, best_fitness, curve = ml.genetic_alg(problem=problem,curve=True,
                                                     mutation_prob = ideal_mutation_rate,
                                                     max_attempts = 1000, 
                                                     max_iters = iter,
                                                     pop_size=ideal_pop_size
                                                     )
    end_time = timeit.default_timer()
    convergence_time = (end_time - start_time) # seconds
    ga_state.append(best_state)
    ga_fitness.append(best_fitness)
    ga_time.append(convergence_time)
print('The fitness at the best route found using genetic algorithms is: ',max(ga_fitness))


ideal_temp = 1000
sa_state = []
sa_fitness = []
sa_time = []
for iter in iterations_range:
    start_time = timeit.default_timer()
    best_state, best_fitness, curve = ml.simulated_annealing(problem=problem, 
                                                             max_attempts = 1000,
                                                             max_iters = iter,
                                                             curve=True,
                                                             schedule=ml.GeomDecay(init_temp=ideal_temp)
                                                            )
    end_time = timeit.default_timer()
    convergence_time = (end_time - start_time) # seconds
    sa_state.append(best_state)
    sa_fitness.append(best_fitness)
    sa_time.append(convergence_time)
    print('The fitness at the best state found using simulated annealing is: ',
    max(sa_fitness))
    
popsize_mimic_ideal = 500 # this came from the results of the experiment commented out, above.
prcnt_ideal = 0.5 # this came from the results of the experiment commented out, above.
mimic_state = []
mimic_fitness = []
mimic_time = []
for iter in iterations_range:
    start_time = timeit.default_timer()
    best_state, best_fitness, curve = ml.mimic(problem=problem,
                                               keep_pct=prcnt_ideal,
                                               max_attempts=1000, 
                                               max_iters=iter,
                                               pop_size=popsize_mimic_ideal, 
                                               curve=True
                                               )
    end_time = timeit.default_timer()
    convergence_time = (end_time - start_time) # seconds
    mimic_state.append(best_state)
    mimic_fitness.append(best_fitness)
    mimic_time.append(convergence_time)
    # print('The fitness at the best state found using MIMIC is: ',max(mimic_best_fitness))
    
Utils.draw_graphs_comparsion_algorithms('Knapsack', 
                                        max_items,
                                        iterations_range, 
                                        rhc_fitness, 
                                        ga_fitness, 
                                        sa_fitness, 
                                        mimic_fitness,
                                        rhc_time,
                                        ga_time, 
                                        sa_time, 
                                        mimic_time
                                        )








