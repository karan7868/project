import matplotlib.pyplot as plt
import mlrose_hiive as ml
import numpy as np
import matplotlib
import random
matplotlib.use('TkAgg')

OUTPUT_DIRECTORY = './Experiments/Knapsack'

class Utils:
    
    @staticmethod
    def draw_graphs_comparsion_algorithms(algorithm, max_items, iterations_range, rhc_fitness_curve, ga_fitness_curve, sa_fitness_curve, mimic_fitness_curve,
                                        rhc_convergence_time, ga_convergence_time, sa_convergence_time, mimic_convergence_time):
        #======= Comparison of all four optimization algorithms ==========#
        fig5, (row, col) = plt.subplots(1, 2, figsize=(15, 5))
        fig5.suptitle(f'Comparing Random Search Optimizers on {algorithm}: Fitness and Convergence Time' %(max_items))
        row.set(xlabel="Number of Iterations", ylabel="Fitness")
        row.grid()
        row.plot(iterations_range, (rhc_fitness_curve), 'o-', color="r", label='Random Hill Climbing')
        row.plot(iterations_range, (sa_fitness_curve), 'o-', color="m", label='Simulated Annealing')
        row.plot(iterations_range, (mimic_fitness_curve), 'o-', color="g", label='MIMIC')
        row.plot(iterations_range, (ga_fitness_curve), 'o-', color="b", label='Genetic Algorithms')
        row.legend(loc="best")
        col.set(xlabel="Iterations", ylabel="Convergence Time (seconds)")
        col.grid()
        col.plot(iterations_range, rhc_convergence_time, 'o-', color="r", label='Random Hill Climbing')
        col.plot(iterations_range, sa_convergence_time, 'o-', color="m",label='Simulated Annealing')
        col.plot(iterations_range, mimic_convergence_time, 'o-', color="g", label='MIMIC')
        col.plot(iterations_range, ga_convergence_time, 'o-', color="b", label='Genetic Algorithms')
        col.legend(loc="best")
        plt.show()
    
    @staticmethod
    # Create the optimization for the algorithms
    def rhs_optimization_params(problem, iteration_list, restart_list):
        rhc = ml.runners.RHCRunner(problem=problem, 
                                experiment_name='Knapsack - OptimalParams - RHC',
                                output_directory=OUTPUT_DIRECTORY, 
                                seed=random.seed(77),
                                iteration_list=iteration_list, 
                                max_attempts=1000,
                                restart_list=restart_list
                                )
        
        print("Starting the rhc.run()")
        rhc_df_run_stats, rhc_df_run_curves = rhc.run()
        ideal_rs = rhc_df_run_stats[['current_restart']].iloc[rhc_df_run_stats[['Fitness']].idxmax()]
        return rhc_df_run_stats, rhc_df_run_curves, ideal_rs

    @staticmethod
    # Optimize the algorithm parameters (Manual)
    def opt_ga_params(problem, iteration_list, population_sizes, mutation_list):
        ga = ml.runners.GARunner(problem=problem, experiment_name='Knapsack - OptimalParams - GA',
                                output_directory=OUTPUT_DIRECTORY,seed=random.seed(77),
                                max_attempts=1000, population_sizes=population_sizes,
                                mutation_rates=mutation_list)
        print("Starting the genetic algorithm")
        stats, curves = ga.run()
        pop_size = stats[['Population Size']].iloc[stats[['Fitness']].idxmax()] # from the output of theexperiment above
        mutation_rate = stats[['Mutation Rate']].iloc[stats[['Fitness']].idxmax()] # from the output of theexperiment above
        return stats, curves, pop_size, mutation_rate

    @staticmethod
    def opt_sa_params(problem, iteration_list, temp_list):
        sa = ml.runners.SARunner(problem=problem, experiment_name='Knapsack - OptimalParams - SA',
                                output_directory=OUTPUT_DIRECTORY, seed=random.seed(77),
                                iteration_list=iteration_list, max_attempts=1000,
                                temperature_list=temp_list
                                )
        stats, curves = sa.run()
        temp = stats[['Temperature']].iloc[stats[['Fitness']].idxmax()] #from the output of the experiment above
        return stats, curves, temp

    @staticmethod
    def opt_mimic_params(problem, iterations, population_size, keep_percent_list):
        mmc = ml.runners.MIMICRunner(problem=problem, 
                                    experiment_name='Knapsack - OptimalParams - MIMIC',
                                    output_directory=OUTPUT_DIRECTORY,
                                    seed=random.seed(77),
                                    iteration_list=iterations,
                                    max_attempts=1000, 
                                    population_sizes=population_size,
                                    keep_percent_list=keep_percent_list)

        stats, curves = mmc.run()
        percent = stats[['Keep Percent']].iloc[stats[['Fitness']].idxmax()] # from the output of the experiment above
        pop_size = stats[['Population Size']].iloc[stats[['Fitness']].idxmax()] # from the output of the experiment above
        return stats, curves, percent, pop_size



