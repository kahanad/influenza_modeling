import numpy as np
import pandas as pd
# import model
from model import vaccination_model_with_age as model
import itertools
import multiprocessing as mp
import pickle

# Prepare network and data
prep_data = model.data_and_network_prep()


def limits(x):
    return min(max(0, x), 1)


# Random parameters to check
# parameters_grid = [{'beta': limits(np.random.normal(0.0163, 0.0163/4)),
#                     'delta': limits(np.random.normal(0.28, 0.28/3)),
#                     'gamma': np.random.uniform(1/6, 1/2),
#                     'I_0_size': np.random.choice([0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0008, 0.0009, 0.001])}
#                    for i in range(10000)]

parameters_grid = [{'beta': limits(np.random.normal(0.000008, 0.000008/20)),
                    'delta': limits(np.random.normal(0.55, 0.55/8)),
                    'gamma': np.random.uniform(1/6, 1/2),
                    'I_0_size': np.random.choice([0.0005,  0.001])}
                   for i in range(1000)]


# Save current grid
with open(f'../../data/vaccination_model/grid_search/grid_search_homo.pickle', 'wb') as pickle_out:
    pickle.dump(parameters_grid, pickle_out)

# Create data for fit
data_for_fit = model.create_data_for_fit(prep_data)


# Define a function for multiprocessing
def run_model_mp(parameters):
    print(mp.current_process())

    # Calculate and save the aggregated log-likelihood - total, children and adult (median of m runs)
    # likelihoods = []
    # likelihoods_by_age = []
    likelihoods_by_subdist = []

    # Number of simulations
    m = 11

    for i in range(m):  # TODO: CHANGE TO 11
        # Run model and get results
        # model_results = model.run_model(parameters, prep_data)  # TODO: CHANGE BACK TO NETWORK MODEL
        model_results = model.run_model(parameters, prep_data, homogenous=True)

        # Calculate log likelihood
        # log_likelihood = model.log_likelihood(model_results['lambdas'], data_for_fit['data_for_fit'])
        # log_likelihood_age = model.log_likelihood_agg_with_age(model_results, data_for_fit)
        log_likelihood_age_subdist = model.log_likelihood_agg_by_subdist(model_results['lambdas'],
                                                                   data_for_fit['data_for_fit_subdist'], prep_data)

        # Add to the lists
        # likelihoods.append(log_likelihood)
        # likelihoods_by_age.append(log_likelihood_age)
        likelihoods_by_subdist.append(log_likelihood_age_subdist)

    # Calculate medians
    # med_likelihood = np.argsort(np.array(likelihoods))[len(likelihoods)//2]
    # med_likelihood_age = np.argsort(np.array(likelihoods_by_age))[len(likelihoods_by_age)//2]
    med_likelihood_subdist = np.argsort(np.array(likelihoods_by_subdist))[len(likelihoods_by_subdist)//2]

    # Return parameters, log-likelihoods and MSEs
    return {'parameters': parameters, 'log_likelihood_subdist': likelihoods_by_subdist[med_likelihood_subdist]}
            # 'log_likelihood': likelihoods[med_likelihood],
            # 'log_likelihood_age': likelihoods_by_age[med_likelihood_age],

def mp_handler():
    # Create a pool of processes
    pool = mp.Pool(28)
    # Process in parallel
    results = pool.map(run_model_mp, parameters_grid)
    return results


if __name__ == '__main__':
    res = mp_handler()

    # Saving the results
    with open(f'../../data/vaccination_model/grid_search/grid_search_res_homo.pickle', 'wb') as pickle_out:
        pickle.dump(res, pickle_out)
