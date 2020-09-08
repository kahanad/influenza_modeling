import numpy as np
import pandas as pd
# import model
import model
import itertools
import multiprocessing as mp
import pickle

# Prepare network and data
prep_data = model.data_and_network_prep()


def limits(x):
    return min(max(0, x), 1)


# Season for grid search
season = 2016
# parameters_grid = [{season: {'beta': limits(np.random.normal(0.00155, 0.00155/3)),
#                     'delta': 1,
#                     'phi': (-1*np.random.uniform(20, 45)/52)*2*np.pi,
#                     'epsilon': 1}}
#                    for i in range(1000)]

# Homogenous
parameters_grid = [{season: {'beta': limits(np.random.normal(0.00000067, 0.00000067/3.5)),
                    'delta': 1,
                    'phi': (-1*np.random.uniform(20, 45)/52)*2*np.pi,
                    'epsilon': 1}}
                   for i in range(3000)]

# Save current grid
with open(f'../../data/influenza_model/results/homo_grid_search_2016_1_new.pickle', 'wb') as pickle_out:
    pickle.dump(parameters_grid, pickle_out)

# Create data for fit
data_for_fit = model.create_data_for_fit_influenza()


# Define a function for multiprocessing
def run_model_mp(parameters):
    print(mp.current_process())

    # Calculate and save the aggregated log-likelihood - total, children and adult (median of 5 runs)
    likelihoods = []
    likelihoods_by_age = []
    likelihoods_by_subdist = []

    for i in range(5):
        # Run model and get results
        # model_results = model.run_influenza_model(parameters, prep_data, season)  # TODO: CHANGE BACK TO NETWORK MODEL
        model_results = model.run_influenza_model(parameters, prep_data, season, homogenous=True)

        # Calculate log likelihood
        # By clinic and age
        log_likelihood = model.log_likelihood_influenza(model_results['lambdas'],
                                                        data_for_fit['by_clinic_age'], season=season)
        # By age
        log_likelihood_age = model.log_likelihood_agg_age_influenza(model_results, data_for_fit, season=season)

        # By subdist and age
        log_likelihood_age_subdist = model.log_likelihood_agg_by_subdist_influenza(model_results['lambdas'],
                                                                                   data_for_fit['by_subdist'],
                                                                                   season, prep_data)

        # Add to the lists
        likelihoods.append(log_likelihood)
        likelihoods_by_age.append(log_likelihood_age)
        likelihoods_by_subdist.append(log_likelihood_age_subdist)

    # Calculate medians
    med_likelihood = np.argsort(np.array(likelihoods))[len(likelihoods)//2]
    med_likelihood_age = np.argsort(np.array(likelihoods_by_age))[len(likelihoods_by_age)//2]
    med_likelihood_subdist = np.argsort(np.array(likelihoods_by_subdist))[len(likelihoods_by_subdist)//2]

    # Return parameters, log-likelihoods and MSEs
    return {'parameters': parameters,
            'log_likelihood': likelihoods[med_likelihood],
            'log_likelihood_age': likelihoods_by_age[med_likelihood_age],
            'log_likelihood_subdist': likelihoods_by_subdist[med_likelihood_subdist]}


def mp_handler():
    # Create a pool of processes
    pool = mp.Pool(24)
    # Process in parallel
    results = pool.map(run_model_mp, parameters_grid)
    return results


if __name__ == '__main__':
    res = mp_handler()

    # Saving the results
    with open(f'../../data/influenza_model/results/homo_grid_search_2016_1_new_res.pickle', 'wb') as pickle_out:
        pickle.dump(res, pickle_out)
